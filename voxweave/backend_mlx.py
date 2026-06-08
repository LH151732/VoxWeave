"""Apple-Silicon (MLX) backend for the local Qwen3 pipeline.

On macOS / MPS the PyTorch Qwen3-ASR + Qwen3-ForcedAligner are served instead by their native
MLX ports from `mlx-audio` (https://github.com/Blaizzy/mlx-audio): purpose-built Metal kernels +
4/8-bit quantization, faster and lower-memory than running the torch models through the MPS
backend. The two models cover the same 11 languages as the torch aligner, so on this backend ALL
alignment (incl. ja/CJK and en, which the torch path routes to MMS/wav2vec2 CTC) goes through the
MLX Qwen3-ForcedAligner — onnxruntime has no Metal provider, so the ONNX MMS aligner could never
run on the GPU here anyway.

Vocal separation (MelBandRoformer) and PANNs song-skip have no MLX port and stay on torch-MPS;
this module only owns ASR + forced alignment. Selection lives in voxweave.backend._use_mlx().

These adapters mirror the tiny slice of the qwen-asr / Qwen3ForcedAligner API that backend.py
calls, so the call sites in backend.py stay backend-agnostic.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from voxweave import config

log = logging.getLogger("voxweave")

# torch HF repo id -> mlx-community quantized repo id. Substring fallback handles custom ids.
_MLX_ASR_REPOS = {
    "Qwen/Qwen3-ASR-0.6B": "mlx-community/Qwen3-ASR-0.6B-8bit",
    "Qwen/Qwen3-ASR-1.7B": "mlx-community/Qwen3-ASR-1.7B-8bit",
}
_DEFAULT_MLX_ASR = "mlx-community/Qwen3-ASR-0.6B-8bit"
MLX_ALIGNER_REPO = os.environ.get(
    "VOXWEAVE_MLX_ALIGNER_REPO", "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
)

_MISSING_MLX = (
    "The MLX backend requires the voxweave[mps] install (mlx-audio + mlx). "
    "Install: `make install VARIANT=mps` (Apple Silicon/macOS only). "
    "Force the torch backend instead with VOXWEAVE_BACKEND=torch. Missing: {mod}"
)

# Process-level singletons; released by release()/release_asr() at end of episode (mirrors backend.py).
_asr = None  # _MlxAsr adapter
_asr_repo = None  # currently loaded MLX ASR repo id (reloaded on --model change)
_aligner = None  # mlx_audio forced-aligner model


def _require(mod: str) -> RuntimeError:
    return RuntimeError(_MISSING_MLX.format(mod=mod))


def _load(repo: str, cache_dir: str):
    """Download repo into cache_dir (VoxWeave's own cache, matching the torch _hf_snapshot path)
    then load from the local snapshot — mlx_audio.stt.load accepts a local dir, so it won't re-fetch
    from the hub. Keeps MLX weights under ~/.cache/voxweave alongside the separator/PANNs weights."""
    try:
        from huggingface_hub import snapshot_download
        from mlx_audio.stt import load
    except ModuleNotFoundError as e:
        raise _require(e.name or "mlx_audio") from e
    local = snapshot_download(repo, cache_dir=cache_dir)
    return load(local)


def _clear_cache() -> None:
    """Best-effort MLX Metal cache reclaim (optimization only; failure is harmless)."""
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:  # noqa: BLE001
        pass


def _mlx_asr_repo(model_id: str | None) -> str:
    """Map a torch ASR repo id (or --model value) to the mlx-community quantized repo.

    Size follows --model / VOXWEAVE_ASR_MODEL: 'Qwen/Qwen3-ASR-1.7B' (or any id containing '1.7')
    -> the 1.7B quant, else the 0.6B quant. VOXWEAVE_MLX_ASR_REPO hard-overrides everything (e.g.
    to pin a 4-bit quant or a non-standard repo), regardless of --model.
    """
    override = os.environ.get("VOXWEAVE_MLX_ASR_REPO", "").strip()
    if override:
        return override
    if model_id and model_id in _MLX_ASR_REPOS:
        return _MLX_ASR_REPOS[model_id]
    if model_id and "1.7" in model_id:
        return "mlx-community/Qwen3-ASR-1.7B-8bit"
    return _DEFAULT_MLX_ASR


class _AsrResult:
    """Mirror of qwen_asr's transcribe() result: only .language + .text are read by backend.py."""

    __slots__ = ("language", "text")

    def __init__(self, language: str | None, text: str):
        self.language = language
        self.text = text


class _MlxAsr:
    """Adapter exposing qwen-asr's `.transcribe(path, language=, return_time_stamps=, context=)`
    over mlx-audio's Qwen3-ASR `.generate()`. context maps to the model's system_prompt (best-effort
    proper-noun biasing; not identical to qwen-asr's native context= field)."""

    def __init__(self, model):
        self._m = model

    def transcribe(
        self,
        wav_path: str,
        *,
        language: str | None = None,
        return_time_stamps: bool = False,  # noqa: ARG002 -- qwen-asr arg name; MLX is text-only here
        context: str | None = None,
    ) -> list[_AsrResult]:
        from voxweave.lang import to_aligner_name

        lang = to_aligner_name(language) if language and language.strip() else None
        out = self._m.generate(
            str(wav_path), language=lang, system_prompt=context or None
        )
        lang_field = getattr(out, "language", None)
        if isinstance(lang_field, (list, tuple)):
            det = next((x for x in lang_field if x), None)
        else:
            det = lang_field or None
        return [_AsrResult(det, out.text)]


def get_asr(model_id: str | None = None):
    """Lazy-load the MLX Qwen3-ASR singleton, reloading if the requested repo changes."""
    global _asr, _asr_repo
    repo = _mlx_asr_repo(model_id)
    if _asr is not None and _asr_repo != repo:
        release_asr()
    if _asr is None:
        log.info("loading MLX ASR=%s", repo)
        _asr = _MlxAsr(_load(repo, config.ASR_CACHE))
        _asr_repo = repo
        log.info("MLX ASR ready")
    return _asr


def _get_aligner():
    """Lazy-load the MLX Qwen3-ForcedAligner singleton."""
    global _aligner
    if _aligner is None:
        log.info("loading MLX forced aligner=%s", MLX_ALIGNER_REPO)
        _aligner = _load(MLX_ALIGNER_REPO, config.ALIGN_CACHE)
        log.info("MLX forced aligner ready")
    return _aligner


def align(wav_path: Path, text: str, language: str) -> list[dict]:
    """Forced alignment via MLX Qwen3-ForcedAligner -> units [{text,start,end}].

    Covers all 11 languages (incl. ja/CJK/en), so this fully replaces the torch MMS/wav2vec2 CTC
    path on the MLX backend. language accepts ISO or full name.
    """
    from voxweave.lang import to_aligner_name

    model = _get_aligner()
    res = model.generate(
        audio=str(wav_path), text=text, language=to_aligner_name(language)
    )
    if isinstance(res, list):  # batch API returns a list; we align a single pair
        res = res[0]
    items = getattr(res, "items", None)
    if items is None:
        items = list(res)  # ForcedAlignResult is also directly iterable over its items
    units = [
        {
            "text": it.text,
            "start": float(it.start_time),
            "end": float(it.end_time),
        }
        for it in items
    ]
    _clear_cache()
    return units


def release_asr() -> None:
    """Drop the MLX ASR singleton (called between transcribe_chunks passes to cut peak memory)."""
    global _asr, _asr_repo
    _asr = None
    _asr_repo = None
    _clear_cache()


def release() -> None:
    """Drop all MLX singletons. Safe no-op when the MLX backend was never used."""
    global _aligner
    release_asr()
    _aligner = None
    _clear_cache()

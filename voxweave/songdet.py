from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from voxweave import config

log = logging.getLogger("voxweave")

# PANNs Cnn14 training SR — must feed 32k; 16k input causes a sample-rate mismatch.
SR = 32000

# Explicit local path wins; otherwise pulled from HF (-> config.AUDIO_CACHE) instead of
# panns_inference's default Zenodo ~/panns_data, keeping all weights under the HF cache root.
PANNS_CKPT = os.path.expanduser(os.environ.get("VOXWEAVE_PANNS_CKPT", ""))
PANNS_REPO = os.environ.get("VOXWEAVE_PANNS_REPO", "thelou1s/panns-inference")
PANNS_REPO_FILE = os.environ.get("VOXWEAVE_PANNS_REPO_FILE", "Cnn14_mAP=0.431.pth")

# AudioSet class indices (class_labels_indices.csv)
IDX_SPEECH = [0, 4, 6, 7]  # Speech, Conversation, Babbling, Speech synthesizer
IDX_SING = [27, 28, 29, 30, 31, 36, 37, 254, 255, 266]  # Singing/Choir/Chant/Rap/...
IDX_MUSIC = [137, 267, 268, 270]  # Music / Background / Theme / Soundtrack

# Tuned on Yofukashi no Uta ep1 separated vocals.
# Either branch (sing or music) hitting its threshold counts as "song/music".
SING_MIN = 0.15
MUSIC_MIN = 0.30
SPEECH_MAX = 0.25
# Clean-dialogue signature on separated vocals: speech dominant, almost no singing/instrumental
# residue. Used during span expansion to trim dialogue flush against the song-core edge rather
# than absorb it. Interior rap verses still carry rhythmic residue (sing >= SING_QUIET_MAX or
# music >= MUSIC_QUIET_MAX), so they do NOT match and are still absorbed (preserves pit-2 protection).
SPEECH_CLEAN_MIN = 0.5
SING_QUIET_MAX = 0.10
MUSIC_QUIET_MAX = 0.20
WIN_SEC = 2.0
HOP_SEC = 1.0
GAP_MERGE_SEC = 2.0
MIN_SPAN_SEC = 3.0
DROP_OVERLAP = (
    0.5  # VAD segment is dropped if overlap fraction with a song span >= this
)
BLOCK_GAP_SEC = 3.0  # adjacent VAD segments within this gap are treated as one voiced block (for span expansion)

_model = None  # AudioTagging singleton — lazy-loaded, reused within the process


def _resolve_panns_ckpt() -> str:
    """Return local path to Cnn14 checkpoint; explicit env path wins, else download from HF (cached)."""
    if PANNS_CKPT and os.path.exists(PANNS_CKPT):
        return PANNS_CKPT
    from huggingface_hub import hf_hub_download

    return hf_hub_download(PANNS_REPO, PANNS_REPO_FILE, cache_dir=config.AUDIO_CACHE)


def _get_model():
    global _model
    if _model is None:
        import torch
        from panns_inference import AudioTagging

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Pass checkpoint explicitly so panns_inference never falls back to its ~/panns_data Zenodo download.
        ckpt = _resolve_panns_ckpt()
        _model = AudioTagging(checkpoint_path=ckpt, device=device)
        log.info("PANNs Cnn14 loaded on %s (%s)", device, ckpt)
    return _model


def reduce_scores(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(n, 527) probabilities → ``(speech, sing, music)`` per-window maxima.

    Extracted so tests can store just these three small arrays and drive
    ``*_from_scores`` variants without running PANNs or a GPU."""
    return (
        probs[:, IDX_SPEECH].max(axis=1),
        probs[:, IDX_SING].max(axis=1),
        probs[:, IDX_MUSIC].max(axis=1),
    )


def song_flags_from_scores(
    speech: np.ndarray,
    sing: np.ndarray,
    music: np.ndarray,
    *,
    sing_min: float = SING_MIN,
    music_min: float = MUSIC_MIN,
    speech_max: float = SPEECH_MAX,
) -> np.ndarray:
    """Three per-window score arrays → boolean: True = song/music. Pure function (core criterion of song_flags)."""
    return ((sing > speech) & (sing > sing_min)) | (
        (music > music_min) & (speech < speech_max)
    )


def sing_flags_from_scores(
    speech: np.ndarray,
    sing: np.ndarray,
    music: np.ndarray,
    *,
    sing_min: float = SING_MIN,
) -> np.ndarray:
    """Three per-window score arrays → boolean: True = contains singing (sing branch only, excludes pure instrumental). Pure function."""
    return (sing > speech) & (sing > sing_min)


def song_flags(
    probs: np.ndarray,
    *,
    sing_min: float = SING_MIN,
    music_min: float = MUSIC_MIN,
    speech_max: float = SPEECH_MAX,
) -> np.ndarray:
    """Per-window (n, 527) probabilities → boolean array: True = song/music, False = speech/silence. Pure function."""
    return song_flags_from_scores(
        *reduce_scores(probs),
        sing_min=sing_min,
        music_min=music_min,
        speech_max=speech_max,
    )


def sing_flags(
    probs: np.ndarray,
    *,
    sing_min: float = SING_MIN,
) -> np.ndarray:
    """(n, 527) → boolean: True = singing/rap/chant dominant (excludes pure instrumental).

    Only spans that pass this test trigger voiced-block expansion (to catch rap verses
    PANNs misclassifies as Speech). Pure-instrumental BGM (music dominant, sing~0) does
    NOT trigger expansion, preventing adjacent dialogue from being absorbed."""
    return sing_flags_from_scores(*reduce_scores(probs), sing_min=sing_min)


def speech_flags_from_scores(
    speech: np.ndarray,
    sing: np.ndarray,
    music: np.ndarray,
    *,
    speech_min: float = SPEECH_CLEAN_MIN,
    sing_max: float = SING_QUIET_MAX,
    music_max: float = MUSIC_QUIET_MAX,
) -> np.ndarray:
    """Per-window scores → boolean: True = clean dialogue (speech dominant, minimal singing/instrumental).

    Used during expansion edge-trimming: genuine dialogue (sing~0/music~0.03) at the song
    boundary is trimmed rather than absorbed. Interior rap verses carry residue and do NOT
    match, so they are still absorbed (preserves pit-2 protection)."""
    return (speech > speech_min) & (sing < sing_max) & (music < music_max)


def speech_flags(
    probs: np.ndarray,
    *,
    speech_min: float = SPEECH_CLEAN_MIN,
    sing_max: float = SING_QUIET_MAX,
    music_max: float = MUSIC_QUIET_MAX,
) -> np.ndarray:
    """Per-window (n, 527) → boolean: True = clean dialogue. Pure function. See speech_flags_from_scores."""
    return speech_flags_from_scores(
        *reduce_scores(probs),
        speech_min=speech_min,
        sing_max=sing_max,
        music_max=music_max,
    )


def merge_spans(
    flags: np.ndarray,
    starts: list[float],
    *,
    win_sec: float = WIN_SEC,
    gap_merge: float = GAP_MERGE_SEC,
    min_span: float = MIN_SPAN_SEC,
) -> list[tuple[float, float]]:
    """Consecutive flagged windows → time spans [(start, end)]; gaps <= gap_merge are merged, spans shorter than min_span are dropped. Pure function."""
    spans: list[list[float]] = []
    cur: list[float] | None = None
    for flag, t in zip(flags, starts, strict=True):
        if not flag:
            continue
        if cur is None:
            cur = [t, t + win_sec]
        elif t - cur[1] <= gap_merge:
            cur[1] = t + win_sec
        else:
            spans.append(cur)
            cur = [t, t + win_sec]
    if cur is not None:
        spans.append(cur)
    return [(a, b) for a, b in spans if b - a >= min_span]


def drop_segments_in_spans(
    segments: list[dict],
    spans: list[tuple[float, float]],
    *,
    overlap: float = DROP_OVERLAP,
) -> list[dict]:
    """Drop any VAD speech segment whose overlap fraction with any song span is >= overlap; return the kept segments. Pure function."""
    if not spans:
        return segments
    kept = []
    for seg in segments:
        dur = seg["end"] - seg["start"]
        if dur <= 0:
            kept.append(seg)
            continue
        ov = 0.0
        for a, b in spans:
            ov += max(0.0, min(seg["end"], b) - max(seg["start"], a))
        if ov / dur < overlap:
            kept.append(seg)
    return kept


def filter_short_spans(
    spans: list[tuple[float, float]], *, min_sec: float
) -> list[tuple[float, float]]:
    """Drop song spans shorter than ``min_sec``.

    Real OP/ED segments (30-90s) far exceed any sane threshold. A brief BGM burst that
    slips through would cause group_segments_by_spans to split the audio there, turning
    the second half into a BGM-dominated chunk that breaks Qwen ASR recall. Apply after
    voiced-block expansion (singing OPs are already stretched long by then)."""
    return [(a, b) for (a, b) in spans if b - a >= min_sec]


def _overlaps(seg: dict, spans: list[tuple[float, float]]) -> bool:
    return any(max(seg["start"], a) < min(seg["end"], b) for a, b in spans)


def expand_spans_to_voiced_blocks(
    segments: list[dict],
    spans: list[tuple[float, float]],
    *,
    expandable: list[tuple[float, float]] | None = None,
    protect: list[tuple[float, float]] | None = None,
    block_gap: float = BLOCK_GAP_SEC,
) -> list[tuple[float, float]]:
    """Absorb entire voiced blocks that overlap a song span; return expanded and merged spans.

    Adjacent VAD segments within block_gap form one voiced block. OP/ED sequences are often
    a continuous "rap verse -> singing chorus" block: PANNs detects only the chorus, but
    anchoring on it pulls the whole block (rap included) into the song span.

    ``expandable`` (default None = all spans may expand) restricts which spans trigger
    whole-block absorption — typically singing spans from :func:`sing_flags`. Pure-instrumental
    BGM spans are excluded here; without this, a BGM cue followed immediately by speech
    (gap < block_gap) would absorb the speech and drop its subtitles.

    ``protect`` (default None = no trimming) lists clean-dialogue spans (see
    :func:`speech_flags`): before absorbing a block, dialogue segments at the leading/trailing
    edges are trimmed inward until a non-dialogue segment is hit. Interior segments (rap verses
    between song windows) are NOT trimmed and are still absorbed (preserves pit-2 protection).
    The song core itself is always preserved; trimming only affects what block expansion adds.
    """
    if not spans or not segments:
        return spans
    exp = spans if expandable is None else expandable
    prot = protect or []

    blocks: list[list[dict]] = [[segments[0]]]
    for s in segments[1:]:
        if s["start"] - blocks[-1][-1]["end"] <= block_gap:
            blocks[-1].append(s)
        else:
            blocks.append([s])

    def _clean_speech(seg: dict) -> bool:
        # Song-core segments are never trimmed even if speech score is high.
        return _overlaps(seg, prot) and not _overlaps(seg, spans)

    out = list(spans)
    for blk in blocks:
        ba, bb = blk[0]["start"], blk[-1]["end"]
        if not any(max(ba, a) < min(bb, b) for a, b in exp):
            continue
        lo, hi = 0, len(blk) - 1
        while lo <= hi and _clean_speech(blk[lo]):
            lo += 1
        while hi >= lo and _clean_speech(blk[hi]):
            hi -= 1
        if lo > hi:  # entire block is clean dialogue — do not absorb (defensive)
            continue
        out.append((blk[lo]["start"], blk[hi]["end"]))

    out.sort()
    merged: list[list[float]] = [list(out[0])]
    for a, b in out[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def group_segments_by_spans(
    segments: list[dict], spans: list[tuple[float, float]]
) -> list[list[dict]]:
    """Split speech segments into groups at song-span boundaries.

    Breaks whenever a song span falls between two adjacent segments, ensuring the
    contiguous [start, end] interval of each packed chunk does not cross a song span.
    Without this, slice_wav's contiguous cut would pull the skipped song back into the
    audio fed to ASR (skipping a span is not the same as excising it from the waveform).
    """
    if not spans or not segments:
        return [segments] if segments else []
    groups: list[list[dict]] = []
    cur: list[dict] = []
    for seg in segments:
        if cur:
            gap_a, gap_b = cur[-1]["end"], seg["start"]
            if any(max(gap_a, a) < min(gap_b, b) for a, b in spans):
                groups.append(cur)
                cur = []
        cur.append(seg)
    if cur:
        groups.append(cur)
    return groups


def detect_song_spans(
    wav_path: Path, *, batch: int = 32, progress=None
) -> tuple[
    list[tuple[float, float]], list[tuple[float, float]], list[tuple[float, float]]
]:
    """Run PANNs on a 32 kHz mono separated-vocals WAV.

    Returns ``(song/music spans, singing spans, clean-dialogue spans)``:
    - song/music spans: used to drop VAD segments.
    - singing spans (subset): spans with human vocals; only these trigger voiced-block
      expansion. Pure-instrumental BGM is absent here, so it never swallows adjacent dialogue.
    - clean-dialogue spans: trimmed from voiced-block boundaries during expansion rather
      than absorbed (see :func:`expand_spans_to_voiced_blocks`).

    Input must be separated vocals (route ii): instruments stripped, so singing vs. speech
    scores are cleanly separated. ``progress(done, total)`` is optional, called per batch.
    """
    data, sr = sf.read(str(wav_path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    assert sr == SR, f"expected {SR} Hz, got {sr!r} — decode_to_wav(sample_rate={SR})"

    win, hop = int(WIN_SEC * SR), int(HOP_SEC * SR)
    if len(data) < win:
        return [], [], []
    starts_idx = list(range(0, len(data) - win + 1, hop))
    wins = np.stack([data[s : s + win] for s in starts_idx])

    model = _get_model()
    batch_starts = list(range(0, len(wins), batch))
    nb = len(batch_starts)
    probs = []
    for bi, i in enumerate(batch_starts):
        out, _ = model.inference(wins[i : i + batch])
        probs.append(out)
        if progress is not None:
            progress(bi + 1, nb)
    P = np.concatenate(probs)

    starts_sec = [s / SR for s in starts_idx]
    spans = merge_spans(song_flags(P), starts_sec)
    sing_starts = [t for t, f in zip(starts_sec, sing_flags(P), strict=True) if f]
    sing_spans = [(a, b) for (a, b) in spans if any(a <= t < b for t in sing_starts)]
    speech_spans = merge_spans(speech_flags(P), starts_sec)
    log.info(
        "song-detect: %d span(s) (%d with singing), %.1fs total; %d clean-dialogue span(s)",
        len(spans),
        len(sing_spans),
        sum(b - a for a, b in spans),
        len(speech_spans),
    )
    return spans, sing_spans, speech_spans

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import soundfile as sf

SAMPLE_RATE = 16000
# Raised from silero default 100ms to 300ms: 200ms chops natural mid-sentence pauses.
VAD_MIN_SILENCE_MS = int(os.environ.get("VOXWEAVE_VAD_MIN_SILENCE_MS", "300"))


def pack_speech_segments(segments: list[dict], max_sec: float) -> list[dict]:
    """Bin-pack silero speech segments [{start,end}] into chunks of <= max_sec, cut at silence boundaries.

    Returns [{start, end, offset}] (offset == start, for timestamp shifting).
    Single segments longer than max_sec are hard-cut (no silence to snap to; word cuts tolerated).
    """
    if not segments:
        return []
    chunks: list[dict] = []

    def emit(start: float, end: float) -> None:
        chunks.append({"start": start, "end": end, "offset": start})

    def open_block(start, end):
        # Returns (None, None) after hard-cutting an overlong segment into slices.
        if end - start > max_sec:
            t = start
            while end - t > max_sec:
                emit(t, t + max_sec)
                t += max_sec
            emit(t, end)
            return None, None
        return start, end

    cur_start, cur_end = open_block(segments[0]["start"], segments[0]["end"])
    for seg in segments[1:]:
        if cur_start is None:
            cur_start, cur_end = open_block(seg["start"], seg["end"])
        elif seg["end"] - cur_start <= max_sec:
            cur_end = seg[
                "end"
            ]  # still within budget; merge into current chunk (including intervening silence)
        else:
            emit(cur_start, cur_end)  # type: ignore[arg-type]  # close at silence boundary
            cur_start, cur_end = open_block(seg["start"], seg["end"])
    if cur_start is not None:
        emit(cur_start, cur_end)  # type: ignore[arg-type]  # cur_end is always assigned together with cur_start
    return chunks


def decode_to_wav(
    media_path: Path,
    *,
    sample_rate: int = SAMPLE_RATE,
    mono: bool = True,
    audio_filter: str | None = None,
) -> Path:
    """Decode media to a temp WAV via ffmpeg; caller is responsible for deletion.

    Default: 16k mono for VAD/ASR. For separation, use sample_rate=44100, mono=False
    (full-band stereo). ``audio_filter`` inserts an ``-af`` stage (e.g. loudnorm).
    """
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="voxweave_")
    os.close(fd)
    out = Path(path)
    ac = ["-ac", "1"] if mono else []
    af = ["-af", audio_filter] if audio_filter else []
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            str(media_path),
            *af,
            *ac,
            "-ar",
            str(sample_rate),
            "-f",
            "wav",
            str(out),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out


def vad_speech_segments(wav_path: Path, *, threshold: float = 0.5) -> list[dict]:
    """silero VAD → speech segments [{start, end}] in seconds.

    threshold=0.5 is the silero default, used for chunking. Lowering to ~0.25 catches
    weakly voiced speech (e.g. secondary speaker attenuated by separation) but increases
    false positives on loud BGM, so only lower in specific scenarios.
    """
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad()
    # soundfile bypasses torchaudio>=2.9's torchcodec requirement
    data, sr = sf.read(str(wav_path), dtype="float32")
    assert sr == SAMPLE_RATE, (
        f"expected {SAMPLE_RATE} Hz wav, got {sr!r} Hz — run decode_to_wav first"
    )
    wav = torch.from_numpy(data)
    return get_speech_timestamps(
        wav,
        model,
        sampling_rate=SAMPLE_RATE,
        return_seconds=True,
        threshold=threshold,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        speech_pad_ms=100,
    )


def slice_wav(wav_path: Path, start: float, end: float) -> Path:
    """Slice the [start,end] segment from a 16k wav, write to a temp wav, return path (caller deletes)."""
    data, sr = sf.read(str(wav_path), dtype="float32")
    a = max(0, int(start * sr))
    b = min(len(data), int(end * sr))
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="voxweave_chunk_")
    os.close(fd)
    out = Path(path)
    sf.write(str(out), data[a:b], sr)
    return out

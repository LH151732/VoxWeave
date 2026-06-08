"""Spike: using PANNs on (separated) vocals to distinguish singing vs speech -- validates route ii feasibility.

Informal artifact; not wired into the pipeline. Three modes:
  clip     <wav>                Short clip: print top AudioSet labels + speech/sing/music scores to verify discrimination
  timeline <wav> [csv]          Sliding-window scan (2s window / 1s hop): print per-window scores + merged singing spans
  separate <fullband_wav>       Hit the separation endpoint, write vocals.flac path to stdout

Audio must be 32k mono wav (PANNs Cnn14 training sample rate).
"""

from __future__ import annotations

import csv as _csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 32000
LABELS_CSV = Path.home() / "panns_data" / "class_labels_indices.csv"

# AudioSet class indices (see class_labels_indices.csv)
IDX_SPEECH = [0, 4, 6, 7]  # Speech, Conversation, Babbling, Speech synthesizer
IDX_SING = [27, 28, 29, 30, 31, 36, 37, 254, 255, 266]  # Singing/Choir/Chant/Rap/...
IDX_MUSIC = [137, 267, 268, 270]  # Music / Background / Theme / Soundtrack

WIN_SEC = 2.0
HOP_SEC = 1.0
SING_ABS_MIN = 0.20  # absolute floor to avoid false positives on silent segments
GAP_MERGE_SEC = 2.0  # merge adjacent spans if gap <= this value
MIN_SPAN_SEC = 3.0  # discard spans shorter than this


def _labels() -> list[str]:
    rows = list(_csv.reader(LABELS_CSV.read_text().splitlines()))
    return [r[2] for r in rows[1:]]  # display_name column


def _at():
    from panns_inference import AudioTagging

    return AudioTagging(checkpoint_path=None, device="cuda")


def _load(wav: Path) -> np.ndarray:
    a, sr = sf.read(str(wav), dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(axis=1)
    assert sr == SR, (
        f"expected {SR} Hz, got {sr}; resample first: ffmpeg -ar 32000 -ac 1"
    )
    return a


def _hms(t: float) -> str:
    return f"{int(t) // 60:02d}:{t % 60:05.2f}"


def cmd_clip(wav: Path) -> None:
    a = _load(wav)
    labels = _labels()
    out, _ = _at().inference(a[None, :])
    p = out[0]
    top = np.argsort(p)[::-1][:10]
    print(f"# {wav.name}  ({len(a) / SR:.1f}s)")
    for i in top:
        print(f"  {p[i]:.3f}  {labels[i]}")
    sp = max(p[i] for i in IDX_SPEECH)
    si = max(p[i] for i in IDX_SING)
    mu = max(p[i] for i in IDX_MUSIC)
    verdict = "SING" if si > sp and si > SING_ABS_MIN else "SPEECH"
    print(f"  -> speech={sp:.3f} sing={si:.3f} music={mu:.3f}  => {verdict}")


def cmd_timeline(wav: Path, csv_out: Path | None) -> None:
    a = _load(wav)
    labels = _labels()
    at = _at()
    win, hop = int(WIN_SEC * SR), int(HOP_SEC * SR)
    starts = list(range(0, max(1, len(a) - win + 1), hop))
    wins = np.stack([a[s : s + win] for s in starts])  # (n, win)

    probs = []
    for i in range(0, len(wins), 32):
        out, _ = at.inference(wins[i : i + 32])
        probs.append(out)
    P = np.concatenate(probs)  # (n, 527)

    speech = P[:, IDX_SPEECH].max(axis=1)
    sing = P[:, IDX_SING].max(axis=1)
    music = P[:, IDX_MUSIC].max(axis=1)
    # Song detection criterion on separated vocals: singing raises `sing` or residual
    # accompaniment raises `music`, while `speech` is suppressed.
    # Either condition flags a window as non-speech (song/music):
    #   (1) sing clearly dominates speech
    #   (2) music is high and speech is low
    #       (music>0.30 excludes silence baseline ~0.13; speech<0.25 excludes dialogue
    #        overlapping light BGM)
    is_song = ((sing > speech) & (sing > 0.15)) | ((music > 0.30) & (speech < 0.25))

    print(f"# timeline {wav.name}  {len(a) / SR:.1f}s, {len(starts)} windows")
    rows = []
    for k, s in enumerate(starts):
        t = s / SR
        top = int(P[k].argmax())
        flag = "SING" if is_song[k] else ""
        print(
            f"{_hms(t)}  spch={speech[k]:.2f} sing={sing[k]:.2f} "
            f"mus={music[k]:.2f}  {labels[top]:<24} {flag}"
        )
        rows.append(
            (f"{t:.1f}", speech[k], sing[k], music[k], labels[top], int(is_song[k]))
        )

    if csv_out:
        with open(csv_out, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["t", "speech", "sing", "music", "top", "is_song"])
            w.writerows(rows)

    # Merge song windows into spans
    spans = []
    cur = None
    for k, s in enumerate(starts):
        t = s / SR
        if is_song[k]:
            if cur is None:
                cur = [t, t + WIN_SEC]
            elif t - cur[1] <= GAP_MERGE_SEC:
                cur[1] = t + WIN_SEC
            else:
                spans.append(cur)
                cur = [t, t + WIN_SEC]
    if cur:
        spans.append(cur)
    spans = [s for s in spans if s[1] - s[0] >= MIN_SPAN_SEC]

    print("\n# detected singing/song spans:")
    if not spans:
        print("  (none)")
    for a0, a1 in spans:
        print(f"  {_hms(a0)} - {_hms(a1)}   ({a1 - a0:.1f}s)")


def main() -> None:
    mode = sys.argv[1]
    if mode == "clip":
        cmd_clip(Path(sys.argv[2]))
    elif mode == "timeline":
        cmd_timeline(
            Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else None
        )
    else:
        sys.exit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()

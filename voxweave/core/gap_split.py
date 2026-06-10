"""Pause/gap-based cue splitting predicate (pure, stdlib only).

Tiered by gap size with VAD confirmation in the danger zone — forced-aligner
gaps can be artifacts (a dropped word looks like a ~0.3-0.7s "pause"):

    gap < clause_ms                  -> never (intra-phrase breath)
    clause_ms <= gap < vad_skip_ms   -> split only if the gap is genuinely silent
                                        (no overlap with a VAD speech span)
    gap >= vad_skip_ms               -> always (real pause; BBC visible-gap zone)

When ``speech_spans is None`` (no audio): degrades to a single gap threshold
``offline_ms`` (trust only larger gaps).
"""

from __future__ import annotations


def _overlaps_speech(
    a: float, b: float, speech_spans: list[tuple[float, float]]
) -> bool:
    """True if [a, b] overlaps any speech span by more than 50ms (VAD edge jitter tolerance)."""
    eps = 0.05
    for s, e in speech_spans:
        if min(b, e) - max(a, s) > eps:
            return True
    return False


def adaptive_clause_ms(
    gaps_ms: list[float],
    *,
    lo: int = 300,
    hi: int = 800,
    min_samples: int = 50,
    percentile: float = 90.0,
    margin: float = 1.15,
) -> int | None:
    """Per-file clause threshold from the sub-2s inter-unit gap distribution.

    clause_ms separates intra-clause articulation gaps from real pauses; the
    static default (400ms, ja x1.4) is a corpus-level compromise — a fast
    talker's p90 articulation gap sits far below a slow narrator's. p90 of
    sub-2s positive gaps, a safety margin, and a [lo, hi] clamp adapt it per
    file. EXPERIMENTAL: opt in via VOXWEAVE_GAP_ADAPTIVE=1 and validate with
    scripts/calib_segmentation.py before trusting a new corpus. Returns None
    when the file has too few gap samples to estimate a distribution.
    """
    sub = sorted(g for g in gaps_ms if 0 < g < 2000)
    if len(sub) < min_samples:
        return None
    idx = (len(sub) - 1) * percentile / 100.0
    base = int(idx)
    frac = idx - base
    if base + 1 < len(sub):
        p = sub[base] * (1 - frac) + sub[base + 1] * frac
    else:
        p = sub[-1]
    return max(lo, min(hi, round(p * margin)))


def gap_qualifies(
    prev_end: float | None,
    next_start: float | None,
    speech_spans: list[tuple[float, float]] | None,
    *,
    clause_ms: int,
    vad_skip_ms: int,
    offline_ms: int,
) -> bool:
    if prev_end is None or next_start is None:
        return False
    gap = next_start - prev_end
    if gap <= 0:
        return False
    gap_ms = gap * 1000.0
    if speech_spans is None:
        return gap_ms >= offline_ms
    if gap_ms < clause_ms:
        return False
    if gap_ms >= vad_skip_ms:
        return True
    # danger zone: only split if the gap interval is genuinely silent
    return not _overlaps_speech(prev_end, next_start, speech_spans)

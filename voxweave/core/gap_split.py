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

# tests/test_gap_split.py
from voxweave.core.gap_split import gap_qualifies

TH = dict(clause_ms=400, vad_skip_ms=1000, offline_ms=700)


def test_below_clause_never_splits():
    # gap = 0.2s < 0.4s clause lower bound -> never split (even when silence is present)
    assert gap_qualifies(1.0, 1.2, [(2.0, 3.0)], **TH) is False


def test_big_gap_unconditional_even_without_vad():
    # gap = 1.5s >= vad_skip 1.0s -> unconditional split
    assert gap_qualifies(1.0, 2.5, None, **TH) is True


def test_mid_gap_needs_vad_silence():
    # gap = 0.6s falls in the 400-1000ms danger zone: gap interval covered by speech -> no split (spurious dropped-word gap)
    assert gap_qualifies(1.0, 1.6, [(0.0, 2.0)], **TH) is False
    # gap interval is genuine silence (speech is elsewhere, does not touch gap interval, clear of eps boundary) -> split
    assert gap_qualifies(1.0, 1.6, [(0.0, 0.9), (1.7, 2.0)], **TH) is True


def test_mid_gap_no_vad_map_uses_offline_threshold():
    # no VAD map: gap=0.6s < offline 0.7s -> no split
    assert gap_qualifies(1.0, 1.6, None, **TH) is False
    # gap=0.8s >= offline 0.7s -> split
    assert gap_qualifies(1.0, 1.8, None, **TH) is True


def test_none_timestamps_never_split():
    assert gap_qualifies(None, 1.6, None, **TH) is False

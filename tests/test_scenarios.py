"""Replay tests for the song-skip decision chain using real clips (zero GPU).

tests/scenarios/*.json are captured by scripts/capture_scenario.py (PANNs per-window scores + VAD segments).
These tests replay the full decision chain with pure functions (songdet + pipeline.plan_song_skip) and assert:
  - final song spans == golden snapshot (expected_song_spans)
  - every timestamp in assert.speech_present_at falls within a retained VAD segment (regression anchor: speech that should have subtitles must not be swallowed)
  - no chunk exceeds max_chunk_sec (both too-short and too-long chunks hurt ASR recall)

To add a new scenario: run `python scripts/capture_scenario.py <clip> <name>`, then manually fill in
speech_present_at in the generated json.  Afterwards `pytest tests/test_scenarios.py` guards it permanently
without needing to re-run the full clip.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from voxweave import songdet
from voxweave.pipeline import MAX_CHUNK_SEC, MIN_SONG_SKIP_SEC, plan_song_skip

SCENARIO_DIR = Path(__file__).parent / "scenarios"
SCENARIOS = sorted(SCENARIO_DIR.glob("*.json")) if SCENARIO_DIR.exists() else []


def _replay(fx: dict):
    """fixture -> (final_song_spans, kept_segs, chunks), replaying the same pure decision chain as pipeline."""
    sc = fx["scores"]
    speech = np.array(sc["speech"], dtype="float32")
    sing = np.array(sc["sing"], dtype="float32")
    music = np.array(sc["music"], dtype="float32")
    t = sc["t"]
    song = songdet.merge_spans(songdet.song_flags_from_scores(speech, sing, music), t)
    sing_fl = songdet.sing_flags_from_scores(speech, sing, music)
    sing_starts = [tt for tt, f in zip(t, sing_fl) if f]
    sing_spans = [(a, b) for (a, b) in song if any(a <= x < b for x in sing_starts)]
    speech_spans = songdet.merge_spans(
        songdet.speech_flags_from_scores(speech, sing, music), t
    )
    segs = [{"start": a, "end": b} for a, b in fx["vad_segs"]]
    _, final, kept, chunks = plan_song_skip(
        song,
        sing_spans,
        segs,
        speech_spans=speech_spans,
        min_skip_sec=MIN_SONG_SKIP_SEC,
        max_chunk_sec=MAX_CHUNK_SEC,
    )
    return final, kept, chunks


def _covered(t: float, segs: list[dict]) -> bool:
    return any(s["start"] <= t <= s["end"] for s in segs)


@pytest.mark.skipif(not SCENARIOS, reason="no scenario fixtures")
@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_scenario_song_skip(path: Path):
    fx = json.loads(path.read_text(encoding="utf-8"))
    a = fx.get("assert", {})
    final, kept, chunks = _replay(fx)

    # 1) final song spans == golden snapshot (logic changes surface here; re-run capture intentionally)
    got = [[round(x, 1), round(y, 1)] for x, y in final]
    assert got == a.get("expected_song_spans", got), (
        f"{path.stem}: song spans changed {got} != {a.get('expected_song_spans')}"
    )

    # 2) every timestamp that should have speech must fall within a retained VAD segment (regression anchor: must not be swallowed by song-skip)
    for tp in a.get("speech_present_at", []):
        assert _covered(tp, kept), (
            f"{path.stem}: speech at t={tp}s was swallowed by song-skip (should be retained)"
        )

    # 3) no oversized chunks (both too-short and too-long chunks hurt ASR recall)
    cap = a.get("max_chunk_sec", MAX_CHUNK_SEC)
    for c in chunks:
        assert c["end"] - c["start"] <= cap + 1e-6, (
            f"{path.stem}: chunk {c['start']:.1f}-{c['end']:.1f} exceeds {cap}s"
        )

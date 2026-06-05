"""Regression tests for pipeline path derivation and sibling file writing.

Key concern: filenames with interior dots (e.g. ``...`` in YouTube titles) must not be
silently truncated by ``Path.with_suffix`` when deriving .vtt/.json sibling paths.
"""

from pathlib import Path

from voxweave import pipeline

# Real filename that triggered the bug: title has interior ``...``
DOTTED = (
    "Fuwawa一次播放所有音樂來轟炸鄰居...氣到Kronii受不了【Hololive 中文】 [s6r36ux1d4Q]"
)


def test_swap_ext_preserves_mid_name_dots():
    m = Path(f"{DOTTED}.webm")
    assert pipeline._swap_ext(m, ".vtt").name == f"{DOTTED}.vtt"
    assert pipeline._swap_ext(m, ".json").name == f"{DOTTED}.json"
    # chained (translate output): .vtt -> .zh.vtt also must not truncate
    v = pipeline._swap_ext(m, ".vtt")
    assert pipeline._swap_ext(v, ".zh.vtt").name == f"{DOTTED}.zh.vtt"


def test_swap_ext_no_suffix_appends():
    assert pipeline._swap_ext(Path("README"), ".vtt").name == "README.vtt"


def test_process_dotted_filename_writes_full_siblings(tmp_path):
    media = (
        tmp_path / f"{DOTTED}.webm"
    )  # does not need to exist: word_segments bypasses transcription
    units = [
        {"text": "hello", "start": 0.0, "end": 1.0},
        {"text": "world", "start": 1.0, "end": 2.0},
    ]
    out = pipeline.process(media, word_segments=("en", units))
    # returned .vtt path must be the full name, not truncated at the first ``...``
    assert out == tmp_path / f"{DOTTED}.vtt"
    assert out.exists()
    assert (tmp_path / f"{DOTTED}.json").exists()
    # truncated name must not exist
    assert not (tmp_path / "Fuwawa一次播放所有音樂來轟炸鄰居...vtt").exists()


def test_process_vtt_has_timestamps_by_default(tmp_path):
    # default process output includes timestamps: cues carry word-level start/end -> timing line written
    media = tmp_path / "ep.mkv"
    units = [
        {"text": "hello", "start": 0.0, "end": 1.0},
        {"text": "world", "start": 1.0, "end": 2.0},
    ]
    out = pipeline.process(media, word_segments=("en", units))
    body = out.read_text(encoding="utf-8")
    assert "-->" in body
    assert "00:00:00.000 -->" in body
    # re-parse as timestamped blocks (align path compat): all blocks carry start/end
    blocks = pipeline.realign.parse_vtt_blocks(body)
    assert blocks and all(b["start"] is not None for b in blocks)


def test_process_no_timestamps_strips(tmp_path):
    # --no-timestamps: plain-text edit draft, no timing lines (edit text/breaks then re-run align)
    media = tmp_path / "ep.mkv"
    units = [{"text": "hi", "start": 0.0, "end": 1.0}]
    out = pipeline.process(media, word_segments=("en", units), timestamps=False)
    body = out.read_text(encoding="utf-8")
    assert "-->" not in body
    assert "hi" in body


def test_write_siblings_drops_ts_line_when_cue_time_missing(tmp_path):
    # defensive: cue missing start/end (rare) -> falls back to plain text, does not crash (fmt_ts rejects None)
    cues = [
        {"text": "a", "start": None, "end": None},
        {"text": "b", "start": 0.0, "end": 1.0},
    ]
    out = pipeline._write_siblings(tmp_path / "x.mkv", cues, [], "en")
    body = out.read_text(encoding="utf-8")
    assert "00:00:00.000 --> 00:00:01.000" in body  # second cue has timing
    # first cue (a) has no timing line: line before "a" must be blank, not "-->"
    lines = body.splitlines()
    assert "a" in lines
    assert lines[lines.index("a") - 1] == ""


def test_find_sibling_media_matches_dotted_name(tmp_path):
    media = tmp_path / f"{DOTTED}.webm"
    media.write_bytes(b"x")
    vtt = tmp_path / f"{DOTTED}.vtt"
    assert pipeline._find_sibling_media(vtt) == media

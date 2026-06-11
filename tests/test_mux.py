# tests/test_mux.py
# pack/burn command construction and helpers: language detection from VTT
# filenames, "VoxWeave <Language>" track titles, sibling media resolution
# (language tag stripped), ffmpeg argv builders (stream mapping, sub codec per
# container, hvc1 tagging, pixel-format/bit-depth policy, mp4 audio fallback),
# and encoder selection. No ffmpeg/ffprobe execution; probed data is injected.
from pathlib import Path

import pytest

from voxweave import mux
from voxweave.export import ass_header

VTT_BODY = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhi\n"


def has_seq(cmd, *seq):
    """True when seq appears as a contiguous run inside cmd."""
    n = len(seq)
    return any(tuple(cmd[i : i + n]) == seq for i in range(len(cmd) - n + 1))


# --- language / title / path helpers ---------------------------------------


def test_detect_vtt_language():
    assert mux.detect_vtt_language(Path("ep.zh.vtt")) == "zh"
    assert mux.detect_vtt_language(Path("ep.en.vtt")) == "en"
    assert mux.detect_vtt_language(Path("Show S01E01.ja.vtt")) == "ja"
    assert mux.detect_vtt_language(Path("ep.vtt")) is None
    assert mux.detect_vtt_language(Path("ep.sdh.vtt")) is None  # not a language
    assert mux.detect_vtt_language(Path("ep...vtt")) is None  # interior dots


def test_track_title():
    assert mux.track_title("zh") == "VoxWeave Chinese"
    assert mux.track_title("en") == "VoxWeave English"
    assert mux.track_title("ja") == "VoxWeave Japanese"
    assert mux.track_title(None) == "VoxWeave"


def test_resolve_media_sibling_and_language_tag(tmp_path):
    media = tmp_path / "ep.mkv"
    media.write_bytes(b"")
    plain = tmp_path / "ep.vtt"
    plain.write_text(VTT_BODY, encoding="utf-8")
    tagged = tmp_path / "ep.zh.vtt"
    tagged.write_text(VTT_BODY, encoding="utf-8")
    assert mux.resolve_media(plain, None) == media
    assert mux.resolve_media(tagged, None) == media  # .zh stripped for lookup
    explicit = tmp_path / "other.mkv"
    explicit.write_bytes(b"")
    assert mux.resolve_media(plain, explicit) == explicit


def test_resolve_media_missing_raises(tmp_path):
    vtt = tmp_path / "lonely.vtt"
    vtt.write_text(VTT_BODY, encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="--media"):
        mux.resolve_media(vtt, None)


def test_default_output_avoids_overwriting_source(tmp_path):
    media = tmp_path / "ep.mkv"
    assert mux.default_output(media, "mp4", "burn") == tmp_path / "ep.mp4"
    assert mux.default_output(media, "mkv", "pack") == tmp_path / "ep.pack.mkv"


# --- pack command construction ----------------------------------------------


def _streams(*entries):
    return [dict(e, index=i) for i, e in enumerate(entries)]


def test_build_pack_cmd_mkv_keeps_everything():
    streams = _streams(
        {"codec_type": "video", "codec_name": "hevc"},
        {"codec_type": "audio", "codec_name": "flac"},
        {"codec_type": "subtitle", "codec_name": "hdmv_pgs_subtitle"},
    )
    cmd = mux.build_pack_cmd(
        Path("ep.mkv"),
        [Path("ep.zh.vtt")],
        Path("ep.pack.mkv"),
        container="mkv",
        source_streams=streams,
    )
    assert cmd[:2] == ["ffmpeg", "-nostdin"]
    assert ["-map", "0"] == cmd[cmd.index("-map") : cmd.index("-map") + 2]
    assert has_seq(cmd, "-map", "1:0")  # appended VTT
    assert "-c:s:1" in cmd and cmd[cmd.index("-c:s:1") + 1] == "srt"
    assert "-metadata:s:s:1" in cmd
    assert "language=chi" in cmd and "title=VoxWeave Chinese" in cmd
    assert (
        "-disposition:s:1" in cmd
        and cmd[cmd.index("-disposition:s:1") + 1] == "default"
    )
    assert cmd[-1] == "ep.pack.mkv"


def test_build_pack_cmd_mp4_drops_image_subs_tags_hvc1():
    streams = _streams(
        {"codec_type": "video", "codec_name": "hevc"},
        {"codec_type": "audio", "codec_name": "aac"},
        {"codec_type": "subtitle", "codec_name": "hdmv_pgs_subtitle"},
        {"codec_type": "subtitle", "codec_name": "subrip"},
    )
    cmd = mux.build_pack_cmd(
        Path("ep.mp4"),
        [Path("ep.en.vtt")],
        Path("ep.pack.mp4"),
        container="mp4",
        source_streams=streams,
    )
    assert has_seq(cmd, "-map", "0:3")  # text sub kept
    assert "0:2" not in cmd  # PGS dropped for mp4
    assert "-c:s" in cmd and cmd[cmd.index("-c:s") + 1] == "mov_text"
    assert "-tag:v" in cmd and cmd[cmd.index("-tag:v") + 1] == "hvc1"
    # one kept text sub -> new track is s:1
    assert "language=eng" in cmd and "title=VoxWeave English" in cmd
    assert "-disposition:s:1" in cmd


def test_build_pack_cmd_multiple_vtts_index_after_existing():
    streams = _streams(
        {"codec_type": "video", "codec_name": "h264"},
        {"codec_type": "subtitle", "codec_name": "ass"},
    )
    cmd = mux.build_pack_cmd(
        Path("ep.mkv"),
        [Path("ep.zh.vtt"), Path("ep.ja.vtt")],
        Path("out.mkv"),
        container="mkv",
        source_streams=streams,
    )
    assert "-c:s:1" in cmd and "-c:s:2" in cmd  # after the existing ass track
    assert "title=VoxWeave Chinese" in cmd and "title=VoxWeave Japanese" in cmd
    assert "-disposition:s:1" in cmd  # only the first new track gets default


def test_pack_rejects_plain_text_draft(tmp_path):
    media = tmp_path / "ep.mkv"
    media.write_bytes(b"")
    vtt = tmp_path / "ep.vtt"
    vtt.write_text("WEBVTT\n\nhello no timestamps\n", encoding="utf-8")
    with pytest.raises(ValueError, match="align"):
        mux.pack([vtt])


# --- burn building blocks ----------------------------------------------------


def test_burn_pix_fmt_policy():
    # hevc/av1 preserve 10-bit; h264 is always 8-bit (nvenc h264 cannot do 10)
    assert mux._burn_pix_fmt("hevc_nvenc", True) == "p010le"
    assert mux._burn_pix_fmt("hevc_nvenc", False) == "nv12"
    assert mux._burn_pix_fmt("av1_nvenc", True) == "p010le"
    assert mux._burn_pix_fmt("h264_nvenc", True) == "nv12"
    assert mux._burn_pix_fmt("hevc_videotoolbox", True) == "p010le"
    assert mux._burn_pix_fmt("libx265", True) == "yuv420p10le"
    assert mux._burn_pix_fmt("libx265", False) == "yuv420p"
    assert mux._burn_pix_fmt("libx264", True) == "yuv420p"


def test_encoder_args_constant_quality_only():
    nv = mux._encoder_args("hevc_nvenc", 23)
    assert has_seq(nv, "-rc", "vbr") and has_seq(nv, "-cq", "23")
    assert has_seq(nv, "-b:v", "0")  # pure CQ, no bitrate target
    assert has_seq(nv, "-b_ref_mode", "middle") and has_seq(nv, "-temporal-aq", "1")
    assert mux._encoder_args("hevc_videotoolbox", 65) == ["-q:v", "65"]
    assert has_seq(mux._encoder_args("libx265", 23), "-crf", "23")
    assert has_seq(mux._encoder_args("libsvtav1", 30), "-preset", "6")


def test_filter_escape():
    assert mux._filter_escape("/tmp/a.ass") == "/tmp/a.ass"
    assert mux._filter_escape("/tmp/a:b,c.ass") == "/tmp/a\\:b\\,c.ass"
    assert mux._filter_escape("C:\\tmp\\a.ass") == "C\\:/tmp/a.ass"


def test_build_burn_cmd_hevc_mp4():
    cmd = mux.build_burn_cmd(
        Path("ep.mkv"),
        Path("/tmp/s.ass"),
        Path("ep.mp4"),
        encoder="hevc_nvenc",
        quality=23,
        container="mp4",
        src_10bit=True,
        audio_codecs=["flac"],
    )
    assert cmd[:2] == ["ffmpeg", "-nostdin"]
    assert has_seq(cmd, "-hwaccel", "cuda")
    vf = cmd[cmd.index("-vf") + 1]
    assert vf == "ass=/tmp/s.ass,format=p010le"
    assert has_seq(cmd, "-map", "0:v:0") and has_seq(cmd, "-map", "0:a?")
    assert "-sn" not in cmd  # subs dropped by explicit mapping, not -sn
    assert has_seq(cmd, "-tag:v", "hvc1")
    assert has_seq(cmd, "-c:a", "aac")  # flac cannot live in mp4
    assert cmd[-1] == "ep.mp4"


def test_build_burn_cmd_mkv_copies_audio_no_tag():
    cmd = mux.build_burn_cmd(
        Path("ep.mkv"),
        Path("/tmp/s.ass"),
        Path("ep.burn.mkv"),
        encoder="libx264",
        quality=19,
        container="mkv",
        src_10bit=False,
        audio_codecs=["flac"],
    )
    assert "-hwaccel" not in cmd  # software path
    assert has_seq(cmd, "-c:a", "copy")
    assert "-tag:v" not in cmd
    assert "format=yuv420p" in cmd[cmd.index("-vf") + 1]


def test_pick_encoder_prefers_hardware(monkeypatch):
    monkeypatch.setattr(
        mux, "_available_encoders", lambda: frozenset({"hevc_nvenc", "libx265"})
    )
    monkeypatch.setattr(mux, "_encoder_works", lambda _name: True)
    monkeypatch.setattr(mux.sys, "platform", "linux")
    assert mux.pick_encoder("hevc") == "hevc_nvenc"


def test_pick_encoder_falls_back_to_software(monkeypatch):
    monkeypatch.setattr(
        mux, "_available_encoders", lambda: frozenset({"hevc_nvenc", "libx265"})
    )
    monkeypatch.setattr(mux, "_encoder_works", lambda _name: False)  # no GPU
    monkeypatch.setattr(mux.sys, "platform", "linux")
    assert mux.pick_encoder("hevc") == "libx265"


def test_pick_encoder_videotoolbox_on_darwin(monkeypatch):
    monkeypatch.setattr(
        mux,
        "_available_encoders",
        lambda: frozenset({"hevc_videotoolbox", "libx265"}),
    )
    monkeypatch.setattr(mux, "_encoder_works", lambda _name: True)
    monkeypatch.setattr(mux.sys, "platform", "darwin")
    assert mux.pick_encoder("hevc") == "hevc_videotoolbox"


def test_pick_encoder_force_and_unknown_codec():
    assert mux.pick_encoder("hevc", force="libx265") == "libx265"
    with pytest.raises(ValueError, match="unsupported codec"):
        mux.pick_encoder("vp9")


# --- ASS header scaling (burn renders at the actual frame size) --------------


def test_ass_header_scales_to_frame():
    h = ass_header(width=3840, height=2160)
    assert "PlayResX: 3840" in h and "PlayResY: 2160" in h
    assert ",144," in h  # 72 * 2 font size
    h = ass_header(width=1920, height=1080, font="Noto Sans CJK SC", font_size=58)
    assert "Style: Default,Noto Sans CJK SC,58," in h

import json
from pathlib import Path
from types import SimpleNamespace

from voxweave import asrfix, pipeline


class FakeClient:
    """Records received messages and returns chat.completions from a preset queue (same shape as the translate test helper)."""

    def __init__(self, contents):
        self._contents = list(contents)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model, messages, **kw):
        self.calls.append(messages)
        content = self._contents.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def _blocks(texts):
    return [
        {"text": t, "start": i * 1.0, "end": i * 1.0 + 0.5} for i, t in enumerate(texts)
    ]


# --------------------------- parse_fixes --------------------------- #
def test_parse_fixes_basic():
    raw = '{"fixes":[{"i":2,"orig":"如金","fixed":"如今","reason":"同音"}]}'
    fx = asrfix.parse_fixes(raw)
    assert fx == [{"i": 2, "orig": "如金", "fixed": "如今", "reason": "同音"}]


def test_parse_fixes_salvages_dirty_and_drops_bad():
    raw = 'noise {"fixes":[{"i":"x"},{"i":1,"orig":"a","fixed":"b"}]} trailing'
    fx = asrfix.parse_fixes(raw)
    assert fx == [{"i": 1, "orig": "a", "fixed": "b", "reason": ""}]


def test_parse_fixes_garbage_returns_empty():
    assert asrfix.parse_fixes("not json at all") == []


def test_parse_fixes_non_str_non_dict_tolerated():
    # non-str/non-dict input must fall through to [] (not raise TypeError from json.loads)
    assert asrfix.parse_fixes(None) == []
    assert asrfix.parse_fixes(123) == []
    assert asrfix.parse_fixes([{"i": 0}]) == []


# --------------------------- apply_fixes (SAFETY GATE) --------------------------- #
def test_apply_fixes_applies_matching():
    blocks = _blocks(["如金仍是主力", "下一句"])
    fixes = [
        {"i": 0, "orig": "如金仍是主力", "fixed": "如今仍是主力", "reason": "同音"}
    ]
    new, applied, rejected = asrfix.apply_fixes(blocks, fixes)
    assert new[0] == "如今仍是主力"
    assert len(applied) == 1 and not rejected


def test_apply_fixes_rejects_orig_mismatch_cross_cue():
    # cross-cue word split: model merged text from adjacent cues into orig -> orig does not match actual cue -> rejected, not applied
    blocks = _blocks(["After how much Mom suffer", "ed You should be leading"])
    fixes = [
        {
            "i": 0,
            "orig": "After how much Mom suffered",
            "fixed": "After how much Mom suffered",
            "reason": "x",
        }
    ]
    new, applied, rejected = asrfix.apply_fixes(blocks, fixes)
    assert new == [b["text"] for b in blocks]  # original text unchanged
    assert not applied and len(rejected) == 1
    assert "orig != cue" in rejected[0]["_why"]


def test_apply_fixes_rejects_out_of_range():
    blocks = _blocks(["a"])
    new, applied, rejected = asrfix.apply_fixes(
        blocks, [{"i": 9, "orig": "a", "fixed": "b", "reason": ""}]
    )
    assert not applied and rejected[0]["_why"] == "index out of range"


def test_apply_fixes_rejects_noop():
    blocks = _blocks(["hello"])
    new, applied, rejected = asrfix.apply_fixes(
        blocks, [{"i": 0, "orig": "hello", "fixed": "hello", "reason": ""}]
    )
    assert not applied and rejected[0]["_why"] == "no-op"


def test_apply_fixes_norm_tolerates_wrapped_newline():
    # multi-line cue (wrapped with \n); model quotes orig space-joined -> normalizes to a match, fix applied normally
    blocks = [{"text": "U S\nMarshall", "start": 0.0, "end": 1.0}]
    fixes = [
        {"i": 0, "orig": "U S Marshall", "fixed": "U S Marshal", "reason": "title"}
    ]
    new, applied, rejected = asrfix.apply_fixes(blocks, fixes)
    assert new[0] == "U S Marshal" and len(applied) == 1 and not rejected


# --------------------------- build_messages / glossary --------------------------- #
_GLOSSARY_MARK = "canonical entities for THIS video"


def test_build_messages_injects_glossary():
    msgs = asrfix.build_messages(
        [{"i": 0, "t": "x"}], glossary={"微热如冰": "Vera Rubin"}
    )
    sys = msgs[0]["content"]
    assert _GLOSSARY_MARK in sys and "Vera Rubin" in sys
    assert json.loads(msgs[1]["content"])["cues"] == [{"i": 0, "t": "x"}]


def test_build_messages_no_glossary_clean():
    # the prompt body itself mentions "GLOSSARY", so check for the injection-specific marker and confirm no specific term is present
    msgs = asrfix.build_messages([{"i": 0, "t": "x"}])
    assert _GLOSSARY_MARK not in msgs[0]["content"]


# --------------------------- render_vtt --------------------------- #
def test_render_vtt_preserves_timestamps():
    blocks = _blocks(["a", "b"])
    out = asrfix.render_vtt(blocks, ["A", "b"])
    assert "00:00:00.000 --> 00:00:00.500" in out
    assert "A" in out and out.startswith("WEBVTT")


def test_render_vtt_text_only_when_no_timestamps():
    blocks = [{"text": "a", "start": None, "end": None}]
    out = asrfix.render_vtt(blocks, ["A"])
    assert "-->" not in out and "A" in out


# --------------------------- pipeline.correct (E2E with mock) --------------------------- #
def _make_vtt(tmp_path: Path, cues) -> Path:
    lines = ["WEBVTT", ""]
    for i, c in enumerate(cues):
        lines += [f"00:00:0{i}.000 --> 00:00:0{i}.500", c, ""]
    p = tmp_path / "ep.vtt"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def test_pipeline_correct_sidecar_does_not_touch_vtt(tmp_path, monkeypatch):
    vtt = _make_vtt(tmp_path, ["如金仍是主力", "正常一句"])
    orig_text = vtt.read_text(encoding="utf-8")
    client = FakeClient(
        [
            '{"fixes":[{"i":0,"orig":"如金仍是主力","fixed":"如今仍是主力","reason":"同音"}]}'
        ]
    )
    monkeypatch.setattr(asrfix, "_make_client", lambda *a, **k: client)
    res = pipeline.correct(vtt, api_key="x")
    # original VTT untouched
    assert vtt.read_text(encoding="utf-8") == orig_text
    # sidecar written and contains the correction
    side = res["out"]
    assert side.name == "ep.asrfix.vtt"
    assert "如今仍是主力" in side.read_text(encoding="utf-8")
    # audit JSON
    audit = json.loads(res["audit"].read_text(encoding="utf-8"))
    assert len(audit["applied"]) == 1
    assert res["applied_in_place"] is False


def test_pipeline_correct_apply_overwrites_vtt_no_audit_json(tmp_path, monkeypatch):
    vtt = _make_vtt(tmp_path, ["如金仍是主力"])
    client = FakeClient(
        [
            '{"fixes":[{"i":0,"orig":"如金仍是主力","fixed":"如今仍是主力","reason":"同音"}]}'
        ]
    )
    monkeypatch.setattr(asrfix, "_make_client", lambda *a, **k: client)
    res = pipeline.correct(vtt, api_key="x", apply=True)
    assert "如今仍是主力" in vtt.read_text(encoding="utf-8")
    assert res["out"] == vtt and res["applied_in_place"] is True
    # apply must NOT leave a new json behind
    assert res["audit"] is None
    assert not (tmp_path / "ep.asrfix.json").exists()
    assert res["aligned"] is False  # align_after not requested


def test_pipeline_correct_apply_auto_aligns(tmp_path, monkeypatch):
    vtt = _make_vtt(tmp_path, ["如金仍是主力"])
    client = FakeClient(
        [
            '{"fixes":[{"i":0,"orig":"如金仍是主力","fixed":"如今仍是主力","reason":"同音"}]}'
        ]
    )
    monkeypatch.setattr(asrfix, "_make_client", lambda *a, **k: client)
    called: dict = {}

    def fake_align(p, **kw):
        called["path"] = p
        return p

    monkeypatch.setattr(pipeline, "align", fake_align)
    res = pipeline.correct(vtt, api_key="x", apply=True, align_after=True)
    assert called["path"] == vtt  # re-aligned the in-place file
    assert res["aligned"] is True


def test_pipeline_correct_empty_diff_skips_align(tmp_path, monkeypatch):
    # nothing applied (safety gate rejects) -> VTT unchanged -> no point aligning
    vtt = _make_vtt(tmp_path, ["业务存在"])
    client = FakeClient(
        ['{"fixes":[{"i":0,"orig":"完全不同的原文","fixed":"乱改","reason":"x"}]}']
    )
    monkeypatch.setattr(asrfix, "_make_client", lambda *a, **k: client)
    called: dict = {}
    monkeypatch.setattr(
        pipeline, "align", lambda p, **kw: called.setdefault("hit", True)
    )
    res = pipeline.correct(vtt, api_key="x", apply=True, align_after=True)
    assert "hit" not in called  # align never invoked
    assert res["aligned"] is False


def test_pipeline_correct_rejects_unsafe_keeps_text(tmp_path, monkeypatch):
    # model proposes an out-of-bounds rewrite (orig does not match actual text) -> safety gate rejects it, sidecar text equals original
    vtt = _make_vtt(tmp_path, ["业务存在"])
    client = FakeClient(
        ['{"fixes":[{"i":0,"orig":"完全不同的原文","fixed":"乱改","reason":"x"}]}']
    )
    monkeypatch.setattr(asrfix, "_make_client", lambda *a, **k: client)
    res = pipeline.correct(vtt, api_key="x")
    assert "乱改" not in res["out"].read_text(encoding="utf-8")
    assert len(res["rejected"]) == 1 and not res["applied"]

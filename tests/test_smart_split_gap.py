# tests/test_smart_split_gap.py
import pytest
from voxweave.core.smart_split import smart_split_segments

# thresholds opt-in (gap-split + dur-cap gated on this); min_cue_s=0 so cleanup
# never extends short cues and gap-count assertions stay exact.
TH = {
    "clause_ms": 400,
    "vad_skip_ms": 1000,
    "offline_ms": 700,
    "min_cue_s": 0.0,
    "max_cue_s": 7.0,
}


def _seg(words, lang):
    text = ("" if lang in {"zh", "ja"} else " ").join(w["word"] for w in words)
    return {
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "text": text,
        "words": words,
    }


def test_runon_split_at_gap_offline():
    # no-punctuation ja: two real words (今日は | 晴れ) with 1.5s silence between them -> splits
    # into two cues. The gap lands on a BudouX word boundary (今日は / 晴れ), so gap-split
    # fires (offline gap-only mode, gap > 0.7s).
    words = [
        {"word": c, "start": 0.0 + i * 0.2, "end": 0.1 + i * 0.2}
        for i, c in enumerate("今日は")
    ] + [
        {"word": c, "start": 2.0 + i * 0.2, "end": 2.1 + i * 0.2}
        for i, c in enumerate("晴れ")
    ]
    cues = smart_split_segments(
        [_seg(words, "ja")], "ja", speech_spans=None, thresholds=TH
    )
    assert len(cues) == 2
    assert cues[0]["text"] == "今日は" and cues[1]["text"] == "晴れ"


def test_no_gap_split_inside_word():
    # OOV CTC timing error: り in 酒造り drifts late, creating a spurious 2.1s gap between
    # 造 and り. But BudouX groups 番酒造りが as a single phrase node -> the gap falls
    # inside the word, the gate suppresses the cut (prevents 番酒造|り word split).
    seq = "番酒造りがわしらの仕事だ"
    times = {}
    t = 0.0
    for i, c in enumerate(seq):
        if c == "り":  # simulate OOV drift: insert 2.1s spurious gap before り
            t += 2.1
        times[i] = t
        t += 0.2
    words = [
        {"word": c, "start": times[i], "end": times[i] + 0.1} for i, c in enumerate(seq)
    ]
    cues = smart_split_segments(
        [_seg(words, "ja")], "ja", speech_spans=None, thresholds=TH
    )
    # 番酒造り must not be split: no adjacent cue pair where one ends with 番酒造 and the next starts with り
    joined = [c["text"] for c in cues]
    assert not any(
        a.endswith("番酒造") and b.startswith("り") for a, b in zip(joined, joined[1:])
    ), joined


def test_no_split_when_continuous():
    words = [
        {"word": c, "start": i * 0.2, "end": i * 0.2 + 0.1}
        for i, c in enumerate("あいうえお")
    ]
    cues = smart_split_segments(
        [_seg(words, "ja")], "ja", speech_spans=None, thresholds=TH
    )
    assert len(cues) == 1  # all gaps 0.1s < offline 0.7 -> no split


def test_max_cue_duration_hard_cap():
    # a 6-char cue whose aligner-assigned span stretches to 10s with no large gap
    # -> the duration hard cap must still split it (no cue longer than 7s)
    words = [
        {"word": c, "start": i * 2.0, "end": i * 2.0 + 0.1}
        for i, c in enumerate("あいうえおか")
    ]
    cues = smart_split_segments(
        [_seg(words, "ja")], "ja", speech_spans=None, thresholds=TH
    )
    assert all(c["end"] - c["start"] <= 7.0 + 1e-6 for c in cues)


def test_vad_confirmed_split():
    # gap=0.6s in the danger zone; VAD map confirms silence -> split
    words = [{"word": "は", "start": 0.0, "end": 0.1}] + [
        {"word": "い", "start": 0.7, "end": 0.8}
    ]
    spans = [(0.0, 0.15), (0.65, 0.85)]
    cues = smart_split_segments(
        [_seg(words, "ja")], "ja", speech_spans=spans, thresholds=TH
    )
    assert len(cues) == 2


def test_budoux_atom_not_split_midphrase():
    # thresholds=TH is required: it is what activates the BudouX len-break gate (do_new path);
    # without it the test becomes vacuous
    pytest.importorskip("budoux")
    # です is a single phrase node: even if length exceeds the budget, it must not be split into で|す
    words = [
        {"word": c, "start": i * 0.15, "end": i * 0.15 + 0.1}
        for i, c in enumerate("天気です")
    ]
    seg = _seg(words, "ja")
    cues = smart_split_segments(
        [seg], "ja", speech_spans=None, thresholds=TH, max_line_length=2
    )
    for c in cues:
        assert "で" not in c["text"] or "です" in c["text"].replace(
            " ", ""
        )  # です must not be split


def test_budoux_embedded_latin_alignment():
    pytest.importorskip("budoux")
    # B1 regression: CJK text with embedded Latin run (GPT-4); timing mapping must not be misaligned
    txt = "これはGPT-4です"
    words = [
        {"word": ch, "start": round(i * 0.3, 3), "end": round(i * 0.3 + 0.1, 3)}
        for i, ch in enumerate(txt)
    ]
    seg = {
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "text": txt,
        "words": words,
    }
    cues = smart_split_segments(
        [seg], "ja", speech_spans=None, thresholds=TH, max_line_length=4
    )
    assert "".join(c["text"].replace(" ", "") for c in cues) == txt  # byte-preserving
    starts = [c["start"] for c in cues]
    assert starts == sorted(starts)  # cue starts monotonic -> timing not scrambled
    assert cues[0]["start"] == 0.0  # first cue aligned to first char (no offset)


def test_budoux_lenbreak_only_at_phrase_boundary_embedded_latin():
    # coordinate-space bug regression: with an embedded Latin run, len-breaks must land on
    # real BudouX phrase boundaries (atom-index space); char-offset/atom-index mismatch must
    # not cause a mid-phrase cut. Continuous timing (no gap_break); small budget forces a len cut.
    pytest.importorskip("budoux")
    from voxweave.core.breakpoints import phrase_atoms

    txt = "GPT-4はとてもすごい技術だと思いますね"
    words = [
        {"word": ch, "start": round(i * 0.1, 3), "end": round(i * 0.1 + 0.08, 3)}
        for i, ch in enumerate(txt)
    ]
    seg = {
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "text": txt,
        "words": words,
    }
    cues = smart_split_segments(
        [seg], "ja", speech_spans=None, thresholds=TH, max_line_length=8
    )
    assert "".join(c["text"].replace(" ", "") for c in cues) == txt  # byte-preserving
    # every cue's start char offset must be a subset of BudouX phrase-start offsets (len cuts only at phrase boundaries)
    phrases = phrase_atoms(txt, "ja")
    pstarts, c = set(), 0
    for ph in phrases:
        pstarts.add(c)
        c += len(ph.replace(" ", ""))
    cstarts, off = set(), 0
    for cue in cues:
        cstarts.add(off)
        off += len(cue["text"].replace(" ", ""))
    assert cstarts <= pstarts


def test_gap_danger_zone_splits_after_topic_particle():
    # control case (proves non-vacuous): 0.6s danger-zone VAD-silence gap, line ends with は
    # (topic particle, penalty 0, good break point) -> correctly splits into two cues.
    # Confirms BudouX sets a phrase boundary here and the danger-zone gap does fire.
    pytest.importorskip("budoux")
    seq = "今日は晴れ"  # 今日は | 晴れ
    starts = [0.0, 0.2, 0.4, 1.1, 1.3]  # は ends 0.5, 晴 starts 1.1 -> gap 0.6s
    words = [{"word": c, "start": s, "end": s + 0.1} for c, s in zip(seq, starts)]
    spans = [(0.0, 0.5), (1.1, 1.5)]  # gap [0.5,1.1] silence
    cues = smart_split_segments(
        [_seg(words, "ja")], "ja", speech_spans=spans, thresholds=TH
    )
    joined = [c["text"] for c in cues]
    assert any(
        a.endswith("は") and b.startswith("晴") for a, b in zip(joined, joined[1:])
    ), joined


def test_gap_danger_zone_suppressed_after_binding_particle():
    # same 0.6s danger-zone VAD-silence gap, but line ends with の (adnominal particle in 大樹の村)
    # -> suppressed; does not cut at の|村.
    pytest.importorskip("budoux")
    seq = "大樹の村"  # 大樹の | 村
    starts = [0.0, 0.2, 0.4, 1.1]  # の ends 0.5, 村 starts 1.1 -> gap 0.6s
    words = [{"word": c, "start": s, "end": s + 0.1} for c, s in zip(seq, starts)]
    spans = [(0.0, 0.5), (1.1, 1.3)]  # gap [0.5,1.1] silence
    cues = smart_split_segments(
        [_seg(words, "ja")], "ja", speech_spans=spans, thresholds=TH
    )
    joined = [c["text"] for c in cues]
    assert not any(
        a.endswith("の") and b.startswith("村") for a, b in zip(joined, joined[1:])
    ), joined


def test_gap_real_silence_still_splits_after_binding_particle():
    # true silence (>=vad_skip 1.0s): even with の at the line end, cut unconditionally
    # (a real pause always cuts; suppression only applies in the danger zone).
    pytest.importorskip("budoux")
    seq = "大樹の村"
    starts = [0.0, 0.2, 0.4, 1.7]  # の ends 0.5, 村 starts 1.7 -> gap 1.2s >= vad_skip
    words = [{"word": c, "start": s, "end": s + 0.1} for c, s in zip(seq, starts)]
    spans = [(0.0, 0.5), (1.7, 1.9)]
    cues = smart_split_segments(
        [_seg(words, "ja")], "ja", speech_spans=spans, thresholds=TH
    )
    joined = [c["text"] for c in cues]
    assert any(
        a.endswith("の") and b.startswith("村") for a, b in zip(joined, joined[1:])
    ), joined


def test_best_len_break_pos_avoids_dangling_particle():
    # Level 1 core (deterministic unit, no BudouX dependency): cur=みんなで作った大樹の,
    # phrase-start indices 0/4/7; incoming atom=村 is a boundary. Greedy point n=10 ends
    # with の (penalty 2); candidates k=7 ends with た (penalty 0) / k=4 ends with で (penalty 0)
    # -> pick penalty-0 with the fullest line: k=7, keeping 大樹の together with 村 on the next line.
    from voxweave.core.smart_split import _best_len_break_pos

    cur = [{"text": c} for c in "みんなで作った大樹の"]
    bnd = [i in (0, 4, 7) for i in range(len(cur))]
    assert _best_len_break_pos(cur, bnd, at_boundary_next=True) == 7


def test_best_len_break_pos_greedy_when_no_penalty():
    # all candidates have penalty 0 -> fall back to greedy/fullest line (n); no unnecessary shortening.
    from voxweave.core.smart_split import _best_len_break_pos

    cur = [{"text": c} for c in "今日は晴れだ"]
    bnd = [i in (0, 3) for i in range(len(cur))]
    assert _best_len_break_pos(cur, bnd, at_boundary_next=True) == len(cur)


def test_phrase_boundary_atoms_in_atom_index_space():
    # unit regression: the boundary set must contain atom indices (all < len(atoms));
    # char offsets must not leak in (they can exceed len(atoms) when a Latin run is embedded)
    pytest.importorskip("budoux")
    from voxweave.core.smart_split import _phrase_boundary_atoms, _tokens

    txt = "GPT-4はAIです"
    atoms = [{"text": t} for t in _tokens(txt, "ja")]  # ['GPT-4','は','AI','で','す']
    b = _phrase_boundary_atoms(atoms, txt, "ja")
    assert all(
        0 <= i < len(atoms) for i in b
    )  # all are atom indices; no stale char-offset (e.g. 6) remaining
    assert 0 in b  # first atom must be a phrase start

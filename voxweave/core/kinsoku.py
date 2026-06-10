"""JIS X 4051 禁則処理 (line-breaking constraints), pure stdlib.

After lines are formed, slides breaks so no line starts with a prohibited char
(closing brackets, small kana, trailing punctuation) or ends with a prohibited
char (opening bracket/quote). Applied to ja and zh; small-kana entries are inert
in zh but the CJK punctuation rules apply to both.
"""

from __future__ import annotations

# Leading-edge prohibition (行頭禁則): these chars cannot begin a line (must hang on the previous line)
LINE_START_PROHIBITED = frozenset(
    "、。，．・：；？！）｝〕〉》」』】〙〗〟"
    "’”»"  # ' " »
    "ァィゥェォッャュョヮ"
    "ぁぃぅぇぉっゃゅょゎ"
    "ーゝゞ々‐゠–〜%"
)
# Trailing-edge prohibition (行末禁則): these chars cannot end a line
LINE_END_PROHIBITED = frozenset(
    "（｛〔〈《「『【〘〖〝‘“«([{"  # ' " «
)

# Surface heuristic (no POS): ending a line on these strands the grammatical relation to
# what follows — a case/adnominal particle binds the preceding noun forward (大樹の|村
# looks broken). High-precision subset only: ambiguous particles that double as conjunctive
# particles (接続助詞) — が adversative / から reason / で connective — are deliberately
# excluded to avoid suppressing real clause breaks.
_BIND_END_HIGH = frozenset(
    "のをにへ"
)  # case/adnominal particles, almost always binds forward
_BIND_END_MED = frozenset(
    "とまでより"
)  # と parallel/quotative, まで/より range: usually binds

# zh equivalents, whole-word semantics (the caller passes the trailing *word*, so 目的/标的
# never match — only the standalone particle/preposition does). Same high-precision policy
# as ja: words with a common clause-final reading are excluded or demoted to MED.
_ZH_BIND_END_HIGH = frozenset(
    {
        "的",  # attributive 的: standalone jieba token is virtually always the particle
        "地",
        "得",  # structural particles; standalone 得 (děi "must") also binds forward
        "把",
        "被",
        "比",
        "跟",  # prepositions: object always follows
        "和",
        "与",
        "或",
        "及",
        "而",  # conjunctions: break goes before them, never after
    }
)
_ZH_BIND_END_MED = frozenset(
    {
        # prepositions with occasional verb readings (他在/他对) — mild penalty only,
        # so they bias len-break tie-breaks but never suppress a danger-zone gap split.
        "在",
        "对",
        "从",
        "向",
        "往",
        "给",
        "让",
        "由",
        # degree adverbs that modify the following word
        "很",
        "太",
        "更",
        "最",
    }
)

_EN_TOKEN_STRIP = ".,!?;:'\"”’"


def line_end_penalty(text: str, lang: str = "") -> int:
    """Penalty for ending a line/cue on ``text`` (the trailing word or phrase).

    0 = fine, 1 = mild (likely binds forward), 2 = bad (function word/particle dangling).

    Signal source by language:
    - ja (and default): last *char* against the kana particle tables — atoms are per-char,
      and a particle is always the final char of its BudouX phrase. Always active: kana
      can't false-positive in other scripts.
    - en: whole token against breakpoints._FORBIDDEN_LEFT (articles/preps/aux/conj).
    - zh: whole word (jieba token) against the zh particle/preposition tables.
    """
    s = text.rstrip()
    if not s:
        return 0
    last = s[-1]
    if last in _BIND_END_HIGH:
        return 2
    if last in _BIND_END_MED:
        return 1
    if lang == "en":
        from .breakpoints import _FORBIDDEN_LEFT

        if s.strip(_EN_TOKEN_STRIP).lower() in _FORBIDDEN_LEFT:
            return 2
    elif lang == "zh":
        if s in _ZH_BIND_END_HIGH:
            return 2
        if s in _ZH_BIND_END_MED:
            return 1
    return 0


def apply_kinsoku(lines: list[str]) -> list[str]:
    """Nudge breaks pairwise to satisfy kinsoku constraints (oikomi/oidashi, single chars only)."""
    if len(lines) < 2:
        return list(lines)
    out = [list(line) for line in lines]
    for i in range(len(out) - 1):
        left, right = out[i], out[i + 1]
        # 行頭禁則: pull a prohibited leading char back to previous line
        while right and right[0] in LINE_START_PROHIBITED and left:
            left.append(right.pop(0))
        # 行末禁則: push a prohibited trailing char down to next line
        while left and left[-1] in LINE_END_PROHIBITED and right:
            right.insert(0, left.pop())
    return ["".join(c) for c in out if c]

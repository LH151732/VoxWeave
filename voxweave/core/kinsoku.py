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


def line_end_penalty(text: str) -> int:
    """Penalty for ending a line on the last char of ``text``.

    0 = fine, 1 = mild (likely binds forward), 2 = bad (case/adnominal particle dangling).
    Non-ja chars always score 0, so spaced languages and Latin runs are unaffected.
    """
    s = text.rstrip()
    if not s:
        return 0
    last = s[-1]
    if last in _BIND_END_HIGH:
        return 2
    if last in _BIND_END_MED:
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

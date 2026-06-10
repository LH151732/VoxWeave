# Per-language conjunction sets used to prefer breaking before a conjunction
# rather than mid-phrase. Cue breaks before these tokens produce more natural readings.
#
# Only the languages voxweave actually supports and tests (en/ja/zh) are listed.
# Tables for ~20 other languages were removed in 2026-06: they had no tests, no
# tuning, and no users — an unknown language now simply falls back to the
# even-split path in ``_fit_split_clause``, which is the same behavior an
# unlisted language always had.

conjunctions_by_language = {
    "en": {
        "and",
        "whether",
        "or",
        "as",
        "but",
        "so",
        "for",
        "nor",
        "which",
        "yet",
        "although",
        "since",
        "unless",
        "when",
        "while",
        "because",
        "if",
        "how",
        "that",
        "than",
        "who",
        "where",
        "what",
        "near",
        "before",
        "after",
        "across",
        "through",
        "until",
        "once",
        "whereas",
        "even",
        "both",
        "either",
        "neither",
        "though",
    },
    "ja": {
        "そして",
        "または",
        "しかし",
        "なぜなら",
        "もし",
        "それとも",
        "だから",
        "それに",
        "なのに",
        "そのため",
        "かつ",
        "それゆえに",
        "ならば",
        "もしくは",
        "ため",
    },
    "zh": {
        "和",
        "或",
        "但是",
        "因为",
        "任何",
        "也",
        "虽然",
        "而且",
        "所以",
        "如果",
        "除非",
        "尽管",
        "既然",
        "即使",
        "只要",
        "直到",
        "然后",
        "因此",
        "不但",
        "而是",
        "不过",
    },
}

commas_by_language = {"ja": "、", "zh": "，"}


def get_comma(lang_code: str) -> str:
    return commas_by_language.get(lang_code, ",")

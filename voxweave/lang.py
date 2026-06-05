from __future__ import annotations

# 11 languages supported by Qwen3-ForcedAligner; ISO <-> Qwen full name (all lowercase)
_ISO_TO_NAME = {
    "yue": "cantonese",
    "zh": "chinese",
    "en": "english",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "ja": "japanese",
    "ko": "korean",
    "pt": "portuguese",
    "ru": "russian",
    "es": "spanish",
}
_NAME_TO_ISO = {name: iso for iso, name in _ISO_TO_NAME.items()}


def _canon(raw: str) -> str:
    if not raw or not raw.strip():
        raise ValueError("language is required")
    return raw.strip().lower()


def is_supported(raw: str) -> bool:
    """Return True if the language is within the aligner's 11 supported languages (accepts ISO code or full name)."""
    try:
        key = _canon(raw)
    except ValueError:
        return False
    return key in _ISO_TO_NAME or key in _NAME_TO_ISO


def to_iso(raw: str) -> str:
    """Normalize full name (English) or ISO code (en) to ISO (en); used by smart_split."""
    key = _canon(raw)
    if key in _ISO_TO_NAME:
        return key
    if key in _NAME_TO_ISO:
        return _NAME_TO_ISO[key]
    raise ValueError(f"unsupported language {raw!r}")


def to_aligner_name(raw: str) -> str:
    """Normalize full name or ISO code to the Qwen full name (all lowercase); used by aligner/align."""
    key = _canon(raw)
    if key in _NAME_TO_ISO:
        return key
    if key in _ISO_TO_NAME:
        return _ISO_TO_NAME[key]
    raise ValueError(f"unsupported language {raw!r}")

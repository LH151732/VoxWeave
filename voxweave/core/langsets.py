from __future__ import annotations

# Languages written without inter-word spaces. Canonical source shared by smart_split (layout /
# joiner) and breakpoints (phrase atoms) so the set lives in exactly one place instead of being
# hand-synced across modules (the two import the other's module, so neither can own it without a
# circular import). Distinct from realign.NO_SPACE_LANGS, which intentionally includes yue and
# excludes th/lo/my for the aligner's per-unit model.
LANGUAGES_WITHOUT_SPACES = {"zh", "ja", "th", "lo", "my"}

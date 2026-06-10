"""Typed schemas for the dict shapes flowing through segmentation and timing.

These are the de facto contracts of the sibling-file pipeline (see the JSON
``word_segments`` produced by transcribe/align); TypedDicts make them explicit
so a typo'd key is a type error instead of a silently-absent value. All are
``total=False``: every consumer tolerates missing keys via ``.get()`` (units
from a timing-less edit have no spans, legacy word_data has no ``word``).
"""

from __future__ import annotations

from typing import TypedDict


class Unit(TypedDict, total=False):
    """One aligned token from an aligner / ``reinject_punct``.

    ``text`` is the unit's surface form (aligner output); pipeline word_data
    carries ``word`` instead (the ASR token used for cursor anchoring in
    ``split_at_sentence_end``). Spans are absolute seconds; either bound may be
    missing for ghost units.
    """

    text: str
    word: str
    start: float | None
    end: float | None


class Atom(TypedDict, total=False):
    """A non-breakable packing unit inside one cue (see ``_build_atoms``).

    Spaced langs: one word. No-space langs: one CJK char or Latin run.
    ``end_pen`` is the precomputed line-end break penalty attached by
    ``_attach_end_penalties`` (0 = clean break point).
    """

    text: str
    start: float | None
    end: float | None
    end_pen: int


class Cue(TypedDict, total=False):
    """One subtitle cue: display text + span + its word-level timing source.

    ``word_data`` items are :class:`Unit` shaped (pipeline cues carry
    char/word-level spans; cues built without timings carry ``[]``).
    """

    text: str
    start: float
    end: float
    word_data: list[Unit]

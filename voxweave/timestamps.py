from __future__ import annotations


def shift_units(units: list[dict], offset: float) -> list[dict]:
    """Add offset (seconds) to the start/end of each unit; returns a new list, does not mutate the input."""
    return [
        {"text": u["text"], "start": u["start"] + offset, "end": u["end"] + offset}
        for u in units
    ]

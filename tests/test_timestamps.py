from voxweave.timestamps import shift_units


def test_shift_adds_offset():
    units = [
        {"text": "a", "start": 0.4, "end": 0.7},
        {"text": "b", "start": 0.7, "end": 1.0},
    ]
    out = shift_units(units, 100.0)
    assert out == [
        {"text": "a", "start": 100.4, "end": 100.7},
        {"text": "b", "start": 100.7, "end": 101.0},
    ]


def test_zero_offset_is_identity_values():
    units = [{"text": "x", "start": 1.0, "end": 2.0}]
    assert shift_units(units, 0.0) == [{"text": "x", "start": 1.0, "end": 2.0}]


def test_does_not_mutate_input():
    units = [{"text": "x", "start": 1.0, "end": 2.0}]
    shift_units(units, 5.0)
    assert units == [{"text": "x", "start": 1.0, "end": 2.0}]


def test_empty():
    assert shift_units([], 3.0) == []

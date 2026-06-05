import pytest

from voxweave.lang import is_supported, to_aligner_name, to_iso


def test_full_name_to_iso():
    assert to_iso("English") == "en"
    assert to_iso("Chinese") == "zh"
    assert to_iso("Cantonese") == "yue"


def test_iso_to_iso_passthrough():
    assert to_iso("zh") == "zh"
    assert to_iso("EN") == "en"


def test_to_aligner_name_from_iso_and_name():
    assert to_aligner_name("zh") == "chinese"
    assert to_aligner_name("Chinese") == "chinese"
    assert to_aligner_name("en") == "english"


def test_is_supported():
    assert is_supported("zh")
    assert is_supported("English")
    assert not is_supported("klingon")
    assert not is_supported(
        "th"
    )  # smart_split knows th but the aligner does not support it


def test_unknown_raises():
    with pytest.raises(ValueError):
        to_iso("klingon")
    with pytest.raises(ValueError):
        to_aligner_name("klingon")

"""Regression tests for the phase-2 fixes in pylarevd/cli.py."""

import pytest

from pylarevd.cli import _parse_events


@pytest.mark.parametrize("spec", ["-3-5", "-3--5", "3--5"])
def test_negative_range_names_the_negative_number(spec):
    """'-3-5' split on its leading sign and failed with int('') instead."""
    with pytest.raises(ValueError, match="negative event number"):
        _parse_events(spec, 10)


def test_ranges_still_parse():
    assert _parse_events("2-4,7", 10) == [2, 3, 4, 7]
    assert _parse_events(" 0 , 1-1 ", 10) == [0, 1]
    with pytest.raises(ValueError, match="reversed range"):
        _parse_events("5-2", 10)
    with pytest.raises(ValueError, match="negative event number -2"):
        _parse_events("-2", 10)

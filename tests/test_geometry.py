"""Geometry / STL loading tests."""

import numpy as np
import pytest

from heatstl.geometry import parse_direction


def test_parse_direction_normalises():
    v = parse_direction("0,0,-2")
    assert v == pytest.approx([0.0, 0.0, -1.0])


def test_parse_direction_oblique():
    v = parse_direction("1,0,-1")
    assert v == pytest.approx([np.sqrt(0.5), 0.0, -np.sqrt(0.5)])


def test_parse_direction_bad_shape():
    with pytest.raises(ValueError):
        parse_direction("0,0")


def test_parse_direction_zero():
    with pytest.raises(ValueError):
        parse_direction("0,0,0")

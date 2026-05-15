"""Unit tests for per-facet heat-flux computation."""

import numpy as np
import pytest

from heatstl.flux import compute_face_flux, normal_flux, oblique_flux


def test_oblique_top_face_full_flux():
    # Beam pointing down (-z) hits a top face (+z normal) at full strength.
    n = np.array([[0.0, 0.0, 1.0]])
    p = np.array([0.0, 0.0, -1.0])
    q = oblique_flux(n, q0=5e6, p_hat=p)
    assert q == pytest.approx([5e6])


def test_oblique_bottom_face_zero():
    n = np.array([[0.0, 0.0, -1.0]])
    p = np.array([0.0, 0.0, -1.0])
    q = oblique_flux(n, q0=5e6, p_hat=p)
    assert q == pytest.approx([0.0])


def test_oblique_side_face_zero():
    n = np.array([[1.0, 0.0, 0.0]])
    p = np.array([0.0, 0.0, -1.0])
    q = oblique_flux(n, q0=5e6, p_hat=p)
    assert q == pytest.approx([0.0])


def test_oblique_45_deg():
    # n̂ tilted 45° from +z toward +x; p̂ = -z. cos(angle) = sqrt(2)/2.
    n = np.array([[np.sqrt(0.5), 0.0, np.sqrt(0.5)]])
    p = np.array([0.0, 0.0, -1.0])
    q = oblique_flux(n, q0=1.0, p_hat=p)
    assert q == pytest.approx([np.sqrt(0.5)])


def test_normal_mode_uniform():
    n = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0]], dtype=float)
    q = normal_flux(n, q0=2.0)
    assert q == pytest.approx([2.0, 2.0, 2.0])


def test_compute_face_flux_dispatch():
    n = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    p = np.array([0.0, 0.0, -1.0])
    assert compute_face_flux(n, q0=1.0, mode="oblique", p_hat=p) == pytest.approx([1.0, 0.0])
    assert compute_face_flux(n, q0=1.0, mode="normal") == pytest.approx([1.0, 1.0])
    with pytest.raises(ValueError):
        compute_face_flux(n, q0=1.0, mode="bogus")
    with pytest.raises(ValueError):
        compute_face_flux(n, q0=1.0, mode="oblique")  # missing p_hat

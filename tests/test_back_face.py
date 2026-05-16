"""Back-face classification under different beam directions."""

from __future__ import annotations

import numpy as np

from heatstl.solver import classify_back_facets


def test_back_face_beam_down():
    normals = np.array([
        [0.0, 0.0, 1.0],   # top — front
        [0.0, 0.0, -1.0],  # bottom — back
        [1.0, 0.0, 0.0],   # side
        [0.0, 1.0, 0.0],   # side
    ])
    p_hat = np.array([0.0, 0.0, -1.0])
    mask = classify_back_facets(normals, p_hat, tol_deg=20.0)
    assert mask.tolist() == [False, True, False, False]


def test_back_face_oblique_beam():
    # Beam at 30° from -z toward +x: p_hat = (sin30, 0, -cos30).
    p_hat = np.array([0.5, 0.0, -np.sqrt(3) / 2.0])
    # A "back" facet has n̂ near +p_hat, i.e. n̂ ≈ p_hat.
    normals = np.array([
        p_hat,                           # exactly back-pointing → match
        -p_hat,                          # front
        [0.0, 1.0, 0.0],                 # side
    ])
    mask = classify_back_facets(normals, p_hat, tol_deg=10.0)
    assert mask.tolist() == [True, False, False]

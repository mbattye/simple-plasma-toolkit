"""Shadow ray-casting against a known occluder geometry."""

from __future__ import annotations

import numpy as np
import trimesh

from heatstl.flux import shadow_mask


def test_shadow_blocking_plate():
    """A plate 0.5 m above z=0 blocks rays cast in +z (so p_hat = -z)."""
    plate = trimesh.creation.box(extents=(2.0, 2.0, 0.01))
    plate.apply_translation([0.0, 0.0, 0.5])

    centroids_under = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    centroids_outside = np.array([[10.0, 0.0, 0.0]])

    p_hat = np.array([0.0, 0.0, -1.0])  # ray cast = -p_hat = +z, hits plate above
    mask = shadow_mask(np.vstack([centroids_under, centroids_outside]), p_hat, plate)
    assert mask.tolist() == [True, True, False]


def test_no_shadow_when_occluder_empty():
    empty = trimesh.Trimesh()
    centroids = np.array([[0.0, 0.0, 0.0]])
    mask = shadow_mask(centroids, np.array([0.0, 0.0, -1.0]), empty)
    assert mask.tolist() == [False]

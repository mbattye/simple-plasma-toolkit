"""Per-facet incident heat flux + shadow ray-casting against neighbour geometry.

Sign convention:
    n̂  outward unit normal of a boundary facet.
    p̂  unit vector pointing TOWARD the surface (direction of incident beam).

    Oblique uniform:   q = q0 * max(0, -p̂ · n̂)
    Uniform normal:    q = q0 on every exposed facet
"""

from __future__ import annotations

import numpy as np
import trimesh


def oblique_flux(normals: np.ndarray, q0: float, p_hat: np.ndarray) -> np.ndarray:
    normals = np.asarray(normals, dtype=float)
    p_hat = np.asarray(p_hat, dtype=float)
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError("normals must have shape (F, 3)")
    if p_hat.shape != (3,):
        raise ValueError("p_hat must have shape (3,)")
    cos_theta = -(normals @ p_hat)
    return q0 * np.maximum(cos_theta, 0.0)


def normal_flux(normals: np.ndarray, q0: float) -> np.ndarray:
    return np.full(normals.shape[0], q0, dtype=float)


def compute_face_flux(
    normals: np.ndarray,
    q0: float,
    mode: str = "oblique",
    p_hat: np.ndarray | None = None,
) -> np.ndarray:
    if mode == "normal":
        return normal_flux(normals, q0)
    if mode == "oblique":
        if p_hat is None:
            raise ValueError("oblique mode requires p_hat")
        return oblique_flux(normals, q0, p_hat)
    raise ValueError(f"unknown flux mode {mode!r}")


def shadow_mask(
    centroids: np.ndarray,
    p_hat: np.ndarray,
    occluder: trimesh.Trimesh,
    *,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Return a boolean mask: True for facets whose ray toward the beam source
    hits the `occluder` geometry.

    A ray is cast from each centroid, nudged off the surface by `epsilon` in
    the -p̂ direction to avoid self-intersection, in direction `-p_hat`.
    """
    if len(occluder.faces) == 0:
        return np.zeros(centroids.shape[0], dtype=bool)
    origins = centroids - epsilon * p_hat
    directions = np.broadcast_to(-p_hat, origins.shape).copy()
    # `intersects_any` returns one bool per ray; cheapest API for this test.
    hits = occluder.ray.intersects_any(ray_origins=origins, ray_directions=directions)
    return np.asarray(hits, dtype=bool)

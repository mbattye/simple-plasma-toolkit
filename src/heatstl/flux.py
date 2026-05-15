"""Per-facet surface heat flux from a prescribed beam direction or uniform value.

Sign convention:
    n̂  is the outward unit normal of a boundary facet.
    p̂  is a unit vector pointing TOWARD the surface (direction of incident beam).

    Oblique uniform:   q = q0 * max(0, -p̂ · n̂)
    Uniform normal:    q = q0 on every exposed facet
"""

from __future__ import annotations

import numpy as np


def oblique_flux(
    normals: np.ndarray,
    q0: float,
    p_hat: np.ndarray,
) -> np.ndarray:
    """q_i = q0 * max(0, -p̂ · n̂_i). Vectorised over facets."""
    normals = np.asarray(normals, dtype=float)
    p_hat = np.asarray(p_hat, dtype=float)
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError("normals must have shape (F, 3)")
    if p_hat.shape != (3,):
        raise ValueError("p_hat must have shape (3,)")
    cos_theta = -(normals @ p_hat)
    return q0 * np.maximum(cos_theta, 0.0)


def normal_flux(normals: np.ndarray, q0: float) -> np.ndarray:
    """Uniform q0 on every facet — independent of orientation."""
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

"""STL loading, unit handling, neighbour tile generation, and direction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

UNIT_TO_M = {"mm": 1e-3, "m": 1.0}


# --------------------------------------------------------------------------- #
# Surface mesh container
# --------------------------------------------------------------------------- #

@dataclass
class SurfaceMesh:
    """A surface triangle mesh in SI units (metres)."""

    vertices: np.ndarray
    faces: np.ndarray
    face_normals: np.ndarray
    face_areas: np.ndarray

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    @property
    def bbox_diag(self) -> float:
        lo = self.vertices.min(axis=0)
        hi = self.vertices.max(axis=0)
        return float(np.linalg.norm(hi - lo))

    @property
    def volume(self) -> float:
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        return float(np.einsum("ij,ij->", v0, np.cross(v1, v2)) / 6.0)

    def to_trimesh(self) -> trimesh.Trimesh:
        return trimesh.Trimesh(
            vertices=self.vertices.copy(),
            faces=self.faces.copy(),
            process=False,
        )


# --------------------------------------------------------------------------- #
# STL loading and direction parsing
# --------------------------------------------------------------------------- #

def load_stl(path: str | Path, unit: str = "mm") -> SurfaceMesh:
    if unit not in UNIT_TO_M:
        raise ValueError(f"unit must be one of {list(UNIT_TO_M)}, got {unit!r}")
    scale = UNIT_TO_M[unit]

    mesh = trimesh.load(str(path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"{path}: did not load as a single triangle mesh")
    if not mesh.is_watertight:
        raise ValueError(f"{path}: STL is not watertight")

    mesh.apply_scale(scale)
    mesh.fix_normals()

    sm = SurfaceMesh(
        vertices=np.asarray(mesh.vertices, dtype=float),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        face_normals=np.asarray(mesh.face_normals, dtype=float),
        face_areas=np.asarray(mesh.area_faces, dtype=float),
    )
    if sm.volume <= 0:
        raise ValueError(
            f"{path}: signed volume {sm.volume:.3e} m^3 — normals appear inward."
        )
    return sm


def parse_direction(spec: str) -> np.ndarray:
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 3:
        raise ValueError(f"direction must be 'x,y,z', got {spec!r}")
    v = np.array([float(p) for p in parts], dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("direction vector has zero length")
    return v / n


def direction_from_angles(theta_deg: float, azimuth_deg: float = 0.0) -> np.ndarray:
    """Build p̂ (beam direction, pointing TOWARD surface) from spherical angles.

    `theta_deg = 0`  → straight-down `(0, 0, -1)`.
    `theta_deg = 90` → grazing in the +x/+y plane.
    `azimuth_deg`    → rotation about +z (0 = toward +x).
    """
    th = np.deg2rad(theta_deg)
    ph = np.deg2rad(azimuth_deg)
    return np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), -np.cos(th)])


# --------------------------------------------------------------------------- #
# Hex-tile neighbour generation (for shadowing geometry)
# --------------------------------------------------------------------------- #

def auto_hex_pitch(surf: SurfaceMesh, p_hat: np.ndarray) -> float:
    """Estimate tile centre-to-centre pitch as 2 × max half-extent of the
    geometry projected onto the plane perpendicular to p̂.

    For a flat-topped hex tile centred at origin this returns 2 × bottom
    apothem, which is the natural face-to-face spacing.
    """
    centre = surf.vertices.mean(axis=0)
    rel = surf.vertices - centre
    along = rel @ p_hat
    perp = rel - along[:, None] * p_hat
    # Build a 2D basis in the perpendicular plane.
    e1, e2 = _orthonormal_basis(p_hat)
    coords = np.column_stack([perp @ e1, perp @ e2])
    half = np.max(np.abs(coords), axis=0)
    return float(2.0 * half.max())


def hex_neighbor_offsets(
    surf: SurfaceMesh,
    p_hat: np.ndarray,
    *,
    pitch: float | None = None,
    gap: float = 0.0,
    start_angle_deg: float = 0.0,
) -> np.ndarray:
    """Return six 3D offsets placing neighbour tiles around the central one.

    Offsets lie in the plane perpendicular to p̂, at 60° increments starting
    from `start_angle_deg`. Distance from origin is `pitch + gap`.
    """
    if pitch is None:
        pitch = auto_hex_pitch(surf, p_hat)
    d = pitch + gap
    e1, e2 = _orthonormal_basis(p_hat)
    offs = []
    for k in range(6):
        ang = np.deg2rad(start_angle_deg + 60.0 * k)
        offs.append(d * (np.cos(ang) * e1 + np.sin(ang) * e2))
    return np.array(offs)


def replicate(surf: SurfaceMesh, offsets: np.ndarray) -> trimesh.Trimesh:
    """Return a single trimesh combining copies of `surf` at each offset.

    Used purely for shadow ray-casting — not for FEM. Vertices are not merged
    across copies (not needed for ray-intersection).
    """
    meshes = []
    for off in offsets:
        m = surf.to_trimesh()
        m.apply_translation(off)
        meshes.append(m)
    if not meshes:
        return trimesh.Trimesh()
    return trimesh.util.concatenate(meshes)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _orthonormal_basis(p_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two unit vectors orthogonal to p_hat and to each other."""
    p_hat = p_hat / np.linalg.norm(p_hat)
    # Pick a non-parallel reference axis.
    ref = np.array([1.0, 0.0, 0.0]) if abs(p_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(p_hat, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(p_hat, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2

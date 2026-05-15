"""STL loading, unit handling, and basic sanity checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

UNIT_TO_M = {"mm": 1e-3, "m": 1.0}


@dataclass
class SurfaceMesh:
    """A surface triangle mesh in SI units (metres)."""

    vertices: np.ndarray  # (V, 3)
    faces: np.ndarray     # (F, 3) int
    face_normals: np.ndarray  # (F, 3) outward unit normals
    face_areas: np.ndarray    # (F,)

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
        # Divergence-theorem volume; positive iff normals point outward.
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        return float(np.einsum("ij,ij->", v0, np.cross(v1, v2)) / 6.0)


def load_stl(path: str | Path, unit: str = "mm") -> SurfaceMesh:
    """Load an STL and rescale to SI (metres). Verify watertightness."""
    if unit not in UNIT_TO_M:
        raise ValueError(f"unit must be one of {list(UNIT_TO_M)}, got {unit!r}")
    scale = UNIT_TO_M[unit]

    mesh = trimesh.load(str(path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"{path}: did not load as a single triangle mesh")
    if not mesh.is_watertight:
        raise ValueError(f"{path}: STL is not watertight")

    mesh.apply_scale(scale)
    # Make sure normals are outward; trimesh fixes winding for watertight meshes.
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
    """Parse an "x,y,z" string into a unit vector."""
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 3:
        raise ValueError(f"direction must be 'x,y,z', got {spec!r}")
    v = np.array([float(p) for p in parts], dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("direction vector has zero length")
    return v / n

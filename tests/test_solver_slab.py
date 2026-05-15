"""1D slab analytic check.

A cuboid slab of thickness L in z, with:
    - q on top face (z = L)
    - T = T_cool on bottom face (z = 0)
    - sides: in v1 our default holds them at T_cool too, which breaks the
      strict 1D analytic. So we build a thin, wide slab so that side-wall
      cooling is small relative to the dominant 1D conduction, and tolerate a
      modest discrepancy.

Analytic 1D solution (sides adiabatic): T(z) = T_cool + q (L - z) / k.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from heatstl.geometry import SurfaceMesh
from heatstl.mesh import mesh_volume
from heatstl.solver import solve_steady


def _box_surface(L: float, W: float) -> SurfaceMesh:
    """Box of size W x W x L (thickness L in z), watertight."""
    box = trimesh.creation.box(extents=(W, W, L))
    # Centre at (0, 0, L/2) so z spans [0, L].
    box.apply_translation([0.0, 0.0, L / 2.0])
    return SurfaceMesh(
        vertices=np.asarray(box.vertices, dtype=float),
        faces=np.asarray(box.faces, dtype=np.int64),
        face_normals=np.asarray(box.face_normals, dtype=float),
        face_areas=np.asarray(box.area_faces, dtype=float),
    )


@pytest.mark.slow
def test_slab_1d_temperature_rise():
    L = 0.02          # 20 mm thick
    W = 0.20          # 200 mm wide  (10x thicker than tall → ~1D in centre)
    q0 = 1e6          # 1 MW/m^2
    k = 150.0
    T_cool = 300.0

    surf = _box_surface(L=L, W=W)
    vol = mesh_volume(surf, mesh_size=W / 12.0)
    result = solve_steady(
        vol, k=k, q0=q0, p_hat=np.array([0.0, 0.0, -1.0]),
        mode="oblique", T_cool=T_cool,
    )

    T_peak = float(result.T.max())
    # Analytic peak (sides adiabatic): T_cool + q L / k.
    T_peak_1d = T_cool + q0 * L / k
    # Sides hold at T_cool, so realised peak is lower. Require: somewhere in
    # the bulk we recover at least 70% of the 1D temperature rise.
    rise = T_peak - T_cool
    rise_1d = T_peak_1d - T_cool
    assert rise / rise_1d > 0.7, f"T rise {rise:.1f} vs 1D {rise_1d:.1f}"
    # And no overshoot beyond 1D.
    assert T_peak <= T_peak_1d + 1e-6

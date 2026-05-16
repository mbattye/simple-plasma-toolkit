"""1D slab analytic checks under different BC modes.

Wide-and-thin slab so side-cooling is a perturbation, then compare the bulk
peak T against the 1D analytic solution.

    Dirichlet on bottom:  T(z) = T_cool + q (L - z) / k                 (sides Dirichlet leak)
    Robin on bottom:      T(z) = T_inf + q (L - z)/k + q/h               (sides adiabatic)
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from heatstl.geometry import SurfaceMesh
from heatstl.pipeline import RunConfig, build_bc
from heatstl.solver import solve_steady


def _box_surface(L: float, W: float) -> SurfaceMesh:
    box = trimesh.creation.box(extents=(W, W, L))
    box.apply_translation([0.0, 0.0, L / 2.0])
    return SurfaceMesh(
        vertices=np.asarray(box.vertices, dtype=float),
        faces=np.asarray(box.faces, dtype=np.int64),
        face_normals=np.asarray(box.face_normals, dtype=float),
        face_areas=np.asarray(box.area_faces, dtype=float),
    )


@pytest.mark.slow
def test_slab_dirichlet_recovers_1d_rise():
    L, W = 0.02, 0.20
    q0, k, T_cool = 1e6, 150.0, 300.0
    surf = _box_surface(L=L, W=W)
    cfg = RunConfig(
        q0=q0, mode="oblique", p_hat=np.array([0.0, 0.0, -1.0]),
        k=k, bc_unheated="dirichlet", T_cool=T_cool, mesh_size_m=W / 12.0,
    )
    out = build_bc(surf, cfg)
    res = solve_steady(out.mesh_io, k=k, bc=out.bc)
    rise = res.T.max() - T_cool
    rise_1d = q0 * L / k
    assert rise / rise_1d > 0.7
    assert res.T.max() <= T_cool + rise_1d + 1e-6


@pytest.mark.slow
def test_slab_adiabatic_back_robin_matches_1d():
    """With adiabatic sides + Robin on the back face, the 1D analytic peak is
    T_inf + q L/k + q/h. Should be matched closely because there is no side
    cooling to perturb the bulk solution."""
    L, W = 0.02, 0.20
    q0, k = 1e6, 150.0
    h, T_inf = 200.0, 300.0
    surf = _box_surface(L=L, W=W)
    cfg = RunConfig(
        q0=q0, mode="oblique", p_hat=np.array([0.0, 0.0, -1.0]),
        k=k, bc_unheated="adiabatic-back-robin",
        back_h=h, back_T_inf=T_inf, back_tol_deg=20.0,
        mesh_size_m=W / 12.0,
    )
    out = build_bc(surf, cfg)
    res = solve_steady(out.mesh_io, k=k, bc=out.bc)
    T_peak_1d = T_inf + q0 * L / k + q0 / h
    rel_err = abs(res.T.max() - T_peak_1d) / (T_peak_1d - T_inf)
    assert rel_err < 0.02, f"{res.T.max():.2f} K vs analytic {T_peak_1d:.2f} K"

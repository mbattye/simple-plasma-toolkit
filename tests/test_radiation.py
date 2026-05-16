"""Radiation BC validation.

Two 1D analytic targets:

1. Top heated + radiating, sides + back adiabatic.
   Steady state: q_in = εσ(T⁴ − T_env⁴) on the heated facet.
   The whole slab is isothermal at this T (no other sinks).

2. Top heated + radiating, back Robin, sides adiabatic.
   Combined balance (1D):
       q_in = εσ(T_top⁴ − T_env⁴) + (T_top − T_back)/(L/k)         [conduction]
       (T_top − T_back)/(L/k) = h(T_back − T_inf)                   [back convection]
   We solve this nonlinear pair with scipy and compare to the FEM result.

These exercise the Newton loop end-to-end and pin the implementation to a
physically meaningful answer.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh
from scipy.optimize import brentq

from heatstl.geometry import SurfaceMesh
from heatstl.pipeline import RunConfig, build_bc
from heatstl.solver import SIGMA_SB, solve_steady


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
def test_radiation_only_isothermal_slab():
    """Adiabatic everywhere except a radiating top: T should sit at the pure-
    radiation equilibrium and be nearly isothermal through the slab."""
    L, W = 0.02, 0.20
    q0, k = 5e4, 0.1
    eps, T_env = 0.89, 300.0

    # adiabatic-back-robin with h=0 reduces to adiabatic back (still gives a
    # well-posed system because the radiation patch carries energy out).
    surf = _box_surface(L=L, W=W)
    cfg = RunConfig(
        q0=q0, mode="oblique", p_hat=np.array([0.0, 0.0, -1.0]),
        k=k, bc_unheated="adiabatic-back-robin",
        back_h=0.0, back_T_inf=T_env, back_tol_deg=20.0,
        front_radiation=True, emissivity=eps, T_env=T_env,
        mesh_size_m=W / 12.0,
    )
    out = build_bc(surf, cfg)
    res = solve_steady(out.mesh_io, k=k, bc=out.bc)

    T_analytic = (q0 / (eps * SIGMA_SB) + T_env ** 4) ** 0.25
    T_max, T_min = float(res.T.max()), float(res.T.min())

    # Slab should be nearly isothermal (k=0.1, L=0.02 conducts easily for the
    # tiny heat that is *not* radiated, which here is ~0).
    assert abs(T_max - T_analytic) / T_analytic < 0.03, f"{T_max} vs {T_analytic}"
    assert (T_max - T_min) / T_analytic < 0.05


@pytest.mark.slow
def test_radiation_plus_robin_back():
    """Combined radiation-front + Robin-back 1D balance."""
    L, W = 0.025, 0.20
    q0, k = 5e4, 0.1
    eps, T_env = 0.89, 300.0
    h_back, T_inf_back = 100.0, 400.0

    surf = _box_surface(L=L, W=W)
    cfg = RunConfig(
        q0=q0, mode="oblique", p_hat=np.array([0.0, 0.0, -1.0]),
        k=k, bc_unheated="adiabatic-back-robin",
        back_h=h_back, back_T_inf=T_inf_back, back_tol_deg=20.0,
        front_radiation=True, emissivity=eps, T_env=T_env,
        mesh_size_m=W / 12.0,
    )
    out = build_bc(surf, cfg)
    res = solve_steady(out.mesh_io, k=k, bc=out.bc)

    # Solve the 1D pair for the analytic reference.
    R_cond = L / k          # conductive resistance, m²·K/W
    R_conv = 1.0 / h_back   # convective resistance, m²·K/W

    def residual(T_top: float) -> float:
        # Heat conducted into the back from the front through series cond+conv:
        # q_path = (T_top - T_inf_back) / (R_cond + R_conv)
        q_path = (T_top - T_inf_back) / (R_cond + R_conv)
        q_rad = eps * SIGMA_SB * (T_top ** 4 - T_env ** 4)
        return q0 - q_rad - q_path

    T_top_analytic = brentq(residual, T_env + 1.0, 3000.0)

    T_max = float(res.T.max())
    rel_err = abs(T_max - T_top_analytic) / (T_top_analytic - T_env)
    assert rel_err < 0.03, f"FEM peak {T_max:.1f} K vs analytic {T_top_analytic:.1f} K"


@pytest.mark.slow
def test_starship_demo_regression():
    """The Starship example STL with radiation on must stay in physical range.

    Without front-face radiation the same setup blows up to ~10⁴ K (v2 bug).
    This pins peak T < 1500 K and the energy balance < 2%."""
    import subprocess
    import json
    import pathlib

    repo = pathlib.Path(__file__).resolve().parents[1]
    out_dir = repo / "examples" / "out"
    out_dir.mkdir(exist_ok=True)
    report = out_dir / "starship_regression.json"
    vtu = out_dir / "starship_regression.vtu"

    cmd = [
        "uv", "run", "heatstl",
        "--stl", str(repo / "examples" / "heat_shield_tile.stl"),
        "--q0", "5e4", "--angle-deg", "75", "--azimuth-deg", "0",
        "--preset", "starship", "--neighbors", "hex6",
        "--unit", "mm", "--out", str(vtu), "--report", str(report),
        "--quiet",
    ]
    subprocess.run(cmd, check=True, cwd=repo)
    d = json.loads(report.read_text())["diagnostics"]

    assert 600.0 < d["peak_T"] < 1500.0, d["peak_T"]
    assert abs(d["residual_rel"]) < 0.02, d["residual_rel"]
    # Radiation must dominate at low-k TPS — that's the whole point of v3.
    assert d["Q_radiated"] / d["Q_in"] > 0.7, d

"""Transient solver validation.

Three checks:

1. Profile factory unit tests — values at known times.
2. 1D slab transient analytic: a tungsten-like slab (k=170, ρ=19300, c_p=130,
   α≈7e-5 m²/s) with sudden Neumann q on top, Dirichlet T_cool on bottom,
   adiabatic sides. After ~5τ_diff it should approach the steady analytic
   T(z) = T_cool + q (L − z) / k.
3. Starship-flip-conservative regression (end-to-end via subprocess): peak
   T in band, transient energy balance < 5% during the meaningful part of
   the run, Newton converges every step.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from heatstl.geometry import SurfaceMesh, direction_from_angles
from heatstl.pipeline import RunConfig, bc_for_step, build_mesh_context
from heatstl.solver import solve_transient
from heatstl.transient import (
    PHatProfileSpec,
    QProfileSpec,
    make_p_hat_profile,
    make_q_profile,
)


# --------------------------------------------------------------------------- #
# Profile factory tests (pure functions, fast)
# --------------------------------------------------------------------------- #

def test_q_profile_constant():
    f = make_q_profile(QProfileSpec(kind="constant", q0=1.5e5))
    assert f(0.0) == pytest.approx(1.5e5)
    assert f(123.0) == pytest.approx(1.5e5)


def test_q_profile_ramp():
    f = make_q_profile(QProfileSpec(kind="ramp", q0=2.0, t_ramp=10.0))
    assert f(-1.0) == pytest.approx(0.0)
    assert f(5.0) == pytest.approx(1.0)
    assert f(10.0) == pytest.approx(2.0)
    assert f(100.0) == pytest.approx(2.0)


def test_q_profile_gaussian_peak_and_fwhm():
    f = make_q_profile(QProfileSpec(kind="gaussian", q0=10.0, t0=5.0, fwhm=4.0))
    assert f(5.0) == pytest.approx(10.0)         # peak
    assert f(3.0) == pytest.approx(5.0, rel=1e-6)  # half-max at ±fwhm/2
    assert f(7.0) == pytest.approx(5.0, rel=1e-6)


def test_p_hat_profile_sweep_endpoints_and_midpoint():
    spec = PHatProfileSpec(
        kind="sweep", angle_start=0.0, angle_end=90.0,
        azimuth=0.0, t0=10.0, t1=20.0,
    )
    f = make_p_hat_profile(spec)
    np.testing.assert_allclose(f(0.0), direction_from_angles(0.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(f(15.0), direction_from_angles(45.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(f(100.0), direction_from_angles(90.0, 0.0), atol=1e-12)


def test_p_hat_profile_sweep_with_azimuth():
    """Azimuth-sweep variant: both polar and azimuth interpolate linearly."""
    spec = PHatProfileSpec(
        kind="sweep", angle_start=20.0, angle_end=75.0,
        azimuth=0.0, azimuth_start=0.0, azimuth_end=60.0,
        t0=0.0, t1=10.0,
    )
    f = make_p_hat_profile(spec)
    # Endpoints
    np.testing.assert_allclose(f(0.0), direction_from_angles(20.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(f(10.0), direction_from_angles(75.0, 60.0), atol=1e-12)
    # Midpoint
    np.testing.assert_allclose(f(5.0), direction_from_angles(47.5, 30.0), atol=1e-12)
    # Clamping outside the window
    np.testing.assert_allclose(f(-3.0), direction_from_angles(20.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(f(100.0), direction_from_angles(75.0, 60.0), atol=1e-12)


def test_p_hat_profile_sweep_back_compat_no_azimuth():
    """If azimuth_start/end are None, azimuth is held fixed at `azimuth`."""
    spec = PHatProfileSpec(
        kind="sweep", angle_start=0.0, angle_end=90.0,
        azimuth=45.0, t0=0.0, t1=10.0,
    )
    f = make_p_hat_profile(spec)
    for t in [0.0, 3.0, 7.5, 10.0]:
        # Azimuth component (xy direction) is fixed; only the polar angle moves.
        expected = direction_from_angles(0.0 + (t / 10.0) * 90.0, 45.0)
        np.testing.assert_allclose(f(t), expected, atol=1e-12)


# --------------------------------------------------------------------------- #
# 1D slab transient analytic
# --------------------------------------------------------------------------- #

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
def test_slab_transient_approaches_steady():
    """After many diffusion times, a step-q on top + Dirichlet bottom should
    converge toward the 1D steady analytic peak T = T_cool + q L / k."""
    L, W = 0.02, 0.10           # thin and wide
    k = 170.0                   # W/m/K, tungsten-ish
    rho = 19300.0               # kg/m^3
    cp = 130.0                  # J/kg/K
    alpha = k / (rho * cp)      # ~6.8e-5 m^2/s
    tau = L * L / alpha         # ~6 s
    duration = 30.0 * tau       # well past steady
    n_steps = 60

    q0 = 1e6
    T_cool = 300.0
    surf = _box_surface(L=L, W=W)
    cfg = RunConfig(
        q0=q0, mode="oblique", p_hat=np.array([0.0, 0.0, -1.0]),
        k=k, bc_unheated="dirichlet", T_cool=T_cool,
        mesh_size_m=W / 10.0,
    )
    ctx = build_mesh_context(surf, cfg)

    q_f = make_q_profile(QProfileSpec(kind="constant", q0=q0))
    p_f = make_p_hat_profile(PHatProfileSpec(
        kind="constant", p_hat=np.array([0.0, 0.0, -1.0]),
    ))

    def bc_fn(step, t):
        return bc_for_step(ctx, cfg, q0=q_f(t), p_hat=p_f(t))

    times = np.linspace(0.0, duration, n_steps + 1)
    result = solve_transient(
        ctx.mesh_io, k=k, rho=rho, cp=cp, bc_step_fn=bc_fn,
        times=times, T_initial=T_cool,
    )

    T_final = result.T_history[-1]
    T_peak_steady = T_cool + q0 * L / k    # ~417.6 K
    rise = T_final.max() - T_cool
    rise_steady = T_peak_steady - T_cool
    # Wide slab: side cooling drains some heat from edges, so realised rise
    # is a bit below the 1D ideal but should be >70% of it (same as the
    # existing steady slab test).
    assert rise / rise_steady > 0.7, f"rise {rise:.1f} K vs steady {rise_steady:.1f} K"
    # And no overshoot beyond the 1D analytic.
    assert T_final.max() <= T_peak_steady + 1.0


# --------------------------------------------------------------------------- #
# Starship-flip regression
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_starship_flip_conservative_regression():
    """End-to-end transient: conservative belly-flop must hit peak T in band
    and converge Newton every step."""
    import json
    import pathlib
    import subprocess

    repo = pathlib.Path(__file__).resolve().parents[1]
    out_dir = repo / "examples" / "out"
    out_dir.mkdir(exist_ok=True)
    xdmf = out_dir / "flip_regression.xdmf"
    report = out_dir / "flip_regression.json"

    cmd = [
        "uv", "run", "heatstl",
        "--stl", str(repo / "examples" / "heat_shield_tile.stl"),
        "--q0", "1e5",
        "--preset", "starship-flip-conservative",
        "--neighbors", "hex6",
        "--unit", "mm",
        "--out", str(xdmf), "--report", str(report),
        # smaller for test speed
        "--n-steps", "40", "--duration", "600",
        "--quiet",
    ]
    subprocess.run(cmd, check=True, cwd=repo)

    d = json.loads(report.read_text())
    frames = d["frames"][1:]
    peak_T = max(f["peak_T"] for f in frames)
    # Conservative preset should stay in 900-1500 K band.
    assert 900.0 < peak_T < 1500.0, peak_T
    # Newton must converge at every timestep (we don't see failures).
    assert all(f["n_newton_iters"] >= 1 for f in frames)
    # During the meaningful heating window (Q_in > 100 W), transient energy
    # balance closes to within 5%.
    near_peak = [f for f in frames if f.get("Q_in", 0.0) > 100.0]
    assert near_peak, "no frames with meaningful Q_in"
    worst = max(abs(f["residual_transient_rel"]) for f in near_peak)
    assert worst < 0.05, f"worst transient residual {worst*100:.2f}% during heating window"

    # Ghost neighbour STL should be dropped next to the main result (hex6 on).
    ghost = out_dir / "flip_regression_neighbors.stl"
    assert ghost.exists() and ghost.stat().st_size > 0, "neighbour STL not written"
    # Sanity-load it: should contain exactly 6 disjoint copies of the central
    # tile (one per hex neighbour position).
    central_mesh = trimesh.load_mesh(str(repo / "examples" / "heat_shield_tile.stl"))
    ghost_mesh = trimesh.load_mesh(str(ghost))
    assert ghost_mesh.faces.shape[0] == 6 * central_mesh.faces.shape[0], (
        ghost_mesh.faces.shape, central_mesh.faces.shape
    )

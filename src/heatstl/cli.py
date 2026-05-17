"""Command-line entry point: heatstl."""

from __future__ import annotations

import sys
import time as wall_time
from pathlib import Path

import click
import numpy as np

from . import __version__
from .diagnostics import compute as compute_diagnostics
from .diagnostics import compute_frame
from .geometry import UNIT_TO_M, direction_from_angles, load_stl, parse_direction
from .io import (
    write_neighbours_stl,
    write_report,
    write_transient_report,
    write_vtu,
    write_vtu_frames,
    write_xdmf_arrow_timeseries,
    write_xdmf_timeseries,
)
from .pipeline import RunConfig, bc_for_step, build_bc, build_mesh_context
from .presets import PRESETS
from .solver import solve_steady, solve_transient
from .transient import (
    PHatProfileSpec,
    QProfileSpec,
    TransientConfig,
    make_p_hat_profile,
    make_q_profile,
)


BC_MODES = [
    "dirichlet",
    "robin",
    "adiabatic-back-dirichlet",
    "adiabatic-back-robin",
]
NEIGHBORS = ["none", "hex6"]
Q_PROFILES = ["constant", "ramp", "gaussian", "piecewise"]
ANGLE_PROFILES = ["constant", "sweep", "piecewise"]


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--stl", "stl_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--q0", required=True, type=float, help="Peak heat flux, W/m^2 (amplitude in transient mode).")
# Beam direction
@click.option("--direction", default=None, help="Beam direction p_hat as 'x,y,z'.")
@click.option("--angle-deg", default=None, type=float, help="Polar angle of beam from -z (deg).")
@click.option("--azimuth-deg", default=0.0, show_default=True, type=float)
@click.option("--mode", type=click.Choice(["oblique", "normal"]), default="oblique", show_default=True)
# Material
@click.option("--k", "k", default=None, type=float, help="Thermal conductivity, W/m/K (default 150).")
@click.option("--rho", default=None, type=float, help="Density, kg/m^3 (transient only).")
@click.option("--cp", default=None, type=float, help="Specific heat, J/kg/K (transient only).")
# BC options
@click.option("--bc-unheated", type=click.Choice(BC_MODES), default=None)
@click.option("--T-cool", "T_cool", default=300.0, show_default=True, type=float)
@click.option("--h", "h_conv", default=100.0, show_default=True, type=float)
@click.option("--T-inf", "T_inf", default=300.0, show_default=True, type=float)
@click.option("--back-h", "back_h", default=None, type=float)
@click.option("--back-T-inf", "back_T_inf", default=None, type=float)
@click.option("--back-tol-deg", default=30.0, show_default=True, type=float)
@click.option("--back-axis", default=None, help="Geometric back direction as 'x,y,z'.")
# Radiation
@click.option("--front-radiation/--no-front-radiation", "front_radiation", default=None)
@click.option("--emissivity", default=None, type=float)
@click.option("--T-env", "T_env", default=None, type=float)
@click.option("--newton-tol", default=1e-4, show_default=True, type=float)
@click.option("--newton-max-iter", default=50, show_default=True, type=int)
# Neighbours
@click.option("--neighbors", type=click.Choice(NEIGHBORS), default="none", show_default=True)
@click.option("--tile-pitch", default=None, type=float)
@click.option("--tile-gap", default=0.0, show_default=True, type=float)
# Transient
@click.option("--transient/--steady", "transient", default=None, help="Run a time-dependent solve. Default: steady (presets may override).")
@click.option("--duration", default=None, type=float, help="Total simulation time, s.")
@click.option("--n-steps", default=None, type=int, help="Number of timesteps.")
@click.option("--T-initial", "T_initial", default=300.0, show_default=True, type=float)
@click.option("--q-profile", type=click.Choice(Q_PROFILES), default=None, help="q(t) profile (default constant).")
@click.option("--q-csv", default=None, type=click.Path(dir_okay=False), help="For 'piecewise': 2-col CSV of (t, q).")
@click.option("--q-ramp-t", default=1.0, show_default=True, type=float)
@click.option("--q-t0", default=0.0, show_default=True, type=float, help="Gaussian peak time, s.")
@click.option("--q-fwhm", default=1.0, show_default=True, type=float, help="Gaussian FWHM, s.")
@click.option("--angle-profile", type=click.Choice(ANGLE_PROFILES), default=None, help="p_hat(t) profile (default constant).")
@click.option("--angle-csv", default=None, type=click.Path(dir_okay=False), help="For 'piecewise': 3-col CSV (t, theta_deg, phi_deg).")
@click.option("--angle-start", default=None, type=float, help="Sweep: start polar angle (deg).")
@click.option("--angle-end", default=None, type=float, help="Sweep: end polar angle (deg).")
@click.option("--azimuth-start", default=None, type=float, help="Sweep: start azimuth (deg). Default holds --azimuth-deg fixed.")
@click.option("--azimuth-end", default=None, type=float, help="Sweep: end azimuth (deg).")
@click.option("--angle-t0", default=None, type=float, help="Sweep: start time, s.")
@click.option("--angle-t1", default=None, type=float, help="Sweep: end time, s.")
# Mesh / units / IO
@click.option("--unit", type=click.Choice(["mm", "m"]), default="mm", show_default=True)
@click.option("--mesh-size", default=None, type=float)
@click.option("--out", "out_path", default=None, type=click.Path(dir_okay=False), help="Output: VTU (steady) or XDMF (transient).")
@click.option("--report", "report_path", default=None, type=click.Path(dir_okay=False))
@click.option("--vtu-frames", "vtu_frames_pattern", default=None, type=str, help="Transient: also write numbered VTUs at this pattern (e.g. 'out/flip_{:04d}.vtu').")
# Presets
@click.option("--preset", type=click.Choice(sorted(PRESETS)), default=None)
@click.option("--verbose/--quiet", default=False)
@click.version_option(__version__)
def main(**opts):
    """Apply a prescribed heat flux to an STL and solve heat conduction
    (steady or transient)."""
    t_wall0 = wall_time.perf_counter()
    pre = PRESETS.get(opts["preset"], {}) if opts["preset"] else {}

    # ---- Resolve preset defaults for un-set flags ----
    def _resolve(key, default, pre_key=None):
        if opts.get(key) is None:
            return pre.get(pre_key or key, default)
        return opts[key]

    k = _resolve("k", 150.0)
    rho = _resolve("rho", 0.0)
    cp = _resolve("cp", 0.0)
    bc_unheated = _resolve("bc_unheated", "dirichlet")
    back_h = _resolve("back_h", 100.0)
    back_T_inf = _resolve("back_T_inf", 400.0)
    back_tol_deg = pre.get("back_tol_deg", opts["back_tol_deg"]) if opts["back_tol_deg"] == 30.0 else opts["back_tol_deg"]
    back_axis_str = opts["back_axis"] if opts["back_axis"] is not None else pre.get("back_axis")
    back_axis_vec = parse_direction(back_axis_str) if back_axis_str else None
    front_radiation = bool(_resolve("front_radiation", False))
    emissivity = _resolve("emissivity", 0.89)
    T_env = _resolve("T_env", 300.0)
    transient = bool(_resolve("transient", False))
    duration = _resolve("duration", 0.0)
    n_steps = _resolve("n_steps", 0)
    q_profile_kind = _resolve("q_profile", "constant", pre_key="q_profile")
    angle_profile_kind = _resolve("angle_profile", "constant", pre_key="angle_profile")
    q_t0 = pre.get("q_t0", opts["q_t0"]) if opts["q_t0"] == 0.0 else opts["q_t0"]
    q_fwhm = pre.get("q_fwhm", opts["q_fwhm"]) if opts["q_fwhm"] == 1.0 else opts["q_fwhm"]
    q_ramp_t = pre.get("q_ramp_t", opts["q_ramp_t"]) if opts["q_ramp_t"] == 1.0 else opts["q_ramp_t"]
    angle_start = _resolve("angle_start", None)
    angle_end = _resolve("angle_end", None)
    azimuth_start = _resolve("azimuth_start", None)
    azimuth_end = _resolve("azimuth_end", None)
    angle_t0 = _resolve("angle_t0", None)
    angle_t1 = _resolve("angle_t1", None)

    # ---- Beam direction (steady or 'initial' for transient) ----
    if opts["direction"] is not None and opts["angle_deg"] is not None:
        raise click.UsageError("Use either --direction OR --angle-deg, not both.")
    if opts["angle_deg"] is not None:
        p_hat = direction_from_angles(opts["angle_deg"], opts["azimuth_deg"])
    elif opts["direction"] is not None:
        p_hat = parse_direction(opts["direction"])
    else:
        p_hat = parse_direction("0,0,-1")

    # ---- Default output paths depending on mode ----
    out_path = opts["out_path"] or ("result.xdmf" if transient else "result.vtu")
    report_path = opts["report_path"] or "result.json"
    verbose = opts["verbose"]

    # ---- Common: load STL ----
    surf = load_stl(opts["stl_path"], unit=opts["unit"])
    if verbose:
        click.echo(
            f"[heatstl] preset={opts['preset'] or 'none'} mode={'transient' if transient else 'steady'} "
            f"surface={surf.n_faces} tris, bbox diag={surf.bbox_diag:.4f} m"
        )

    mesh_size_m = (opts["mesh_size"] * UNIT_TO_M[opts["unit"]]) if opts["mesh_size"] is not None else None
    tile_pitch_m = (opts["tile_pitch"] * UNIT_TO_M[opts["unit"]]) if opts["tile_pitch"] is not None else None

    cfg = RunConfig(
        q0=opts["q0"], mode=opts["mode"], p_hat=p_hat, k=k,
        bc_unheated=bc_unheated, T_cool=opts["T_cool"],
        h=opts["h_conv"], T_inf=opts["T_inf"],
        back_h=back_h, back_T_inf=back_T_inf, back_tol_deg=back_tol_deg,
        back_axis=back_axis_vec,
        front_radiation=front_radiation, emissivity=emissivity, T_env=T_env,
        newton_tol=opts["newton_tol"], newton_max_iter=opts["newton_max_iter"],
        neighbors=opts["neighbors"], tile_pitch=tile_pitch_m, tile_gap=opts["tile_gap"] * UNIT_TO_M[opts["unit"]],
        mesh_size_m=mesh_size_m,
    )

    if not transient:
        _run_steady(cfg, surf, opts, out_path, report_path, k, p_hat, t_wall0)
        return

    # ---- Transient validation ----
    if duration <= 0 or n_steps <= 0:
        raise click.UsageError("Transient mode requires positive --duration and --n-steps.")
    if rho <= 0 or cp <= 0:
        raise click.UsageError("Transient mode requires positive --rho and --cp (or use a preset that sets them).")

    q_spec = QProfileSpec(
        kind=q_profile_kind, q0=opts["q0"],
        t_ramp=q_ramp_t, t0=q_t0, fwhm=q_fwhm, csv=opts["q_csv"],
    )
    angle_spec = PHatProfileSpec(
        kind=angle_profile_kind, p_hat=p_hat,
        angle_start=angle_start if angle_start is not None else 0.0,
        angle_end=angle_end if angle_end is not None else 0.0,
        azimuth=opts["azimuth_deg"],
        azimuth_start=azimuth_start,
        azimuth_end=azimuth_end,
        t0=angle_t0 if angle_t0 is not None else 0.0,
        t1=angle_t1 if angle_t1 is not None else max(duration, 1.0),
        csv=opts["angle_csv"],
    )
    cfg_trans = TransientConfig(
        duration=duration, n_steps=n_steps, rho=rho, cp=cp,
        T_initial=opts["T_initial"], q_profile=q_spec, p_hat_profile=angle_spec,
    )

    _run_transient(cfg, cfg_trans, surf, opts, out_path, report_path,
                   k, rho, cp, p_hat, t_wall0, verbose)


# --------------------------------------------------------------------------- #
# Steady dispatcher
# --------------------------------------------------------------------------- #

def _run_steady(cfg, surf, opts, out_path, report_path, k, p_hat, t_wall0):
    verbose = opts["verbose"]
    if verbose:
        click.echo("[heatstl] meshing + classifying boundary facets…")
    out = build_bc(surf, cfg)

    if verbose:
        n_tets = sum(cb.data.shape[0] for cb in out.mesh_io.cells if cb.type == "tetra")
        click.echo(
            f"[heatstl] mesh: {n_tets} tets, "
            f"heated={out.n_central_heated} shadowed={out.n_shadowed}, "
            f"robin={out.bc.robin_facets.size} dirichlet={out.bc.dirichlet_facets.size} "
            f"radiation={out.bc.radiation_facets.size}"
        )

    result = solve_steady(
        out.mesh_io, k=k, bc=out.bc,
        newton_tol=cfg.newton_tol, newton_max_iter=cfg.newton_max_iter,
    )
    diag = compute_diagnostics(result, k=k)
    if verbose and out.bc.radiation_facets.size > 0:
        click.echo(
            f"[heatstl] Newton: {result.n_newton_iters} iters, "
            f"final rel update = {result.newton_residual:.2e}"
        )

    write_vtu(out_path, result)

    # Ghost neighbour tiles (drop the same lattice copies used for shadowing
    # as a translucent STL — useful for ParaView visualisation).
    neighbours_uri: str | None = None
    if out.occluder is not None:
        out_p = Path(out_path)
        neighbours_path = out_p.with_name(out_p.stem + "_neighbors.stl")
        write_neighbours_stl(neighbours_path, out.occluder)
        neighbours_uri = str(neighbours_path)
        if verbose:
            click.echo(f"[heatstl] wrote ghost neighbours {neighbours_path}")

    meta = {
        "version": __version__, "stl": str(Path(opts["stl_path"]).resolve()),
        "preset": opts["preset"], "q0_W_m2": opts["q0"], "p_hat": p_hat.tolist(),
        "mode": opts["mode"], "k_W_m_K": k,
        "bc_unheated": cfg.bc_unheated, "T_cool_K": cfg.T_cool,
        "h_W_m2_K": cfg.h, "T_inf_K": cfg.T_inf,
        "back_h_W_m2_K": cfg.back_h, "back_T_inf_K": cfg.back_T_inf,
        "back_tol_deg": cfg.back_tol_deg,
        "back_axis": cfg.back_axis.tolist() if cfg.back_axis is not None else None,
        "neighbors": cfg.neighbors, "n_shadowed_facets": out.n_shadowed,
        "front_radiation": cfg.front_radiation, "emissivity": cfg.emissivity, "T_env_K": cfg.T_env,
        "newton_tol": cfg.newton_tol, "newton_max_iter": cfg.newton_max_iter,
        "wall_seconds": wall_time.perf_counter() - t_wall0,
    }
    write_report(report_path, diag, meta)

    click.echo(f"[heatstl] wrote {out_path} and {report_path}")
    click.echo(
        f"[heatstl] peak T = {diag.peak_T:.2f} K  |  "
        f"Q_in = {diag.Q_in:.3e} W  |  Q_rad = {diag.Q_radiated:.3e} W  |  "
        f"Q_cond = {diag.Q_conducted_out:.3e} W  |  residual = {diag.residual_rel*100:+.2f}%"
    )


# --------------------------------------------------------------------------- #
# Transient dispatcher
# --------------------------------------------------------------------------- #

def _run_transient(cfg, cfg_trans, surf, opts, out_path, report_path,
                   k, rho, cp, p_hat, t_wall0, verbose):
    if verbose:
        click.echo("[heatstl] building mesh context (one-shot)…")
    ctx = build_mesh_context(surf, cfg)
    if verbose:
        n_tets = sum(cb.data.shape[0] for cb in ctx.mesh_io.cells if cb.type == "tetra")
        click.echo(
            f"[heatstl] mesh: {n_tets} tets, back_facets={int(ctx.back_mask.sum())}, "
            f"radiation_facets={int(ctx.radiation_mask.sum())}, "
            f"occluder={'yes' if ctx.occluder is not None else 'no'}"
        )

    q_of_t = make_q_profile(cfg_trans.q_profile)
    p_of_t = make_p_hat_profile(cfg_trans.p_hat_profile)

    def bc_step_fn(step_idx, t):
        return bc_for_step(ctx, cfg, q0=q_of_t(t), p_hat=p_of_t(t))

    times = cfg_trans.time_grid()
    if verbose:
        click.echo(
            f"[heatstl] transient: duration={cfg_trans.duration:g}s, "
            f"n_steps={cfg_trans.n_steps}, dt={cfg_trans.duration/cfg_trans.n_steps:g}s"
        )

    def progress(step, t, T_max, n_iters):
        if verbose:
            click.echo(
                f"  step {step:>4d}/{cfg_trans.n_steps} "
                f"t={t:8.2f}s  T_max={T_max:8.2f} K  newton={n_iters}"
            )

    result = solve_transient(
        ctx.mesh_io, k=k, rho=rho, cp=cp,
        bc_step_fn=bc_step_fn, times=times, T_initial=cfg_trans.T_initial,
        newton_tol=cfg.newton_tol, newton_max_iter=cfg.newton_max_iter,
        progress=progress,
    )

    # Per-step diagnostics. Note on energy balance:
    #   transient: Q_in - Q_out - dU/dt ≈ 0   where dU/dt = ∫ρc_p ∂T/∂t dV
    #   we compute dU/dt via the mass matrix M as (M @ (T_n+1 - T_n)).sum()/dt
    # The 'residual_rel' field from `compute_frame` is the *steady-state*
    # balance and is only meaningful near temporal extrema.
    per_step = []
    for i, T in enumerate(result.T_history):
        if i == 0:
            per_step.append({
                "t_s": float(times[0]),
                "peak_T": float(T.max()),
                "min_T": float(T.min()),
                "newton_iters": 0,
            })
            continue
        bc_i = result.bc_history[i - 1]
        diag = compute_frame(result.mesh, T, bc_i, k=k,
                             n_newton_iters=result.n_newton_history[i - 1])
        d = diag.as_dict()
        d["t_s"] = float(times[i])
        d["q0_W_m2"] = float(q_of_t(float(times[i])))
        d["p_hat"] = p_of_t(float(times[i])).tolist()
        d["n_shadowed"] = result.n_shadowed_history[i - 1]

        # Transient energy balance.
        dt = float(times[i] - times[i - 1])
        dT = T - result.T_history[i - 1]
        dU_dt = float((result.mass_matrix @ dT).sum() / dt)  # W
        d["dU_dt_W"] = dU_dt
        if d["Q_in"] != 0.0:
            d["residual_transient_rel"] = (d["Q_in"] - d["Q_out_total"] - dU_dt) / d["Q_in"]
        else:
            d["residual_transient_rel"] = float("nan")
        per_step.append(d)

    write_xdmf_timeseries(out_path, result)
    if opts["vtu_frames_pattern"]:
        n_written = len(write_vtu_frames(opts["vtu_frames_pattern"], result))
        if verbose:
            click.echo(f"[heatstl] wrote {n_written} VTU frames at {opts['vtu_frames_pattern']!r}")

    # Companion arrow XDMF: a single point above the tile carrying p̂(t) as
    # a vector field. Lets the user open both files in ParaView and see an
    # animated arrow indicating beam direction as the attitude sweeps.
    out_p = Path(out_path)
    arrow_path = out_p.with_name(out_p.stem + "_arrow.xdmf")
    up_dir = (-cfg.back_axis) if cfg.back_axis is not None else np.array([0.0, 0.0, 1.0])
    anchor = surf.vertices.mean(axis=0) + 0.5 * surf.bbox_diag * up_dir
    arrow_length = 0.3 * surf.bbox_diag
    p_hat_history = [p_of_t(float(t)) for t in times]
    q0_history = [float(q_of_t(float(t))) for t in times]
    write_xdmf_arrow_timeseries(
        arrow_path, times, p_hat_history, q0_history,
        anchor=anchor, arrow_length=arrow_length,
    )
    if verbose:
        click.echo(f"[heatstl] wrote arrow companion {arrow_path}")

    # Ghost neighbour STL (only when hex6 neighbours are enabled).
    if ctx.occluder is not None:
        neighbours_path = out_p.with_name(out_p.stem + "_neighbors.stl")
        write_neighbours_stl(neighbours_path, ctx.occluder)
        if verbose:
            click.echo(f"[heatstl] wrote ghost neighbours {neighbours_path}")

    meta = {
        "version": __version__, "stl": str(Path(opts["stl_path"]).resolve()),
        "preset": opts["preset"], "transient": True,
        "duration_s": cfg_trans.duration, "n_steps": cfg_trans.n_steps,
        "rho_kg_m3": rho, "cp_J_kg_K": cp, "k_W_m_K": k,
        "T_initial_K": cfg_trans.T_initial,
        "q_profile": cfg_trans.q_profile.__dict__,
        "p_hat_profile": {
            **cfg_trans.p_hat_profile.__dict__,
            "p_hat": (
                cfg_trans.p_hat_profile.p_hat.tolist()
                if cfg_trans.p_hat_profile.p_hat is not None else None
            ),
        },
        "bc_unheated": cfg.bc_unheated, "back_h_W_m2_K": cfg.back_h,
        "back_T_inf_K": cfg.back_T_inf, "back_axis": cfg.back_axis.tolist() if cfg.back_axis is not None else None,
        "front_radiation": cfg.front_radiation, "emissivity": cfg.emissivity, "T_env_K": cfg.T_env,
        "neighbors": cfg.neighbors, "wall_seconds": wall_time.perf_counter() - t_wall0,
    }
    write_transient_report(report_path, meta, times, per_step)

    # Summary
    peak = max(d["peak_T"] for d in per_step)
    t_peak = per_step[int(np.argmax([d["peak_T"] for d in per_step]))]["t_s"]
    click.echo(f"[heatstl] wrote {out_path} and {report_path}")
    click.echo(
        f"[heatstl] peak T over run = {peak:.2f} K at t = {t_peak:.2f} s  "
        f"|  wall = {wall_time.perf_counter() - t_wall0:.1f} s"
    )


if __name__ == "__main__":
    sys.exit(main())

"""Command-line entry point: heatstl."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click

from . import __version__
from .diagnostics import compute as compute_diagnostics
from .geometry import UNIT_TO_M, direction_from_angles, load_stl, parse_direction
from .io import write_report, write_vtu
from .pipeline import RunConfig, build_bc
from .presets import PRESETS
from .solver import solve_steady


BC_MODES = [
    "dirichlet",
    "robin",
    "adiabatic-back-dirichlet",
    "adiabatic-back-robin",
]
NEIGHBORS = ["none", "hex6"]


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--stl", "stl_path", required=True, type=click.Path(exists=True, dir_okay=False), help="Watertight STL file.")
@click.option("--q0", required=True, type=float, help="Peak heat flux, W/m^2.")
# Direction: either explicit cartesian, or spherical angles. Mutually exclusive.
@click.option("--direction", default=None, help="Beam direction p_hat as 'x,y,z' (pointing toward surface).")
@click.option("--angle-deg", default=None, type=float, help="Polar angle (deg) of beam from -z. 0 = straight down.")
@click.option("--azimuth-deg", default=0.0, show_default=True, type=float, help="Azimuth (deg) of beam in xy-plane, from +x.")
@click.option("--mode", type=click.Choice(["oblique", "normal"]), default="oblique", show_default=True)
# Material
@click.option("--k", "k", default=None, type=float, help="Thermal conductivity, W/m/K. (Default 150; --preset overrides.)")
# BC options
@click.option("--bc-unheated", type=click.Choice(BC_MODES), default=None, help="BC on non-heated facets. Default 'dirichlet'; --preset may override.")
@click.option("--T-cool", "T_cool", default=300.0, show_default=True, type=float, help="Dirichlet temperature, K.")
@click.option("--h", "h_conv", default=100.0, show_default=True, type=float, help="Robin convective coefficient, W/m^2/K (used by 'robin' mode).")
@click.option("--T-inf", "T_inf", default=300.0, show_default=True, type=float, help="Robin sink temperature, K (used by 'robin' mode).")
@click.option("--back-h", "back_h", default=None, type=float, help="Robin h on auto-detected back face, W/m^2/K.")
@click.option("--back-T-inf", "back_T_inf", default=None, type=float, help="Robin T_inf on back face, K.")
@click.option("--back-tol-deg", default=30.0, show_default=True, type=float, help="Angle tolerance (deg) for back-face detection.")
@click.option("--back-axis", default=None, help="Geometric back direction as 'x,y,z' (default: same as beam --direction). Pin this for oblique beams so back-face detection stays stable.")
# Front-face radiation (nonlinear)
@click.option("--front-radiation/--no-front-radiation", "front_radiation", default=None, help="Re-radiation εσ(T⁴-T_env⁴) on heated facets. Default off; 'starship' preset turns it on.")
@click.option("--emissivity", default=None, type=float, help="Emissivity ε of the heated face (default 0.89 for silica TPS).")
@click.option("--T-env", "T_env", default=None, type=float, help="Radiative sink temperature, K (default 300).")
@click.option("--newton-tol", default=1e-4, show_default=True, type=float, help="Relative tolerance for the radiation Newton loop.")
@click.option("--newton-max-iter", default=50, show_default=True, type=int, help="Max iterations for the radiation Newton loop.")
# Neighbours
@click.option("--neighbors", type=click.Choice(NEIGHBORS), default="none", show_default=True, help="Surround central tile with neighbours and apply shadowing.")
@click.option("--tile-pitch", default=None, type=float, help="Centre-to-centre tile pitch (STL units). Auto: 2 × projected half-extent.")
@click.option("--tile-gap", default=0.0, show_default=True, type=float, help="Gap added to tile pitch (STL units).")
# Mesh / units / IO
@click.option("--unit", type=click.Choice(["mm", "m"]), default="mm", show_default=True)
@click.option("--mesh-size", default=None, type=float, help="gmsh target element size, in STL units. Default: bbox_diag/30.")
@click.option("--out", "out_path", default="result.vtu", show_default=True, type=click.Path(dir_okay=False))
@click.option("--report", "report_path", default="result.json", show_default=True, type=click.Path(dir_okay=False))
# Presets
@click.option("--preset", type=click.Choice(sorted(PRESETS)), default=None, help="Apply a built-in defaults preset (e.g. 'starship').")
@click.option("--verbose/--quiet", default=False)
@click.version_option(__version__)
def main(
    stl_path,
    q0,
    direction,
    angle_deg,
    azimuth_deg,
    mode,
    k,
    bc_unheated,
    T_cool,
    h_conv,
    T_inf,
    back_h,
    back_T_inf,
    back_tol_deg,
    back_axis,
    front_radiation,
    emissivity,
    T_env,
    newton_tol,
    newton_max_iter,
    neighbors,
    tile_pitch,
    tile_gap,
    unit,
    mesh_size,
    out_path,
    report_path,
    preset,
    verbose,
):
    """Apply a prescribed heat flux to an STL and solve steady heat conduction."""
    t0 = time.perf_counter()

    # Apply preset for any options the user didn't set explicitly.
    preset_data = PRESETS.get(preset, {}) if preset else {}
    if k is None:
        k = preset_data.get("k", 150.0)
    if bc_unheated is None:
        bc_unheated = preset_data.get("bc_unheated", "dirichlet")
    if back_h is None:
        back_h = preset_data.get("back_h", 100.0)
    if back_T_inf is None:
        back_T_inf = preset_data.get("back_T_inf", 400.0)
    if "back_tol_deg" in preset_data and back_tol_deg == 30.0:
        back_tol_deg = preset_data["back_tol_deg"]
    if back_axis is None:
        back_axis = preset_data.get("back_axis")
    back_axis_vec = parse_direction(back_axis) if back_axis else None
    if front_radiation is None:
        front_radiation = bool(preset_data.get("front_radiation", False))
    if emissivity is None:
        emissivity = preset_data.get("emissivity", 0.89)
    if T_env is None:
        T_env = preset_data.get("T_env", 300.0)

    # Beam direction.
    if direction is not None and angle_deg is not None:
        raise click.UsageError("Use either --direction OR --angle-deg/--azimuth-deg, not both.")
    if angle_deg is not None:
        p_hat = direction_from_angles(angle_deg, azimuth_deg)
    elif direction is not None:
        p_hat = parse_direction(direction)
    else:
        p_hat = parse_direction("0,0,-1")

    if verbose:
        click.echo(
            f"[heatstl] preset={preset or 'none'}  k={k} W/m/K  bc={bc_unheated}  "
            f"neighbors={neighbors}  p_hat={p_hat.round(4).tolist()}"
        )

    surf = load_stl(stl_path, unit=unit)
    if verbose:
        click.echo(
            f"[heatstl] surface: {surf.n_faces} tris, bbox diag = {surf.bbox_diag:.4f} m, "
            f"volume = {surf.volume:.3e} m^3"
        )

    mesh_size_m = (mesh_size * UNIT_TO_M[unit]) if mesh_size is not None else None
    tile_pitch_m = (tile_pitch * UNIT_TO_M[unit]) if tile_pitch is not None else None
    tile_gap_m = tile_gap * UNIT_TO_M[unit]

    cfg = RunConfig(
        q0=q0,
        mode=mode,
        p_hat=p_hat,
        k=k,
        bc_unheated=bc_unheated,
        T_cool=T_cool,
        h=h_conv,
        T_inf=T_inf,
        back_h=back_h,
        back_T_inf=back_T_inf,
        back_tol_deg=back_tol_deg,
        back_axis=back_axis_vec,
        front_radiation=front_radiation,
        emissivity=emissivity,
        T_env=T_env,
        newton_tol=newton_tol,
        newton_max_iter=newton_max_iter,
        neighbors=neighbors,
        tile_pitch=tile_pitch_m,
        tile_gap=tile_gap_m,
        mesh_size_m=mesh_size_m,
    )

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
        click.echo(f"[heatstl] solving…")

    result = solve_steady(
        out.mesh_io, k=k, bc=out.bc,
        newton_tol=newton_tol, newton_max_iter=newton_max_iter,
    )
    diag = compute_diagnostics(result, k=k)
    if verbose and out.bc.radiation_facets.size > 0:
        click.echo(
            f"[heatstl] Newton: {result.n_newton_iters} iters, "
            f"final rel update = {result.newton_residual:.2e}"
        )

    write_vtu(out_path, result)
    meta = {
        "version": __version__,
        "stl": str(Path(stl_path).resolve()),
        "preset": preset,
        "q0_W_m2": q0,
        "p_hat": p_hat.tolist(),
        "mode": mode,
        "k_W_m_K": k,
        "bc_unheated": bc_unheated,
        "T_cool_K": T_cool,
        "h_W_m2_K": h_conv,
        "T_inf_K": T_inf,
        "back_h_W_m2_K": back_h,
        "back_T_inf_K": back_T_inf,
        "back_tol_deg": back_tol_deg,
        "back_axis": back_axis_vec.tolist() if back_axis_vec is not None else None,
        "neighbors": neighbors,
        "tile_pitch_user_units": tile_pitch,
        "tile_gap_user_units": tile_gap,
        "unit": unit,
        "n_shadowed_facets": out.n_shadowed,
        "front_radiation": front_radiation,
        "emissivity": emissivity,
        "T_env_K": T_env,
        "newton_tol": newton_tol,
        "newton_max_iter": newton_max_iter,
        "wall_seconds": time.perf_counter() - t0,
    }
    write_report(report_path, diag, meta)

    click.echo(f"[heatstl] wrote {out_path} and {report_path}")
    click.echo(
        f"[heatstl] peak T = {diag.peak_T:.2f} K  |  "
        f"Q_in = {diag.Q_in:.3e} W  |  "
        f"Q_rad = {diag.Q_radiated:.3e} W  |  "
        f"Q_cond = {diag.Q_conducted_out:.3e} W  |  "
        f"residual = {diag.residual_rel*100:+.2f}%"
    )


if __name__ == "__main__":
    sys.exit(main())

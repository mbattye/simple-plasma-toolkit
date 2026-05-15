"""Command-line entry point: heatstl."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
import numpy as np

from . import __version__
from .diagnostics import compute as compute_diagnostics
from .geometry import load_stl, parse_direction
from .io import write_report, write_vtu
from .mesh import mesh_volume
from .solver import solve_steady


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--stl", "stl_path", required=True, type=click.Path(exists=True, dir_okay=False), help="Watertight STL file.")
@click.option("--q0", required=True, type=float, help="Peak heat flux, W/m^2.")
@click.option("--direction", default="0,0,-1", show_default=True, help="Beam direction p_hat as 'x,y,z' (pointing toward surface).")
@click.option("--mode", type=click.Choice(["oblique", "normal"]), default="oblique", show_default=True)
@click.option("--T-cool", "T_cool", default=300.0, show_default=True, type=float, help="Dirichlet temperature on non-heated faces, K.")
@click.option("--k", "k", default=150.0, show_default=True, type=float, help="Thermal conductivity, W/m/K.")
@click.option("--unit", type=click.Choice(["mm", "m"]), default="mm", show_default=True)
@click.option("--mesh-size", default=None, type=float, help="gmsh target element size, in STL units. Default: bbox_diag/30.")
@click.option("--out", "out_path", default="result.vtu", show_default=True, type=click.Path(dir_okay=False))
@click.option("--report", "report_path", default="result.json", show_default=True, type=click.Path(dir_okay=False))
@click.option("--verbose/--quiet", default=False)
@click.version_option(__version__)
def main(
    stl_path: str,
    q0: float,
    direction: str,
    mode: str,
    T_cool: float,
    k: float,
    unit: str,
    mesh_size: float | None,
    out_path: str,
    report_path: str,
    verbose: bool,
) -> None:
    """Apply a prescribed heat flux to an STL and solve steady heat conduction."""
    from .geometry import UNIT_TO_M
    t0 = time.perf_counter()

    p_hat = parse_direction(direction)

    if verbose:
        click.echo(f"[heatstl] loading {stl_path} (unit={unit})")
    surf = load_stl(stl_path, unit=unit)
    if verbose:
        click.echo(
            f"[heatstl] surface: {surf.n_faces} tris, bbox diag = {surf.bbox_diag:.4f} m, "
            f"volume = {surf.volume:.3e} m^3"
        )

    mesh_size_m = (mesh_size * UNIT_TO_M[unit]) if mesh_size is not None else None
    if verbose:
        ms = mesh_size_m if mesh_size_m is not None else surf.bbox_diag / 30.0
        click.echo(f"[heatstl] gmsh meshing (target size = {ms:.4f} m)…")
    vol = mesh_volume(surf, mesh_size=mesh_size_m, verbose=verbose)
    if verbose:
        n_tets = sum(cb.data.shape[0] for cb in vol.cells if cb.type == "tetra")
        click.echo(f"[heatstl] volume mesh: {n_tets} tets")

    if verbose:
        click.echo(f"[heatstl] solving (mode={mode}, q0={q0:g} W/m^2, k={k} W/m/K, T_cool={T_cool} K)…")
    result = solve_steady(
        vol,
        k=k,
        q0=q0,
        p_hat=p_hat,
        mode=mode,
        T_cool=T_cool,
    )

    diag = compute_diagnostics(result, k=k)
    if verbose:
        click.echo(
            f"[heatstl] peak T = {diag.peak_T:.2f} K, "
            f"Q_in = {diag.Q_in:.3e} W, Q_out = {diag.Q_out:.3e} W, "
            f"residual = {diag.residual_rel*100:+.2f}%"
        )

    write_vtu(out_path, result)
    meta = {
        "version": __version__,
        "stl": str(Path(stl_path).resolve()),
        "q0_W_m2": q0,
        "direction": p_hat.tolist(),
        "mode": mode,
        "T_cool_K": T_cool,
        "k_W_m_K": k,
        "unit": unit,
        "mesh_size_user": mesh_size,
        "wall_seconds": time.perf_counter() - t0,
    }
    write_report(report_path, diag, meta)

    click.echo(f"[heatstl] wrote {out_path} and {report_path}")
    # Summary line, regardless of verbose:
    click.echo(
        f"[heatstl] peak T = {diag.peak_T:.2f} K  |  "
        f"Q_in = {diag.Q_in:.3e} W  |  energy residual = {diag.residual_rel*100:+.2f}%"
    )


if __name__ == "__main__":
    sys.exit(main())

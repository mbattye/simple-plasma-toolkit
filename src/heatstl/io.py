"""VTU / XDMF time-series + JSON output writers."""

from __future__ import annotations

import json
from pathlib import Path

import meshio
import numpy as np

from .diagnostics import Diagnostics
from .solver import SolveResult, TransientResult


def write_vtu(path: str | Path, result: SolveResult) -> None:
    mesh = result.mesh
    points = mesh.p.T
    tets = mesh.t.T
    out = meshio.Mesh(
        points=points,
        cells=[("tetra", tets)],
        point_data={"T": np.asarray(result.T)},
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.write(str(path))


def write_report(path: str | Path, diag: Diagnostics, meta: dict) -> None:
    payload = {"meta": meta, "diagnostics": diag.as_dict()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2))


# --------------------------------------------------------------------------- #
# Transient outputs
# --------------------------------------------------------------------------- #

def write_xdmf_timeseries(path: str | Path, result: TransientResult) -> None:
    """Write the mesh + per-step T field as an XDMF + HDF5 pair.

    ParaView opens the .xdmf and scrubs through time directly.

    Note: meshio's TimeSeriesWriter writes the .h5 file using only the
    basename (in the current working directory), so we cd into the output
    directory while writing to keep the .h5 next to the .xdmf.
    """
    import os
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    points = result.mesh.p.T
    tets = result.mesh.t.T
    prev_cwd = Path.cwd()
    try:
        os.chdir(path.parent)
        with meshio.xdmf.TimeSeriesWriter(path.name) as writer:
            writer.write_points_cells(points, [("tetra", tets)])
            for t, T in zip(result.times, result.T_history):
                writer.write_data(float(t), point_data={"T": np.asarray(T)})
    finally:
        os.chdir(prev_cwd)


def write_vtu_frames(pattern: str | Path, result: TransientResult) -> list[str]:
    """Write one .vtu per timestep using a pattern like 'out/flip_{:04d}.vtu'.

    Returns the list of written paths.
    """
    pattern_str = str(pattern)
    Path(pattern_str).parent.mkdir(parents=True, exist_ok=True)
    points = result.mesh.p.T
    tets = result.mesh.t.T
    written: list[str] = []
    for i, T in enumerate(result.T_history):
        p = pattern_str.format(i)
        m = meshio.Mesh(
            points=points,
            cells=[("tetra", tets)],
            point_data={"T": np.asarray(T)},
        )
        m.write(p)
        written.append(p)
    return written


def write_transient_report(
    path: str | Path,
    meta: dict,
    times: np.ndarray,
    per_step_diag: list[dict],
) -> None:
    payload = {
        "meta": meta,
        "time_s": [float(t) for t in times],
        "frames": per_step_diag,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2))

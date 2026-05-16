"""VTU + JSON output writers."""

from __future__ import annotations

import json
from pathlib import Path

import meshio
import numpy as np

from .diagnostics import Diagnostics
from .solver import SolveResult


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

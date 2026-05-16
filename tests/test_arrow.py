"""Arrow companion XDMF is written and readable, with correct contents."""

from __future__ import annotations

import pathlib
import subprocess

import meshio
import numpy as np
import pytest


@pytest.mark.slow
def test_arrow_xdmf_written_and_has_correct_direction_history():
    repo = pathlib.Path(__file__).resolve().parents[1]
    out_dir = repo / "examples" / "out"
    out_dir.mkdir(exist_ok=True)
    xdmf = out_dir / "flip_arrow_test.xdmf"
    arrow_xdmf = out_dir / "flip_arrow_test_arrow.xdmf"
    report = out_dir / "flip_arrow_test.json"

    subprocess.run(
        [
            "uv", "run", "heatstl",
            "--stl", str(repo / "examples" / "heat_shield_tile.stl"),
            "--q0", "1e5",
            "--preset", "starship-flip-conservative",
            "--neighbors", "none",                 # speed: skip shadowing
            "--unit", "mm",
            "--out", str(xdmf),
            "--report", str(report),
            "--n-steps", "10", "--duration", "100",
            "--quiet",
        ],
        check=True, cwd=repo,
    )

    assert arrow_xdmf.exists(), arrow_xdmf

    # Read the time series and pull out per-step incident_unit.
    with meshio.xdmf.TimeSeriesReader(str(arrow_xdmf)) as reader:
        points, cells = reader.read_points_cells()
        assert points.shape == (1, 3), points.shape
        directions: list[np.ndarray] = []
        for k in range(reader.num_steps):
            _t, pd, _cd = reader.read_data(k)
            assert "incident_unit" in pd
            v = np.asarray(pd["incident_unit"])
            assert v.shape == (1, 3)
            directions.append(v[0])

    # The conservative-flip preset sweeps polar from 60° to 90° between
    # t=100 and t=500 s, with t0_run=0. At t=100 the sweep is just
    # starting; at t<=100 all directions equal the start direction (60°
    # polar, azimuth 0). So in this short 100-second test the direction
    # should be effectively constant.
    # Each direction is unit p̂ scaled by arrow_length, but direction unit
    # should match up to that scale.
    norms = np.linalg.norm(directions, axis=1)
    assert np.allclose(norms, norms[0], rtol=1e-6), norms
    # All directions should point in the same direction (the sweep hasn't
    # really begun varying yet for the short window).
    for d in directions[1:]:
        assert np.allclose(d / norms[0], directions[0] / norms[0], atol=1e-6)

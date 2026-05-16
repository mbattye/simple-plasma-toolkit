"""Volume meshing of an STL using the gmsh Python API.

We import the STL as the boundary surface and let gmsh tetrahedralise the
interior. The result is returned as a meshio mesh for downstream consumption.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import gmsh
import meshio
import numpy as np

from .geometry import SurfaceMesh


@contextmanager
def _gmsh_session(verbose: bool = False):
    # gmsh.initialize() unconditionally installs a SIGINT handler with
    # signal.signal(...), which raises ValueError when called outside the
    # main thread (e.g. from a FastAPI sync-endpoint threadpool). We
    # monkey-patch signal.signal around the initialise call so that
    # specific error is swallowed; signal handling is only useful for
    # interactive gmsh CLI use, not for our headless mesher.
    import signal as _signal
    _real_signal = _signal.signal

    def _safe_signal(sig, handler):
        try:
            return _real_signal(sig, handler)
        except ValueError:
            return None

    _signal.signal = _safe_signal
    try:
        gmsh.initialize()
    finally:
        _signal.signal = _real_signal
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        yield
    finally:
        gmsh.finalize()


def mesh_volume(
    surf: SurfaceMesh,
    mesh_size: float | None = None,
    *,
    verbose: bool = False,
) -> meshio.Mesh:
    """Volume-mesh the watertight SurfaceMesh and return a meshio Mesh of tets.

    Parameters
    ----------
    surf : SurfaceMesh
        Already-scaled-to-SI surface mesh (units: metres).
    mesh_size : float, optional
        Target element size in metres. Defaults to bbox_diag / 30.
    """
    if mesh_size is None:
        mesh_size = surf.bbox_diag / 30.0

    with TemporaryDirectory() as td:
        # gmsh's STL classifier is the easiest path: write the (SI-scaled) STL,
        # let gmsh build a volume from the closed surface.
        stl_path = Path(td) / "surf.stl"
        _write_stl(surf, stl_path)

        with _gmsh_session(verbose=verbose):
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
            gmsh.merge(str(stl_path))
            # Classify surface entities so we can build a volume.
            angle = np.deg2rad(40.0)
            gmsh.model.mesh.classifySurfaces(angle, True, True, np.deg2rad(180))
            gmsh.model.mesh.createGeometry()
            surfaces = gmsh.model.getEntities(2)
            loop = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfaces])
            gmsh.model.geo.addVolume([loop])
            gmsh.model.geo.synchronize()
            gmsh.model.mesh.generate(3)

            msh_path = Path(td) / "volume.msh"
            gmsh.write(str(msh_path))

        mesh = meshio.read(str(msh_path))

    # Keep only tets and the boundary triangles; drop lower-order entities.
    cells = {}
    for cb in mesh.cells:
        if cb.type in ("tetra", "triangle"):
            cells.setdefault(cb.type, []).append(cb.data)
    if "tetra" not in cells:
        raise RuntimeError("gmsh produced no tetrahedra")
    new_cells = [
        meshio.CellBlock(t, np.vstack(cells[t])) for t in ("triangle", "tetra") if t in cells
    ]
    return meshio.Mesh(points=mesh.points, cells=new_cells)


def _write_stl(surf: SurfaceMesh, path: Path) -> None:
    """Write a SurfaceMesh out as binary STL via meshio."""
    meshio.write_points_cells(
        str(path),
        points=surf.vertices,
        cells=[("triangle", surf.faces)],
        file_format="stl",
    )

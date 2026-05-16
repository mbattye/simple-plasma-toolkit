"""Post-processing diagnostics: peak T, total heat-in / heat-out, energy balance."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import skfem
from skfem import ElementTetP1, FacetBasis

from .solver import SolveResult


@dataclass
class Diagnostics:
    peak_T: float
    min_T: float
    Q_in: float
    Q_out: float
    residual_rel: float
    n_tets: int
    n_boundary_facets: int
    n_heated_facets: int
    n_robin_facets: int
    n_dirichlet_facets: int

    def as_dict(self) -> dict:
        return asdict(self)


def _facet_areas(mesh: skfem.MeshTet, facet_indices: np.ndarray) -> np.ndarray:
    f_nodes = mesh.facets[:, facet_indices]
    p = mesh.p
    a = p[:, f_nodes[0]].T
    b = p[:, f_nodes[1]].T
    c = p[:, f_nodes[2]].T
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def _conduction_flux_out(
    mesh: skfem.MeshTet, T: np.ndarray, k: float, facets: np.ndarray
) -> float:
    """∫ -k ∇T · n̂ dA over the given facets (positive = energy leaving the body)."""
    if facets.size == 0:
        return 0.0
    fb = FacetBasis(mesh, ElementTetP1(), facets=facets)
    gradT = fb.interpolate(T).grad
    n_qp = fb.normals
    integrand = -k * np.einsum("ijk,ijk->jk", gradT, n_qp)
    return float(np.sum(integrand * fb.dx))


def compute(result: SolveResult, k: float) -> Diagnostics:
    mesh = result.mesh
    bc = result.bc

    if bc.heated_facets.size > 0:
        A_heated = _facet_areas(mesh, bc.heated_facets)
        Q_in = float(np.sum(bc.q_heated * A_heated))
    else:
        Q_in = 0.0

    # Sum the conducted flux out over every BC patch that can carry energy:
    # Dirichlet, Robin. Adiabatic facets contribute 0 to good precision.
    sink_facets = np.concatenate([bc.dirichlet_facets, bc.robin_facets]).astype(np.int64)
    Q_out = _conduction_flux_out(mesh, result.T, k, sink_facets)

    residual = (Q_in - Q_out) / Q_in if Q_in != 0 else float("nan")

    return Diagnostics(
        peak_T=float(np.max(result.T)),
        min_T=float(np.min(result.T)),
        Q_in=Q_in,
        Q_out=Q_out,
        residual_rel=float(residual),
        n_tets=int(mesh.t.shape[1]),
        n_boundary_facets=int(result.boundary_indices.size),
        n_heated_facets=int(bc.heated_facets.size),
        n_robin_facets=int(bc.robin_facets.size),
        n_dirichlet_facets=int(bc.dirichlet_facets.size),
    )

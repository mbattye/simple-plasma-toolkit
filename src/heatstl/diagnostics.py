"""Post-processing diagnostics: peak T, total heat-in / heat-out, energy balance."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import skfem
from skfem import ElementTetP1, FacetBasis
from skfem.helpers import dot, grad

from .solver import SolveResult


@dataclass
class Diagnostics:
    peak_T: float
    min_T: float
    Q_in: float           # W, total heat in across heated facets
    Q_out: float          # W, total -k∇T·n̂ integrated over cooled facets
    residual_rel: float   # (Q_in - Q_out) / Q_in
    n_tets: int
    n_boundary_facets: int

    def as_dict(self) -> dict:
        return asdict(self)


def _facet_areas(mesh: skfem.MeshTet, facet_indices: np.ndarray) -> np.ndarray:
    f_nodes = mesh.facets[:, facet_indices]
    p = mesh.p
    a = p[:, f_nodes[0]].T
    b = p[:, f_nodes[1]].T
    c = p[:, f_nodes[2]].T
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def compute(result: SolveResult, k: float) -> Diagnostics:
    mesh = result.mesh

    # Heat in: simple area-weighted sum of per-facet q.
    if result.facets_heated.size > 0:
        A_heated = _facet_areas(mesh, result.facets_heated)
        Q_in = float(np.sum(result.q_on_heated * A_heated))
    else:
        Q_in = 0.0

    # Heat out: integrate -k ∇T · n̂ over cooled facets.
    if result.facets_cooled.size > 0:
        fb = FacetBasis(mesh, ElementTetP1(), facets=result.facets_cooled)
        T_at_qp = fb.interpolate(result.T)
        gradT_qp = T_at_qp.grad  # (3, n_facets, n_qp)
        # Outward unit normals from FacetBasis quadrature.
        n_qp = fb.normals  # (3, n_facets, n_qp)
        # Integrand: -k ∇T · n̂   ; integrate via quadrature weights.
        integrand = -k * np.einsum("ijk,ijk->jk", gradT_qp, n_qp)
        Q_out = float(np.sum(integrand * fb.dx))
    else:
        Q_out = 0.0

    residual = (Q_in - Q_out) / Q_in if Q_in != 0 else float("nan")

    return Diagnostics(
        peak_T=float(np.max(result.T)),
        min_T=float(np.min(result.T)),
        Q_in=Q_in,
        Q_out=Q_out,
        residual_rel=float(residual),
        n_tets=int(mesh.t.shape[1]),
        n_boundary_facets=int(result.boundary_indices.size),
    )

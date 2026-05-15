"""Steady-state heat conduction FEM solve on a tet mesh using scikit-fem.

Governing PDE:   -∇·(k ∇T) = 0  in Ω
Neumann BC:      -k ∂T/∂n = q   on Γ_heated   (incoming flux)
Dirichlet BC:    T = T_cool                    on Γ_cool

We integrate q against test functions to form the boundary load vector, then
condense out Dirichlet DOFs and solve a single sparse linear system.
"""

from __future__ import annotations

from dataclasses import dataclass

import meshio
import numpy as np
import skfem
from scipy.sparse.linalg import spsolve
from skfem import Basis, BilinearForm, ElementTetP1, FacetBasis, LinearForm, MeshTet
from skfem.helpers import dot, grad


@dataclass
class SolveResult:
    T: np.ndarray                     # nodal temperatures, K
    mesh: skfem.MeshTet
    basis: Basis
    facets_heated: np.ndarray         # boundary facet indices
    facets_cooled: np.ndarray
    q_on_heated: np.ndarray           # per-heated-facet flux, W/m^2
    boundary_normals: np.ndarray      # all boundary facets, (Nb, 3)
    boundary_indices: np.ndarray      # facet indices of every boundary facet


def _meshio_to_skfem(mesh: meshio.Mesh) -> skfem.MeshTet:
    pts = np.asarray(mesh.points, dtype=float)
    tets = None
    for cb in mesh.cells:
        if cb.type == "tetra":
            tets = np.asarray(cb.data, dtype=np.int64)
            break
    if tets is None:
        raise ValueError("meshio mesh has no tetra cells")
    # skfem expects (3, N) and (4, Ne).
    return MeshTet(pts.T, tets.T)


def _boundary_facet_normals(mesh: skfem.MeshTet) -> tuple[np.ndarray, np.ndarray]:
    """Return (facet_indices, outward_unit_normals) for boundary facets."""
    bidx = mesh.boundary_facets()
    # facets array is (3, n_facets) of node indices.
    f_nodes = mesh.facets[:, bidx]
    p = mesh.p  # (3, n_nodes)
    a = p[:, f_nodes[0]]
    b = p[:, f_nodes[1]]
    c = p[:, f_nodes[2]]
    n = np.cross((b - a).T, (c - a).T)  # (n_facets, 3)
    norms = np.linalg.norm(n, axis=1)
    n_unit = n / norms[:, None]

    # Make outward: each boundary facet belongs to exactly one tetra; flip if
    # it points toward the opposing vertex centroid.
    # mesh.f2t is (2, n_facets); for boundary facets the second row is -1.
    f2t = mesh.f2t
    tet = f2t[0, bidx]
    centroids = mesh.p[:, mesh.t[:, tet]].mean(axis=1).T  # (n_facets, 3)
    face_centroids = ((a + b + c) / 3.0).T                # (n_facets, 3)
    outward = face_centroids - centroids
    flip = np.einsum("ij,ij->i", n_unit, outward) < 0.0
    n_unit[flip] *= -1.0
    return bidx, n_unit


def solve_steady(
    mesh_io: meshio.Mesh,
    *,
    k: float,
    q0: float,
    p_hat: np.ndarray,
    mode: str,
    T_cool: float,
) -> SolveResult:
    """Assemble and solve the linear steady heat-conduction system."""
    from .flux import compute_face_flux

    mesh = _meshio_to_skfem(mesh_io)
    basis = Basis(mesh, ElementTetP1())

    bidx, b_normals = _boundary_facet_normals(mesh)
    q_all = compute_face_flux(b_normals, q0=q0, mode=mode, p_hat=p_hat)
    heated_mask = q_all > 0.0
    facets_heated = bidx[heated_mask]
    facets_cooled = bidx[~heated_mask]
    q_heated = q_all[heated_mask]

    @BilinearForm
    def a_form(u, v, _w):
        return k * dot(grad(u), grad(v))

    A = a_form.assemble(basis)

    # Boundary load: ∫ q v dA over heated facets.
    if facets_heated.size > 0:
        fb = FacetBasis(mesh, ElementTetP1(), facets=facets_heated)
        # FacetBasis preserves the order of `facets`, so row i of dx
        # corresponds to facets_heated[i]. Broadcast per-facet q to (F, n_qp).
        q_qp = np.broadcast_to(q_heated[:, None], fb.dx.shape).copy()

        @LinearForm
        def b_form(v, w):
            return w["q"] * v

        b = b_form.assemble(fb, q=q_qp)
    else:
        b = np.zeros(basis.N)

    # Dirichlet on cooled facets.
    if facets_cooled.size == 0:
        raise ValueError(
            "No cooled facets — problem would be singular. "
            "Check that p_hat does not give nonzero flux on every facet."
        )
    D = basis.get_dofs(facets=facets_cooled).all()
    T = np.full(basis.N, np.nan)
    T[D] = T_cool

    T = skfem.solve(*skfem.condense(A, b, x=T, D=D), solver=lambda M, rhs: spsolve(M, rhs))

    return SolveResult(
        T=np.asarray(T),
        mesh=mesh,
        basis=basis,
        facets_heated=facets_heated,
        facets_cooled=facets_cooled,
        q_on_heated=q_heated,
        boundary_normals=b_normals,
        boundary_indices=bidx,
    )

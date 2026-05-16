"""Steady-state heat conduction FEM solve on a tet mesh using scikit-fem.

PDE:        -∇·(k ∇T) = 0  in Ω
Neumann:    -k ∂T/∂n = q                on Γ_heated   (incoming flux)
Dirichlet:  T = T_D                      on Γ_D
Robin:      -k ∂T/∂n = h (T - T_inf)     on Γ_R
Natural:    -k ∂T/∂n = 0                 on every other boundary facet (adiabatic)

A `BCAssembly` collects per-facet membership + per-facet data for each kind of
BC. The solver assembles a single linear system and condenses out Dirichlet
DOFs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import meshio
import numpy as np
import skfem
from scipy.sparse.linalg import spsolve
from skfem import Basis, BilinearForm, ElementTetP1, FacetBasis, LinearForm, MeshTet
from skfem.helpers import dot, grad


# --------------------------------------------------------------------------- #
# Boundary-condition assembly container
# --------------------------------------------------------------------------- #

@dataclass
class BCAssembly:
    """Per-facet BC membership and parameter arrays.

    All `*_facets` are global facet indices into the skfem mesh; the
    parallel parameter arrays are the same length as their facet array.
    """

    heated_facets: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    q_heated: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))

    dirichlet_facets: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    T_dirichlet: float = 0.0

    robin_facets: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    h_robin: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    T_inf_robin: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))

    def validate(self) -> None:
        if self.heated_facets.shape != self.q_heated.shape:
            raise ValueError("heated_facets and q_heated length mismatch")
        if self.robin_facets.shape != self.h_robin.shape:
            raise ValueError("robin_facets and h_robin length mismatch")
        if self.robin_facets.shape != self.T_inf_robin.shape:
            raise ValueError("robin_facets and T_inf_robin length mismatch")
        if self.dirichlet_facets.size == 0 and self.robin_facets.size == 0:
            raise ValueError(
                "No Dirichlet or Robin facets — problem is singular. "
                "All boundary facets would be adiabatic."
            )


# --------------------------------------------------------------------------- #
# Mesh utilities
# --------------------------------------------------------------------------- #

@dataclass
class SolveResult:
    T: np.ndarray
    mesh: skfem.MeshTet
    basis: Basis
    bc: BCAssembly
    boundary_indices: np.ndarray   # all boundary facet indices
    boundary_normals: np.ndarray   # outward unit normals, (Nb, 3)
    boundary_centroids: np.ndarray  # (Nb, 3)


def meshio_to_skfem(mesh: meshio.Mesh) -> skfem.MeshTet:
    pts = np.asarray(mesh.points, dtype=float)
    tets = None
    for cb in mesh.cells:
        if cb.type == "tetra":
            tets = np.asarray(cb.data, dtype=np.int64)
            break
    if tets is None:
        raise ValueError("meshio mesh has no tetra cells")
    return MeshTet(pts.T, tets.T)


def boundary_facet_info(mesh: skfem.MeshTet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (facet_indices, outward_unit_normals, centroids) for boundary facets."""
    bidx = mesh.boundary_facets()
    f_nodes = mesh.facets[:, bidx]
    p = mesh.p
    a = p[:, f_nodes[0]].T
    b = p[:, f_nodes[1]].T
    c = p[:, f_nodes[2]].T

    raw_n = np.cross(b - a, c - a)
    norms = np.linalg.norm(raw_n, axis=1)
    n_unit = raw_n / norms[:, None]

    centroids = (a + b + c) / 3.0

    # Make outward: each boundary facet belongs to exactly one tetra.
    tet = mesh.f2t[0, bidx]
    tet_centroids = mesh.p[:, mesh.t[:, tet]].mean(axis=1).T
    flip = np.einsum("ij,ij->i", n_unit, centroids - tet_centroids) < 0.0
    n_unit[flip] *= -1.0
    return bidx, n_unit, centroids


def classify_back_facets(
    normals: np.ndarray, p_hat: np.ndarray, tol_deg: float = 30.0
) -> np.ndarray:
    """Boolean mask over boundary facets: True for back-facing facets (n̂·p̂ > cos tol)."""
    cos_tol = np.cos(np.deg2rad(tol_deg))
    return (normals @ p_hat) > cos_tol


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #

def solve_steady(mesh_io: meshio.Mesh, *, k: float, bc: BCAssembly) -> SolveResult:
    bc.validate()
    mesh = meshio_to_skfem(mesh_io)
    basis = Basis(mesh, ElementTetP1())

    @BilinearForm
    def a_form(u, v, _w):
        return k * dot(grad(u), grad(v))

    A = a_form.assemble(basis)
    rhs = np.zeros(basis.N)

    # --- Heated facets: ∫ q v dA ---
    if bc.heated_facets.size > 0:
        fb_h = FacetBasis(mesh, ElementTetP1(), facets=bc.heated_facets)
        q_qp = np.broadcast_to(bc.q_heated[:, None], fb_h.dx.shape).copy()

        @LinearForm
        def heat_load(v, w):
            return w["q"] * v

        rhs += heat_load.assemble(fb_h, q=q_qp)

    # --- Robin facets: A += ∫ h u v dA;  rhs += ∫ h T_inf v dA ---
    if bc.robin_facets.size > 0:
        fb_r = FacetBasis(mesh, ElementTetP1(), facets=bc.robin_facets)
        h_qp = np.broadcast_to(bc.h_robin[:, None], fb_r.dx.shape).copy()
        Tinf_qp = np.broadcast_to(bc.T_inf_robin[:, None], fb_r.dx.shape).copy()

        @BilinearForm
        def robin_lhs(u, v, w):
            return w["h"] * u * v

        @LinearForm
        def robin_rhs(v, w):
            return w["h"] * w["T_inf"] * v

        A = A + robin_lhs.assemble(fb_r, h=h_qp)
        rhs += robin_rhs.assemble(fb_r, h=h_qp, T_inf=Tinf_qp)

    # --- Dirichlet condensation ---
    if bc.dirichlet_facets.size > 0:
        D = basis.get_dofs(facets=bc.dirichlet_facets).all()
        T = np.full(basis.N, np.nan)
        T[D] = bc.T_dirichlet
        T = skfem.solve(
            *skfem.condense(A, rhs, x=T, D=D),
            solver=lambda M, b: spsolve(M, b),
        )
    else:
        T = spsolve(A, rhs)

    T = np.asarray(T)

    bidx, b_normals, b_centroids = boundary_facet_info(mesh)
    return SolveResult(
        T=T,
        mesh=mesh,
        basis=basis,
        bc=bc,
        boundary_indices=bidx,
        boundary_normals=b_normals,
        boundary_centroids=b_centroids,
    )

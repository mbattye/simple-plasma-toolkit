"""Steady-state heat conduction FEM solve on a tet mesh using scikit-fem.

PDE:        -∇·(k ∇T) = 0  in Ω
Neumann:    -k ∂T/∂n = q                  on Γ_heated   (incoming flux)
Dirichlet:  T = T_D                        on Γ_D
Robin:      -k ∂T/∂n = h (T - T_inf)       on Γ_R
Radiation:   k ∂T/∂n = q_in - εσ(T⁴-T_env⁴)  on Γ_rad   (nonlinear, Newton-solved)
Natural:    -k ∂T/∂n = 0                   on every other boundary facet (adiabatic)

A `BCAssembly` collects per-facet membership + per-facet data for each kind of
BC. The solver assembles a single linear system and condenses out Dirichlet
DOFs. If a radiation patch is present, we Newton-iterate on the linearised
system; otherwise a single linear solve.
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

    # Radiation patch (nonlinear). May overlap with heated_facets — the
    # radiation contribution is added on top of any other BC on the same
    # facet, which is the physical picture: a facet absorbs the beam AND
    # re-radiates simultaneously.
    radiation_facets: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    eps_radiation: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    T_env_radiation: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))

    def validate(self) -> None:
        if self.heated_facets.shape != self.q_heated.shape:
            raise ValueError("heated_facets and q_heated length mismatch")
        if self.robin_facets.shape != self.h_robin.shape:
            raise ValueError("robin_facets and h_robin length mismatch")
        if self.robin_facets.shape != self.T_inf_robin.shape:
            raise ValueError("robin_facets and T_inf_robin length mismatch")
        if self.radiation_facets.shape != self.eps_radiation.shape:
            raise ValueError("radiation_facets and eps_radiation length mismatch")
        if self.radiation_facets.shape != self.T_env_radiation.shape:
            raise ValueError("radiation_facets and T_env_radiation length mismatch")
        if (
            self.dirichlet_facets.size == 0
            and self.robin_facets.size == 0
            and self.radiation_facets.size == 0
        ):
            raise ValueError(
                "No Dirichlet, Robin or radiation facets — problem is singular. "
                "All boundary facets would be adiabatic."
            )


# --------------------------------------------------------------------------- #
# Mesh utilities
# --------------------------------------------------------------------------- #

SIGMA_SB = 5.670374419e-8   # Stefan–Boltzmann constant, W/m²/K⁴


@dataclass
class TimeStepResult:
    """One frame from a transient solve."""
    t: float
    T: np.ndarray
    bc: "BCAssembly"
    n_newton_iters: int
    newton_residual: float
    n_shadowed: int


@dataclass
class TransientResult:
    """Output of `solve_transient`: time grid and per-step T frames + BCs."""
    times: np.ndarray
    T_history: list           # list[np.ndarray], len = len(times)
    bc_history: list          # list[BCAssembly]
    mesh: "skfem.MeshTet"
    basis: "Basis"
    boundary_indices: np.ndarray
    boundary_normals: np.ndarray
    boundary_centroids: np.ndarray
    n_shadowed_history: list  # list[int]
    n_newton_history: list    # list[int]
    mass_matrix: object = None  # scipy sparse; needed for transient energy balance


@dataclass
class SolveResult:
    T: np.ndarray
    mesh: skfem.MeshTet
    basis: Basis
    bc: BCAssembly
    boundary_indices: np.ndarray   # all boundary facet indices
    boundary_normals: np.ndarray   # outward unit normals, (Nb, 3)
    boundary_centroids: np.ndarray  # (Nb, 3)
    n_newton_iters: int = 0
    newton_residual: float = 0.0   # final max-rel update; 0 for linear case


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

def _radiation_initial_guess(bc: BCAssembly, basis: Basis) -> np.ndarray:
    """Pure-radiation equilibrium estimate: q_avg = εσ T^4 ⇒ T = (q_avg/(εσ))^¼.

    Used as the starting iterate for Newton. Robust because the linearised
    Robin-like coefficient 4εσT_k³ is well-conditioned away from T_k = 0.
    """
    if bc.heated_facets.size > 0 and bc.radiation_facets.size > 0:
        q_avg = float(np.mean(bc.q_heated))
        eps_avg = float(np.mean(bc.eps_radiation))
        T_env_avg = float(np.mean(bc.T_env_radiation))
        # T⁴ = q_avg/(εσ) + T_env⁴
        T0 = (q_avg / (eps_avg * SIGMA_SB) + T_env_avg ** 4) ** 0.25
        return np.full(basis.N, T0)
    # No radiation patch: 0 is fine, the system is linear.
    return np.zeros(basis.N)


def solve_steady(
    mesh_io: meshio.Mesh,
    *,
    k: float,
    bc: BCAssembly,
    newton_tol: float = 1e-4,
    newton_max_iter: int = 50,
) -> SolveResult:
    bc.validate()
    mesh = meshio_to_skfem(mesh_io)
    basis = Basis(mesh, ElementTetP1())

    # --- Stiffness (k-Laplacian) — same every Newton iter ---
    @BilinearForm
    def a_form(u, v, _w):
        return k * dot(grad(u), grad(v))

    A_cond = a_form.assemble(basis)
    rhs_const = np.zeros(basis.N)

    # --- Heated facets: ∫ q v dA (constant in T) ---
    if bc.heated_facets.size > 0:
        fb_h = FacetBasis(mesh, ElementTetP1(), facets=bc.heated_facets)
        q_qp = np.broadcast_to(bc.q_heated[:, None], fb_h.dx.shape).copy()

        @LinearForm
        def heat_load(v, w):
            return w["q"] * v

        rhs_const = rhs_const + heat_load.assemble(fb_h, q=q_qp)

    # --- Robin facets (constant in T): A += ∫ h u v dA, rhs += ∫ h T_inf v dA ---
    A_robin = None
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

        A_robin = robin_lhs.assemble(fb_r, h=h_qp)
        rhs_const = rhs_const + robin_rhs.assemble(fb_r, h=h_qp, T_inf=Tinf_qp)

    # --- Radiation patch (nonlinear): assembled inside the Newton loop ---
    fb_rad = None
    eps_qp = T_env_qp = None
    if bc.radiation_facets.size > 0:
        fb_rad = FacetBasis(mesh, ElementTetP1(), facets=bc.radiation_facets)
        eps_qp = np.broadcast_to(bc.eps_radiation[:, None], fb_rad.dx.shape).copy()
        T_env_qp = np.broadcast_to(bc.T_env_radiation[:, None], fb_rad.dx.shape).copy()

    @BilinearForm
    def rad_lhs(u, v, w):
        # 4 ε σ T_k³ · u v
        return 4.0 * w["eps"] * SIGMA_SB * w["Tk"] ** 3 * u * v

    @LinearForm
    def rad_rhs(v, w):
        # (3 ε σ T_k⁴ + ε σ T_env⁴) · v
        return (3.0 * w["eps"] * SIGMA_SB * w["Tk"] ** 4
                + w["eps"] * SIGMA_SB * w["T_env"] ** 4) * v

    # --- Dirichlet prep ---
    if bc.dirichlet_facets.size > 0:
        D = basis.get_dofs(facets=bc.dirichlet_facets).all()
    else:
        D = None

    def _solve_linear(A_total: object, rhs_total: np.ndarray) -> np.ndarray:
        if D is not None:
            x = np.full(basis.N, np.nan)
            x[D] = bc.T_dirichlet
            return np.asarray(
                skfem.solve(
                    *skfem.condense(A_total, rhs_total, x=x, D=D),
                    solver=lambda M, b: spsolve(M, b),
                )
            )
        return np.asarray(spsolve(A_total, rhs_total))

    # --- Newton iteration (single linear solve if no radiation patch) ---
    T = _radiation_initial_guess(bc, basis)
    n_iters = 0
    final_resid = 0.0

    if fb_rad is None:
        # Linear case — assemble once and go.
        A_total = A_cond if A_robin is None else (A_cond + A_robin)
        T = _solve_linear(A_total, rhs_const)
        n_iters = 1
    else:
        for it in range(1, newton_max_iter + 1):
            # Interpolate current T at the radiation-facet quadrature points.
            # FacetBasis.interpolate returns a DiscreteField — `.value` for P1.
            Tk_qp_arr = np.asarray(fb_rad.interpolate(T))
            # Numerical safety: keep T_k positive for the linearisation.
            Tk_qp_arr = np.maximum(Tk_qp_arr, 1.0)

            A_rad = rad_lhs.assemble(fb_rad, Tk=Tk_qp_arr, eps=eps_qp)
            rhs_rad = rad_rhs.assemble(fb_rad, Tk=Tk_qp_arr, eps=eps_qp, T_env=T_env_qp)

            A_total = A_cond + A_rad + (A_robin if A_robin is not None else 0.0)
            rhs_total = rhs_const + rhs_rad

            T_new = _solve_linear(A_total, rhs_total)

            denom = max(float(np.max(np.abs(T))), 1.0)
            resid = float(np.max(np.abs(T_new - T))) / denom
            T = T_new
            n_iters = it
            final_resid = resid
            if resid < newton_tol:
                break
        else:
            raise RuntimeError(
                f"Radiation Newton did not converge in {newton_max_iter} iters; "
                f"final relative update = {final_resid:.3e}"
            )

    bidx, b_normals, b_centroids = boundary_facet_info(mesh)
    return SolveResult(
        T=T,
        mesh=mesh,
        basis=basis,
        bc=bc,
        boundary_indices=bidx,
        boundary_normals=b_normals,
        boundary_centroids=b_centroids,
        n_newton_iters=n_iters,
        newton_residual=final_resid,
    )


# --------------------------------------------------------------------------- #
# Transient solver: backward Euler + inner Newton for radiation
# --------------------------------------------------------------------------- #

def solve_transient(
    mesh_io,
    *,
    k: float,
    rho: float,
    cp: float,
    bc_step_fn,         # callable: (step_idx, t) -> (BCAssembly, n_shadowed)
    times: np.ndarray,
    T_initial: float = 300.0,
    newton_tol: float = 1e-4,
    newton_max_iter: int = 50,
    progress=None,      # optional callable(step_idx, t, T_max, n_iters)
) -> TransientResult:
    """Backward-Euler march of  ρc_p ∂T/∂t = ∇·(k∇T) on the volume.

    `times` must be a strictly-increasing 1-D array of times (s). T at
    times[0] is set to T_initial; the loop then solves for times[1:].

    At each step we assemble the dt-dependent matrix
        S = M/Δt + K + A_robin
    and march
        S T^{n+1} = (M/Δt) T^n + b_heated + b_robin + radiation_increment(T^{n+1})
    where the radiation increment is itself Newton-iterated.
    """
    mesh = meshio_to_skfem(mesh_io)
    basis = Basis(mesh, ElementTetP1())

    # --- Time-invariant assemblies ---
    @BilinearForm
    def stiff_form(u, v, _w):
        return k * dot(grad(u), grad(v))

    @BilinearForm
    def mass_form(u, v, _w):
        return rho * cp * u * v

    K = stiff_form.assemble(basis)
    M = mass_form.assemble(basis)

    # --- Forms used per-step (composed from current BC) ---
    @BilinearForm
    def robin_lhs(u, v, w):
        return w["h"] * u * v

    @LinearForm
    def robin_rhs(v, w):
        return w["h"] * w["T_inf"] * v

    @LinearForm
    def heat_load(v, w):
        return w["q"] * v

    @BilinearForm
    def rad_lhs(u, v, w):
        return 4.0 * w["eps"] * SIGMA_SB * w["Tk"] ** 3 * u * v

    @LinearForm
    def rad_rhs(v, w):
        return (3.0 * w["eps"] * SIGMA_SB * w["Tk"] ** 4
                + w["eps"] * SIGMA_SB * w["T_env"] ** 4) * v

    # --- Initial condition ---
    T = np.full(basis.N, T_initial)
    T_history = [T.copy()]
    bc_history: list = []
    n_shadowed_history: list = []
    n_newton_history: list = []

    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("`times` must be a 1-D array of length >= 2")

    for step in range(1, times.size):
        t_new = float(times[step])
        dt = float(times[step] - times[step - 1])
        if dt <= 0:
            raise ValueError(f"non-positive dt at step {step}: {dt}")

        bc, n_shadowed = bc_step_fn(step, t_new)
        bc.validate()

        # Per-step constant load (heated facets + Robin RHS).
        rhs_const = (M @ T) / dt

        if bc.heated_facets.size > 0:
            fb_h = FacetBasis(mesh, ElementTetP1(), facets=bc.heated_facets)
            q_qp = np.broadcast_to(bc.q_heated[:, None], fb_h.dx.shape).copy()
            rhs_const = rhs_const + heat_load.assemble(fb_h, q=q_qp)

        A_robin = None
        if bc.robin_facets.size > 0:
            fb_r = FacetBasis(mesh, ElementTetP1(), facets=bc.robin_facets)
            h_qp = np.broadcast_to(bc.h_robin[:, None], fb_r.dx.shape).copy()
            Tinf_qp = np.broadcast_to(bc.T_inf_robin[:, None], fb_r.dx.shape).copy()
            A_robin = robin_lhs.assemble(fb_r, h=h_qp)
            rhs_const = rhs_const + robin_rhs.assemble(fb_r, h=h_qp, T_inf=Tinf_qp)

        fb_rad = None
        eps_qp = T_env_qp = None
        if bc.radiation_facets.size > 0:
            fb_rad = FacetBasis(mesh, ElementTetP1(), facets=bc.radiation_facets)
            eps_qp = np.broadcast_to(bc.eps_radiation[:, None], fb_rad.dx.shape).copy()
            T_env_qp = np.broadcast_to(bc.T_env_radiation[:, None], fb_rad.dx.shape).copy()

        # Dirichlet setup (fresh each step in case heated mask changed).
        if bc.dirichlet_facets.size > 0:
            D = basis.get_dofs(facets=bc.dirichlet_facets).all()
        else:
            D = None

        def _solve_with(A_tot, rhs_tot):
            if D is not None:
                x = np.full(basis.N, np.nan)
                x[D] = bc.T_dirichlet
                return np.asarray(
                    skfem.solve(
                        *skfem.condense(A_tot, rhs_tot, x=x, D=D),
                        solver=lambda Mx, b: spsolve(Mx, b),
                    )
                )
            return np.asarray(spsolve(A_tot, rhs_tot))

        # --- Newton inner loop (or 1 linear solve if no radiation) ---
        T_iter = T.copy()
        n_iters = 0
        resid = 0.0
        S_no_rad = M / dt + K + (A_robin if A_robin is not None else 0.0)

        if fb_rad is None:
            T_iter = _solve_with(S_no_rad, rhs_const)
            n_iters = 1
        else:
            for it in range(1, newton_max_iter + 1):
                Tk_qp = np.asarray(fb_rad.interpolate(T_iter))
                Tk_qp = np.maximum(Tk_qp, 1.0)
                A_rad = rad_lhs.assemble(fb_rad, Tk=Tk_qp, eps=eps_qp)
                rhs_rad = rad_rhs.assemble(fb_rad, Tk=Tk_qp, eps=eps_qp, T_env=T_env_qp)
                T_new = _solve_with(S_no_rad + A_rad, rhs_const + rhs_rad)
                denom = max(float(np.max(np.abs(T_iter))), 1.0)
                resid = float(np.max(np.abs(T_new - T_iter))) / denom
                T_iter = T_new
                n_iters = it
                if resid < newton_tol:
                    break
            else:
                raise RuntimeError(
                    f"Newton failed at step {step} (t={t_new:g} s): "
                    f"final relative update = {resid:.3e}"
                )

        T = T_iter
        T_history.append(T.copy())
        bc_history.append(bc)
        n_shadowed_history.append(int(n_shadowed))
        n_newton_history.append(int(n_iters))

        if progress is not None:
            progress(step, t_new, float(T.max()), n_iters)

    bidx, b_normals, b_centroids = boundary_facet_info(mesh)
    return TransientResult(
        times=times,
        T_history=T_history,
        bc_history=bc_history,
        mesh=mesh,
        basis=basis,
        boundary_indices=bidx,
        boundary_normals=b_normals,
        boundary_centroids=b_centroids,
        n_shadowed_history=n_shadowed_history,
        n_newton_history=n_newton_history,
        mass_matrix=M,
    )

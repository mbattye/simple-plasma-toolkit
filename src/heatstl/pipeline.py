"""High-level orchestration: from `SurfaceMesh` + config to `SolveResult`.

For steady runs the path is one-shot: mesh once, classify boundary, build
BCAssembly, hand to the solver.

For transient runs we split the work:

    `MeshContext`        — one-shot work: mesh, normals, back classification,
                           radiation set, neighbour occluder. Time-invariant.
    `bc_for_step(...)`   — cheap per-step builder: recomputes heated mask,
                           shadow mask, and per-facet q given (q0, p̂) at
                           the current time. Reuses the static patches
                           (Robin/Dirichlet/radiation) from the context.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from .flux import compute_face_flux, shadow_mask
from .geometry import SurfaceMesh, hex_neighbor_offsets, replicate
from .mesh import mesh_volume
from .solver import BCAssembly, boundary_facet_info, classify_back_facets, meshio_to_skfem


@dataclass
class RunConfig:
    """Everything the pipeline needs above and beyond the SurfaceMesh."""

    # Flux (in transient runs these are the "nominal" values used for
    # auto-pitch and the radiation initial guess only — the actual per-step
    # q0 and p_hat come from the profiles).
    q0: float
    mode: str
    p_hat: np.ndarray

    # Material
    k: float

    # BC on non-heated boundary
    bc_unheated: str
    T_cool: float = 300.0
    h: float = 100.0
    T_inf: float = 300.0
    back_h: float = 100.0
    back_T_inf: float = 400.0
    back_tol_deg: float = 30.0
    back_axis: np.ndarray | None = None

    # Front-face radiation
    front_radiation: bool = False
    emissivity: float = 0.89
    T_env: float = 300.0
    newton_tol: float = 1e-4
    newton_max_iter: int = 50

    # Neighbours / shadowing
    neighbors: str = "none"
    tile_pitch: float | None = None
    tile_gap: float = 0.0

    # Meshing
    mesh_size_m: float | None = None


@dataclass
class MeshContext:
    """One-shot, time-invariant data shared across all timesteps."""

    surf: SurfaceMesh
    mesh_io: object              # meshio.Mesh
    bidx: np.ndarray             # boundary facet indices, (Nb,)
    normals: np.ndarray          # outward unit normals, (Nb, 3)
    centroids: np.ndarray        # facet centroids, (Nb, 3)
    tile_axis: np.ndarray        # back direction (geometric); shadowing plane normal
    back_mask: np.ndarray        # bool over boundary facets
    radiation_mask: np.ndarray   # bool over boundary facets (front-facing set)
    occluder: trimesh.Trimesh | None  # neighbour geometry for shadowing


@dataclass
class PipelineOutput:
    bc: BCAssembly
    mesh_io: object
    n_shadowed: int
    n_central_heated: int


# --------------------------------------------------------------------------- #
# One-shot mesh context
# --------------------------------------------------------------------------- #

def build_mesh_context(surf: SurfaceMesh, cfg: RunConfig) -> MeshContext:
    mesh_io = mesh_volume(surf, mesh_size=cfg.mesh_size_m)
    skf_mesh = meshio_to_skfem(mesh_io)
    bidx, normals, centroids = boundary_facet_info(skf_mesh)

    tile_axis = cfg.back_axis if cfg.back_axis is not None else cfg.p_hat
    back_mask = classify_back_facets(normals, tile_axis, tol_deg=cfg.back_tol_deg)

    if cfg.front_radiation:
        front_axis = -tile_axis
        radiation_mask = (normals @ front_axis) > 0.05
    else:
        radiation_mask = np.zeros(bidx.size, dtype=bool)

    occluder: trimesh.Trimesh | None = None
    if cfg.neighbors == "hex6":
        offsets = hex_neighbor_offsets(
            surf, tile_axis, pitch=cfg.tile_pitch, gap=cfg.tile_gap
        )
        occluder = replicate(surf, offsets)
    elif cfg.neighbors != "none":
        raise ValueError(f"unknown neighbors mode {cfg.neighbors!r}")

    return MeshContext(
        surf=surf,
        mesh_io=mesh_io,
        bidx=bidx,
        normals=normals,
        centroids=centroids,
        tile_axis=tile_axis,
        back_mask=back_mask,
        radiation_mask=radiation_mask,
        occluder=occluder,
    )


# --------------------------------------------------------------------------- #
# Per-step BC builder
# --------------------------------------------------------------------------- #

def bc_for_step(
    ctx: MeshContext,
    cfg: RunConfig,
    *,
    q0: float,
    p_hat: np.ndarray,
) -> tuple[BCAssembly, int]:
    """Build the time-step BCAssembly. Returns (bc, n_shadowed)."""
    q = compute_face_flux(ctx.normals, q0=q0, mode=cfg.mode, p_hat=p_hat)

    n_shadowed = 0
    if ctx.occluder is not None and len(ctx.occluder.faces) > 0:
        sh = shadow_mask(ctx.centroids, p_hat, ctx.occluder)
        n_shadowed = int(np.sum(sh & (q > 0)))
        q = np.where(sh, 0.0, q)

    heated = q > 0.0
    non_heated = ~heated

    bc = BCAssembly()
    bc.heated_facets = ctx.bidx[heated]
    bc.q_heated = q[heated]

    if cfg.front_radiation:
        rad_idx = ctx.bidx[ctx.radiation_mask]
        bc.radiation_facets = rad_idx
        bc.eps_radiation = np.full(rad_idx.size, cfg.emissivity)
        bc.T_env_radiation = np.full(rad_idx.size, cfg.T_env)

    mode = cfg.bc_unheated
    back = ctx.back_mask
    if mode == "dirichlet":
        bc.dirichlet_facets = ctx.bidx[non_heated]
        bc.T_dirichlet = cfg.T_cool
    elif mode == "robin":
        rf = ctx.bidx[non_heated]
        bc.robin_facets = rf
        bc.h_robin = np.full(rf.size, cfg.h)
        bc.T_inf_robin = np.full(rf.size, cfg.T_inf)
    elif mode == "adiabatic-back-dirichlet":
        sel = non_heated & back
        bc.dirichlet_facets = ctx.bidx[sel]
        bc.T_dirichlet = cfg.T_cool
    elif mode == "adiabatic-back-robin":
        sel = non_heated & back
        rf = ctx.bidx[sel]
        bc.robin_facets = rf
        bc.h_robin = np.full(rf.size, cfg.back_h)
        bc.T_inf_robin = np.full(rf.size, cfg.back_T_inf)
    else:
        raise ValueError(f"unknown bc_unheated mode {mode!r}")

    return bc, n_shadowed


# --------------------------------------------------------------------------- #
# Steady wrapper (back-compat with v1-v3 callers)
# --------------------------------------------------------------------------- #

def build_bc(surf: SurfaceMesh, cfg: RunConfig) -> PipelineOutput:
    """One-shot mesh + BC build for the steady case."""
    ctx = build_mesh_context(surf, cfg)
    bc, n_shadowed = bc_for_step(ctx, cfg, q0=cfg.q0, p_hat=cfg.p_hat)
    return PipelineOutput(
        bc=bc,
        mesh_io=ctx.mesh_io,
        n_shadowed=n_shadowed,
        n_central_heated=int(bc.heated_facets.size),
    )

"""High-level orchestration: from `SurfaceMesh` + config to `SolveResult`.

Pulls together flux computation, optional shadowing against hex neighbours,
boundary classification, and BC assembly. Kept thin and explicit so the CLI
stays a translation layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .flux import compute_face_flux, shadow_mask
from .geometry import SurfaceMesh, hex_neighbor_offsets, replicate
from .mesh import mesh_volume
from .solver import BCAssembly, boundary_facet_info, classify_back_facets, meshio_to_skfem


@dataclass
class RunConfig:
    """Everything the pipeline needs above and beyond the SurfaceMesh."""

    # Flux
    q0: float
    mode: str               # 'oblique' | 'normal'
    p_hat: np.ndarray

    # Material
    k: float

    # BC on non-heated boundary
    bc_unheated: str        # 'dirichlet' | 'robin' | 'adiabatic-back-dirichlet' | 'adiabatic-back-robin'
    T_cool: float = 300.0
    h: float = 100.0
    T_inf: float = 300.0
    back_h: float = 100.0
    back_T_inf: float = 400.0
    back_tol_deg: float = 30.0
    # Geometric axis pointing from the heated front toward the back. Decoupled
    # from p_hat so oblique beams still find the same back face. Defaults to
    # p_hat when None.
    back_axis: np.ndarray | None = None

    # Front-face radiation (nonlinear). When True, every heated facet also
    # radiates: net flux = q_in − εσ(T⁴ − T_env⁴). Solved via Newton.
    front_radiation: bool = False
    emissivity: float = 0.89
    T_env: float = 300.0

    # Newton tolerances
    newton_tol: float = 1e-4
    newton_max_iter: int = 50

    # Optional neighbours / shadowing
    neighbors: str = "none"      # 'none' | 'hex6'
    tile_pitch: float | None = None
    tile_gap: float = 0.0

    # Meshing
    mesh_size_m: float | None = None


@dataclass
class PipelineOutput:
    bc: BCAssembly
    mesh_io: object          # meshio.Mesh
    n_shadowed: int          # facets that would have been heated but are shadowed
    n_central_heated: int


def build_bc(surf: SurfaceMesh, cfg: RunConfig) -> PipelineOutput:
    """Volume-mesh the central tile and build the BCAssembly."""
    mesh_io = mesh_volume(surf, mesh_size=cfg.mesh_size_m)
    skf_mesh = meshio_to_skfem(mesh_io)
    bidx, normals, centroids = boundary_facet_info(skf_mesh)

    # Incident heat flux (before shadowing).
    q = compute_face_flux(normals, q0=cfg.q0, mode=cfg.mode, p_hat=cfg.p_hat)

    # Tile axis: perpendicular to the tile's flat back face. Used to lay out
    # hex neighbours and classify the back face. Falls back to p_hat if not
    # set so behaviour matches v1 single-tile, normal-incidence cases.
    tile_axis = cfg.back_axis if cfg.back_axis is not None else cfg.p_hat

    # Optional shadowing.
    n_shadowed = 0
    if cfg.neighbors == "hex6":
        offsets = hex_neighbor_offsets(
            surf, tile_axis, pitch=cfg.tile_pitch, gap=cfg.tile_gap
        )
        occluder = replicate(surf, offsets)
        sh = shadow_mask(centroids, cfg.p_hat, occluder)
        # Only count shadowing of facets that were going to be heated.
        n_shadowed = int(np.sum(sh & (q > 0)))
        q = np.where(sh, 0.0, q)
    elif cfg.neighbors != "none":
        raise ValueError(f"unknown neighbors mode {cfg.neighbors!r}")

    heated = q > 0.0
    # Back-face classification — uses tile_axis (geometric), not p_hat (beam).
    back = classify_back_facets(normals, tile_axis, tol_deg=cfg.back_tol_deg)

    bc = BCAssembly()
    bc.heated_facets = bidx[heated]
    bc.q_heated = q[heated]

    # Radiation patch: every facet that points "forward" (away from the
    # tile back) radiates to the environment, regardless of whether the
    # current beam is actually hitting it.
    #
    # We DO NOT tie this to the heated mask, because (a) physically the
    # cool side of a hot tile still radiates while the beam is off-axis,
    # and (b) numerical noise in facet normals produces ~1e-11 illuminated
    # facets on geometric sides that would otherwise be forced toward
    # T_env by an εσT⁴ outflux they can't supply.
    if cfg.front_radiation:
        front_axis = -tile_axis
        cos_front = normals @ front_axis
        rad_mask = cos_front > 0.05  # ≤ ~87° from front
        bc.radiation_facets = bidx[rad_mask]
        bc.eps_radiation = np.full(bc.radiation_facets.size, cfg.emissivity)
        bc.T_env_radiation = np.full(bc.radiation_facets.size, cfg.T_env)

    non_heated = ~heated
    mode = cfg.bc_unheated
    if mode == "dirichlet":
        bc.dirichlet_facets = bidx[non_heated]
        bc.T_dirichlet = cfg.T_cool
    elif mode == "robin":
        rf = bidx[non_heated]
        bc.robin_facets = rf
        bc.h_robin = np.full(rf.size, cfg.h)
        bc.T_inf_robin = np.full(rf.size, cfg.T_inf)
    elif mode == "adiabatic-back-dirichlet":
        sel = non_heated & back
        bc.dirichlet_facets = bidx[sel]
        bc.T_dirichlet = cfg.T_cool
    elif mode == "adiabatic-back-robin":
        sel = non_heated & back
        rf = bidx[sel]
        bc.robin_facets = rf
        bc.h_robin = np.full(rf.size, cfg.back_h)
        bc.T_inf_robin = np.full(rf.size, cfg.back_T_inf)
    else:
        raise ValueError(f"unknown bc_unheated mode {mode!r}")

    return PipelineOutput(
        bc=bc,
        mesh_io=mesh_io,
        n_shadowed=n_shadowed,
        n_central_heated=int(bc.heated_facets.size),
    )

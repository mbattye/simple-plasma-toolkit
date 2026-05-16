"""FastAPI service wrapper for heatstl.

Install with: pip install heatstl[service]
Run with:     uvicorn heatstl.service.app:app --host 0.0.0.0 --port 8080

Endpoints:
    GET  /health                — liveness probe
    GET  /presets               — list built-in preset names + descriptions
    POST /solve/steady          — steady heat-conduction solve
    POST /solve/transient       — transient time-stepping solve

Optional auth: set ``ENGINE_SECRET`` and clients must send the same
value in an ``X-Engine-Token`` header. Mirrors diagnostic-designer.

STL input is always a URL (``http(s)://`` or ``gs://``); the engine
fetches it itself. Heavy outputs are published via the artifact store
configured at startup (see :mod:`heatstl.service.artifact_store`); the
response carries URIs the Analog grid can resolve.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlparse

try:
    import httpx
    from fastapi import FastAPI, Header, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:
    raise ImportError(
        "FastAPI / httpx required for the service. Install with: "
        "pip install heatstl[service]"
    ) from exc

import numpy as np

from heatstl import __version__
from heatstl.diagnostics import compute as compute_diagnostics
from heatstl.diagnostics import compute_frame
from heatstl.geometry import UNIT_TO_M, direction_from_angles, load_stl, parse_direction
from heatstl.io import (
    write_report,
    write_transient_report,
    write_vtu,
    write_vtu_frames,
    write_xdmf_arrow_timeseries,
    write_xdmf_timeseries,
)
from heatstl.pipeline import RunConfig, bc_for_step, build_bc, build_mesh_context
from heatstl.presets import PRESETS
from heatstl.service.artifact_store import (
    ArtifactStore,
    ArtifactStoreConfigurationError,
    get_default_store,
)
from heatstl.solver import solve_steady, solve_transient
from heatstl.transient import (
    PHatProfileSpec,
    QProfileSpec,
    TransientConfig,
    make_p_hat_profile,
    make_q_profile,
)


logger = logging.getLogger("heatstl.service")
ENGINE_SECRET = os.environ.get("ENGINE_SECRET", "")


# Startup dependency check — surface missing native libs in Cloud Run logs.
for mod_name in ("gmsh", "skfem", "trimesh", "meshio", "h5py"):
    try:
        __import__(mod_name)
        logger.info("%s loaded", mod_name)
    except ImportError as e:
        logger.warning("%s NOT installed: %s", mod_name, e)


app = FastAPI(
    title="heatstl",
    version=__version__,
    description=(
        "Steady and transient heat conduction on an STL with prescribed "
        "heat flux. Output: VTU / XDMF + h5 + JSON diagnostics, "
        "published via the configured artifact store."
    ),
)


# --------------------------------------------------------------------------- #
# Auth + helpers
# --------------------------------------------------------------------------- #

def _check_token(x_engine_token: Optional[str] = Header(None)) -> None:
    if ENGINE_SECRET and x_engine_token != ENGINE_SECRET:
        raise HTTPException(status_code=401, detail="Invalid engine token")


def _fetch_stl(url: str, target: Path, timeout_s: float = 60.0) -> None:
    """Download an STL into `target`. Supports http(s):// and gs:// URLs."""
    scheme = urlparse(url).scheme.lower()
    if scheme in ("http", "https"):
        with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
            r = client.get(url)
            r.raise_for_status()
            target.write_bytes(r.content)
        return
    if scheme == "gs":
        try:
            from google.cloud import storage  # type: ignore
        except ImportError as e:
            raise HTTPException(
                status_code=500,
                detail="google-cloud-storage missing; install heatstl[service]",
            ) from e
        client = storage.Client()
        bucket_name, _, blob_name = url[len("gs://"):].partition("/")
        if not bucket_name or not blob_name:
            raise HTTPException(status_code=400, detail=f"invalid gs:// URL: {url!r}")
        blob = client.bucket(bucket_name).blob(blob_name)
        blob.download_to_filename(str(target))
        return
    raise HTTPException(status_code=400, detail=f"unsupported URL scheme: {scheme!r}")


# Lazy store: built on first use so import-time failure with bad config
# doesn't take down the whole service.
_store: Optional[ArtifactStore] = None


def _store_now() -> ArtifactStore:
    global _store
    if _store is None:
        try:
            _store = get_default_store()
        except ArtifactStoreConfigurationError as e:
            raise HTTPException(status_code=500, detail=str(e))
    return _store


# --------------------------------------------------------------------------- #
# Pydantic request / response models
# --------------------------------------------------------------------------- #

class CommonOpts(BaseModel):
    """Flags shared between steady and transient runs."""

    stl_url: str = Field(..., description="HTTP(S) or gs:// URL to a watertight STL.")
    instance_id: str = Field(..., description="Caller-supplied id; used as the artefact key prefix.")
    q0: float = Field(..., description="Peak heat flux, W/m². For transient: profile amplitude.")
    unit: Literal["mm", "m"] = "mm"
    preset: Optional[str] = None

    # Beam direction
    direction: Optional[str] = Field(None, description="Beam direction 'x,y,z'.")
    angle_deg: Optional[float] = Field(None, description="Polar angle from -z, deg.")
    azimuth_deg: float = 0.0
    mode: Literal["oblique", "normal"] = "oblique"

    # Material
    k: Optional[float] = None

    # BC on non-heated facets
    bc_unheated: Optional[Literal[
        "dirichlet", "robin", "adiabatic-back-dirichlet", "adiabatic-back-robin"
    ]] = None
    T_cool: float = 300.0
    h: float = 100.0
    T_inf: float = 300.0
    back_h: Optional[float] = None
    back_T_inf: Optional[float] = None
    back_tol_deg: float = 30.0
    back_axis: Optional[str] = None

    # Radiation
    front_radiation: Optional[bool] = None
    emissivity: Optional[float] = None
    T_env: Optional[float] = None
    newton_tol: float = 1e-4
    newton_max_iter: int = 50

    # Neighbours / shadowing
    neighbors: Literal["none", "hex6"] = "none"
    tile_pitch: Optional[float] = None
    tile_gap: float = 0.0

    # Mesh
    mesh_size: Optional[float] = None


class SteadyRequest(CommonOpts):
    """Steady solve."""


class TransientRequest(CommonOpts):
    """Transient solve."""

    duration: float = Field(..., gt=0)
    n_steps: int = Field(..., gt=0)
    rho: Optional[float] = None
    cp: Optional[float] = None
    T_initial: float = 300.0

    # q profile
    q_profile: Literal["constant", "ramp", "gaussian", "piecewise"] = "constant"
    q_csv_url: Optional[str] = None
    q_ramp_t: float = 1.0
    q_t0: float = 0.0
    q_fwhm: float = 1.0

    # p_hat profile
    angle_profile: Literal["constant", "sweep", "piecewise"] = "constant"
    angle_csv_url: Optional[str] = None
    angle_start: Optional[float] = None
    angle_end: Optional[float] = None
    angle_t0: Optional[float] = None
    angle_t1: Optional[float] = None

    write_vtu_frames: bool = False


class SteadyResponse(BaseModel):
    instance_id: str
    version: str
    result_uri: str
    report_uri: str
    peak_T: float
    min_T: float
    Q_in: float
    Q_radiated: float
    Q_conducted_out: float
    residual_rel: float
    n_newton_iters: int
    wall_seconds: float


class TransientResponse(BaseModel):
    instance_id: str
    version: str
    result_uri: str          # XDMF
    h5_uri: str              # companion h5 file
    arrow_uri: str           # arrow XDMF
    arrow_h5_uri: str
    report_uri: str          # JSON with per-step diagnostics
    vtu_frame_uris: list[str] = Field(default_factory=list)
    peak_T: float
    t_peak: float
    n_steps: int
    wall_seconds: float


class PresetInfo(BaseModel):
    name: str
    keys: list[str]


# --------------------------------------------------------------------------- #
# Common: build RunConfig from request + preset
# --------------------------------------------------------------------------- #

def _resolve_config(req: CommonOpts) -> tuple[RunConfig, np.ndarray, dict]:
    pre = PRESETS.get(req.preset, {}) if req.preset else {}

    def _pick(req_val, key, default):
        return req_val if req_val is not None else pre.get(key, default)

    k_val = _pick(req.k, "k", 150.0)
    bc_unheated = _pick(req.bc_unheated, "bc_unheated", "dirichlet")
    back_h = _pick(req.back_h, "back_h", 100.0)
    back_T_inf = _pick(req.back_T_inf, "back_T_inf", 400.0)
    back_tol_deg = pre.get("back_tol_deg", req.back_tol_deg) if req.back_tol_deg == 30.0 else req.back_tol_deg
    back_axis_str = req.back_axis or pre.get("back_axis")
    back_axis_vec = parse_direction(back_axis_str) if back_axis_str else None
    front_radiation = bool(_pick(req.front_radiation, "front_radiation", False))
    emissivity = _pick(req.emissivity, "emissivity", 0.89)
    T_env = _pick(req.T_env, "T_env", 300.0)

    # Beam direction
    if req.direction is not None and req.angle_deg is not None:
        raise HTTPException(
            status_code=400,
            detail="provide either 'direction' OR 'angle_deg' / 'azimuth_deg', not both",
        )
    if req.angle_deg is not None:
        p_hat = direction_from_angles(req.angle_deg, req.azimuth_deg)
    elif req.direction is not None:
        p_hat = parse_direction(req.direction)
    else:
        p_hat = parse_direction("0,0,-1")

    mesh_size_m = (req.mesh_size * UNIT_TO_M[req.unit]) if req.mesh_size is not None else None
    tile_pitch_m = (req.tile_pitch * UNIT_TO_M[req.unit]) if req.tile_pitch is not None else None
    tile_gap_m = req.tile_gap * UNIT_TO_M[req.unit]

    cfg = RunConfig(
        q0=req.q0, mode=req.mode, p_hat=p_hat, k=k_val,
        bc_unheated=bc_unheated, T_cool=req.T_cool,
        h=req.h, T_inf=req.T_inf,
        back_h=back_h, back_T_inf=back_T_inf, back_tol_deg=back_tol_deg,
        back_axis=back_axis_vec,
        front_radiation=front_radiation, emissivity=emissivity, T_env=T_env,
        newton_tol=req.newton_tol, newton_max_iter=req.newton_max_iter,
        neighbors=req.neighbors, tile_pitch=tile_pitch_m, tile_gap=tile_gap_m,
        mesh_size_m=mesh_size_m,
    )
    meta_extras = {
        "preset": req.preset,
        "back_axis_str": back_axis_str,
        "front_radiation": front_radiation,
        "emissivity": emissivity,
        "T_env_K": T_env,
    }
    return cfg, p_hat, meta_extras


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/presets", response_model=list[PresetInfo])
def list_presets() -> list[PresetInfo]:
    return [PresetInfo(name=k, keys=list(v.keys())) for k, v in PRESETS.items()]


@app.post("/solve/steady", response_model=SteadyResponse)
def solve_steady_endpoint(
    req: SteadyRequest,
    x_engine_token: Optional[str] = Header(None),
) -> SteadyResponse:
    _check_token(x_engine_token)
    t0 = time.perf_counter()
    store = _store_now()

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        stl_path = td_path / "input.stl"
        _fetch_stl(req.stl_url, stl_path)

        surf = load_stl(stl_path, unit=req.unit)
        cfg, p_hat, meta_extras = _resolve_config(req)

        out = build_bc(surf, cfg)
        result = solve_steady(
            out.mesh_io, k=cfg.k, bc=out.bc,
            newton_tol=cfg.newton_tol, newton_max_iter=cfg.newton_max_iter,
        )
        diag = compute_diagnostics(result, k=cfg.k)

        vtu_path = td_path / "result.vtu"
        json_path = td_path / "report.json"
        write_vtu(vtu_path, result)
        meta = {
            "version": __version__,
            "instance_id": req.instance_id,
            "stl_url": req.stl_url,
            "q0_W_m2": req.q0,
            "p_hat": p_hat.tolist(),
            "mode": req.mode,
            "k_W_m_K": cfg.k,
            "bc_unheated": cfg.bc_unheated,
            "neighbors": cfg.neighbors,
            "n_shadowed_facets": out.n_shadowed,
            **meta_extras,
        }
        write_report(json_path, diag, meta)

        result_uri = store.put(
            f"{req.instance_id}/result.vtu",
            vtu_path.read_bytes(),
            content_type="application/octet-stream",
        )
        report_uri = store.put(
            f"{req.instance_id}/report.json",
            json_path.read_bytes(),
            content_type="application/json",
        )

    return SteadyResponse(
        instance_id=req.instance_id,
        version=__version__,
        result_uri=result_uri,
        report_uri=report_uri,
        peak_T=diag.peak_T,
        min_T=diag.min_T,
        Q_in=diag.Q_in,
        Q_radiated=diag.Q_radiated,
        Q_conducted_out=diag.Q_conducted_out,
        residual_rel=diag.residual_rel,
        n_newton_iters=diag.n_newton_iters,
        wall_seconds=time.perf_counter() - t0,
    )


@app.post("/solve/transient", response_model=TransientResponse)
def solve_transient_endpoint(
    req: TransientRequest,
    x_engine_token: Optional[str] = Header(None),
) -> TransientResponse:
    _check_token(x_engine_token)
    t0 = time.perf_counter()
    store = _store_now()

    cfg, p_hat, meta_extras = _resolve_config(req)
    pre = PRESETS.get(req.preset, {}) if req.preset else {}

    rho = req.rho if req.rho is not None else pre.get("rho")
    cp = req.cp if req.cp is not None else pre.get("cp")
    if not rho or not cp:
        raise HTTPException(
            status_code=400,
            detail="transient solves need 'rho' and 'cp' (or a preset providing them)",
        )

    # Build profile specs (and download any CSVs first).
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)

        stl_path = td_path / "input.stl"
        _fetch_stl(req.stl_url, stl_path)

        q_csv_local: Optional[str] = None
        if req.q_csv_url:
            q_csv_local = str(td_path / "q.csv")
            _fetch_stl(req.q_csv_url, Path(q_csv_local))  # same downloader works for any file
        angle_csv_local: Optional[str] = None
        if req.angle_csv_url:
            angle_csv_local = str(td_path / "angle.csv")
            _fetch_stl(req.angle_csv_url, Path(angle_csv_local))

        q_spec = QProfileSpec(
            kind=req.q_profile, q0=req.q0,
            t_ramp=req.q_ramp_t, t0=req.q_t0, fwhm=req.q_fwhm,
            csv=q_csv_local,
        )
        angle_spec = PHatProfileSpec(
            kind=req.angle_profile, p_hat=p_hat,
            angle_start=req.angle_start if req.angle_start is not None else 0.0,
            angle_end=req.angle_end if req.angle_end is not None else 0.0,
            azimuth=req.azimuth_deg,
            t0=req.angle_t0 if req.angle_t0 is not None else 0.0,
            t1=req.angle_t1 if req.angle_t1 is not None else max(req.duration, 1.0),
            csv=angle_csv_local,
        )
        cfg_trans = TransientConfig(
            duration=req.duration, n_steps=req.n_steps,
            rho=float(rho), cp=float(cp), T_initial=req.T_initial,
            q_profile=q_spec, p_hat_profile=angle_spec,
        )

        surf = load_stl(stl_path, unit=req.unit)
        ctx = build_mesh_context(surf, cfg)
        q_of_t = make_q_profile(cfg_trans.q_profile)
        p_of_t = make_p_hat_profile(cfg_trans.p_hat_profile)

        def bc_step_fn(step_idx: int, t: float):
            return bc_for_step(ctx, cfg, q0=q_of_t(t), p_hat=p_of_t(t))

        times = cfg_trans.time_grid()
        result = solve_transient(
            ctx.mesh_io, k=cfg.k, rho=cfg_trans.rho, cp=cfg_trans.cp,
            bc_step_fn=bc_step_fn, times=times, T_initial=cfg_trans.T_initial,
            newton_tol=cfg.newton_tol, newton_max_iter=cfg.newton_max_iter,
        )

        # Per-step diagnostics including transient energy balance.
        per_step = []
        for i, T in enumerate(result.T_history):
            if i == 0:
                per_step.append({
                    "t_s": float(times[0]),
                    "peak_T": float(T.max()),
                    "min_T": float(T.min()),
                    "n_newton_iters": 0,
                })
                continue
            bc_i = result.bc_history[i - 1]
            d = compute_frame(result.mesh, T, bc_i, k=cfg.k,
                              n_newton_iters=result.n_newton_history[i - 1]).as_dict()
            d["t_s"] = float(times[i])
            d["q0_W_m2"] = float(q_of_t(float(times[i])))
            d["p_hat"] = p_of_t(float(times[i])).tolist()
            d["n_shadowed"] = result.n_shadowed_history[i - 1]
            dt = float(times[i] - times[i - 1])
            dT = T - result.T_history[i - 1]
            dU_dt = float((result.mass_matrix @ dT).sum() / dt)
            d["dU_dt_W"] = dU_dt
            d["residual_transient_rel"] = (
                (d["Q_in"] - d["Q_out_total"] - dU_dt) / d["Q_in"]
                if d["Q_in"] != 0.0 else float("nan")
            )
            per_step.append(d)

        # Write artefacts to the temp dir, then upload.
        out_xdmf = td_path / "result.xdmf"
        arrow_xdmf = td_path / "result_arrow.xdmf"
        report_json = td_path / "report.json"

        write_xdmf_timeseries(out_xdmf, result)

        up_dir = (-cfg.back_axis) if cfg.back_axis is not None else np.array([0.0, 0.0, 1.0])
        anchor = surf.vertices.mean(axis=0) + 0.5 * surf.bbox_diag * up_dir
        arrow_length = 0.3 * surf.bbox_diag
        p_hat_history = [p_of_t(float(t)) for t in times]
        q0_history = [float(q_of_t(float(t))) for t in times]
        write_xdmf_arrow_timeseries(
            arrow_xdmf, times, p_hat_history, q0_history,
            anchor=anchor, arrow_length=arrow_length,
        )

        meta = {
            "version": __version__,
            "instance_id": req.instance_id,
            "stl_url": req.stl_url,
            "transient": True,
            "duration_s": cfg_trans.duration, "n_steps": cfg_trans.n_steps,
            "rho_kg_m3": cfg_trans.rho, "cp_J_kg_K": cfg_trans.cp,
            "k_W_m_K": cfg.k, "T_initial_K": cfg_trans.T_initial,
            "q_profile": q_spec.__dict__,
            "p_hat_profile": {
                **angle_spec.__dict__,
                "p_hat": angle_spec.p_hat.tolist() if angle_spec.p_hat is not None else None,
            },
            "bc_unheated": cfg.bc_unheated,
            "back_h_W_m2_K": cfg.back_h, "back_T_inf_K": cfg.back_T_inf,
            "neighbors": cfg.neighbors,
            **meta_extras,
        }
        write_transient_report(report_json, meta, times, per_step)

        # Upload — XDMF references .h5 by basename, so we upload them
        # under the same prefix with the same stem.
        key_prefix = req.instance_id
        result_uri = store.put(f"{key_prefix}/result.xdmf", out_xdmf.read_bytes(), content_type="application/xml")
        h5_path = out_xdmf.with_suffix(".h5")
        h5_uri = store.put(f"{key_prefix}/result.h5", h5_path.read_bytes(), content_type="application/octet-stream")
        arrow_uri = store.put(f"{key_prefix}/result_arrow.xdmf", arrow_xdmf.read_bytes(), content_type="application/xml")
        arrow_h5_path = arrow_xdmf.with_suffix(".h5")
        arrow_h5_uri = store.put(f"{key_prefix}/result_arrow.h5", arrow_h5_path.read_bytes(), content_type="application/octet-stream")
        report_uri = store.put(f"{key_prefix}/report.json", report_json.read_bytes(), content_type="application/json")

        vtu_uris: list[str] = []
        if req.write_vtu_frames:
            frame_dir = td_path / "frames"
            frame_dir.mkdir(exist_ok=True)
            pattern = str(frame_dir / "frame_{:04d}.vtu")
            written = write_vtu_frames(pattern, result)
            for i, path in enumerate(written):
                blob = Path(path).read_bytes()
                vtu_uris.append(
                    store.put(
                        f"{key_prefix}/frames/frame_{i:04d}.vtu",
                        blob, content_type="application/octet-stream",
                    )
                )

    peak_T = max(d["peak_T"] for d in per_step)
    t_peak = per_step[int(np.argmax([d["peak_T"] for d in per_step]))]["t_s"]

    return TransientResponse(
        instance_id=req.instance_id,
        version=__version__,
        result_uri=result_uri,
        h5_uri=h5_uri,
        arrow_uri=arrow_uri,
        arrow_h5_uri=arrow_h5_uri,
        report_uri=report_uri,
        vtu_frame_uris=vtu_uris,
        peak_T=float(peak_T),
        t_peak=float(t_peak),
        n_steps=req.n_steps,
        wall_seconds=time.perf_counter() - t0,
    )

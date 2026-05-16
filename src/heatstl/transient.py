"""Time-varying flux and beam-direction profiles for transient runs.

A profile is a plain callable ``t -> value``. We expose simple factories
that read a profile spec (kind + a few scalars) and return such a callable.
Two families:

    q0(t)  : float
    p̂(t)   : np.ndarray shape (3,), unit vector

Supported kinds:
    constant        — single scalar / direction held for all t
    ramp            — q ramps from 0 to q0 over t_ramp (q profile only)
    gaussian        — q0 · exp(-(t-t0)² / 2σ²),  σ = fwhm / 2√(2 ln 2)  (q only)
    sweep           — angle linearly interpolates between (θ0, t0) and (θ1, t1)
                      held at endpoints outside. Azimuth fixed.            (p̂ only)
    piecewise       — linear interpolation over a CSV table (either)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .geometry import direction_from_angles


# --------------------------------------------------------------------------- #
# Spec containers (filled from CLI, parsed once)
# --------------------------------------------------------------------------- #

@dataclass
class QProfileSpec:
    kind: str
    q0: float = 0.0
    t_ramp: float = 1.0
    t0: float = 0.0
    fwhm: float = 1.0
    csv: str | None = None


@dataclass
class PHatProfileSpec:
    kind: str
    p_hat: np.ndarray | None = None   # for 'constant'
    angle_start: float = 0.0          # deg, polar
    angle_end: float = 0.0
    azimuth: float = 0.0
    t0: float = 0.0
    t1: float = 1.0
    csv: str | None = None


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #

def make_q_profile(spec: QProfileSpec) -> Callable[[float], float]:
    kind = spec.kind
    if kind == "constant":
        q0 = spec.q0
        return lambda t: float(q0)
    if kind == "ramp":
        q0, t_ramp = spec.q0, max(spec.t_ramp, 1e-12)
        return lambda t: float(q0 * min(1.0, max(0.0, t) / t_ramp))
    if kind == "gaussian":
        q0, t0 = spec.q0, spec.t0
        sigma = max(spec.fwhm, 1e-12) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        return lambda t: float(q0 * np.exp(-((t - t0) ** 2) / (2.0 * sigma ** 2)))
    if kind == "piecewise":
        if spec.csv is None:
            raise ValueError("piecewise q profile requires --q-csv")
        data = np.loadtxt(spec.csv, delimiter=",", comments="#")
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"{spec.csv}: expected 2-column (t, q) CSV")
        ts, qs = data[:, 0], data[:, 1]
        order = np.argsort(ts)
        ts, qs = ts[order], qs[order]
        return lambda t: float(np.interp(t, ts, qs, left=qs[0], right=qs[-1]))
    raise ValueError(f"unknown q profile kind {kind!r}")


def make_p_hat_profile(spec: PHatProfileSpec) -> Callable[[float], np.ndarray]:
    kind = spec.kind
    if kind == "constant":
        if spec.p_hat is None:
            raise ValueError("constant p_hat profile requires p_hat")
        p = np.asarray(spec.p_hat, dtype=float)
        return lambda t: p
    if kind == "sweep":
        a0, a1, az = spec.angle_start, spec.angle_end, spec.azimuth
        t0, t1 = spec.t0, spec.t1
        if t1 <= t0:
            raise ValueError("sweep requires t1 > t0")

        def f(t: float) -> np.ndarray:
            frac = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
            angle = a0 + frac * (a1 - a0)
            return direction_from_angles(angle, az)

        return f
    if kind == "piecewise":
        if spec.csv is None:
            raise ValueError("piecewise p_hat profile requires --angle-csv")
        data = np.loadtxt(spec.csv, delimiter=",", comments="#")
        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError(f"{spec.csv}: expected 3-column (t, theta_deg, phi_deg) CSV")
        ts, ths, phs = data[:, 0], data[:, 1], data[:, 2]
        order = np.argsort(ts)
        ts, ths, phs = ts[order], ths[order], phs[order]

        def f(t: float) -> np.ndarray:
            th = float(np.interp(t, ts, ths, left=ths[0], right=ths[-1]))
            ph = float(np.interp(t, ts, phs, left=phs[0], right=phs[-1]))
            return direction_from_angles(th, ph)

        return f
    raise ValueError(f"unknown p_hat profile kind {kind!r}")


# --------------------------------------------------------------------------- #
# Transient run config
# --------------------------------------------------------------------------- #

@dataclass
class TransientConfig:
    duration: float
    n_steps: int
    rho: float
    cp: float
    T_initial: float = 300.0
    q_profile: QProfileSpec | None = None
    p_hat_profile: PHatProfileSpec | None = None

    def time_grid(self) -> np.ndarray:
        return np.linspace(0.0, self.duration, self.n_steps + 1)

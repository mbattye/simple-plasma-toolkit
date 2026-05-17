"""Material / BC presets.

Each preset is a partial dict of CLI option defaults; the CLI applies the
preset's values for any flag the user did NOT set explicitly.
"""

from __future__ import annotations

# SpaceX Starship hexagonal heat-shield tile.
#
# Material: silica/ceramic-fibre tile, similar in class to the Shuttle LI-900
# / mullite TPS family. Public estimates put thermal conductivity at roughly
# 0.05-0.15 W/m/K at moderate temperatures; we use 0.1 as a representative
# scalar value (constant-k simplification — see future k(T) work).
#
# Back boundary: tile bonded via a strain-isolation pad (SIP / felt) to the
# 304L stainless steel hull. We model that combined SIP + stud + skin path as
# a Robin BC with an effective contact conductance h and a representative
# steel skin temperature T_inf.
#
# Front boundary: in flight, the hot ceramic face re-radiates a large fraction
# of the incoming heat as IR. Emissivity of LI-900-class TPS is ~0.85-0.90 at
# operating temperatures. Without this radiation channel a steady model with
# low-k TPS predicts unphysically large temperatures (energy has nowhere to
# go but conduct through 25 mm of k=0.1 material). So we turn radiation on by
# default in this preset — that is the physics that actually caps T near
# ~1700 K in flight.
STARSHIP_TILE = {
    "k": 0.1,                     # W/m/K
    "bc_unheated": "adiabatic-back-robin",
    "back_h": 100.0,              # W/m^2/K  effective skin coupling
    "back_T_inf": 400.0,          # K        cool-side steel temperature
    "back_tol_deg": 30.0,
    "back_axis": "0,0,-1",        # tile back is -z, independent of beam angle
    "front_radiation": True,
    "emissivity": 0.89,
    "T_env": 300.0,               # radiative sink (sky / ambient)
}


# Transient extension of the steady Starship tile: belly-flop reentry profile
# with a slow attitude sweep. Two variants — conservative and realistic —
# differ in the q0 amplitude and the AoA sweep range.
#
# Material additions (ρ, c_p) are LI-900-class silica fibre:
#   ρ  ≈ 144 kg/m³
#   c_p ≈ 1200 J/kg/K
#   α  = k/(ρc_p) ≈ 6e-7 m²/s
#   τ  = L²/α  for L=25 mm  ≈ 1000 s
# So 600 s of belly-flop is ~0.6 thermal time constants — the back face is
# only just starting to feel the front-side heat by the end of the pulse.
#
# Conservative: peak q0 = 1e5 W/m². Peak T should stay in the 800-1200 K
# band, well inside LI-900 capability and comfortably away from where
# constant-k breaks down. Suitable for default demos.
#
# Realistic: peak q0 = 1e6 W/m². Closer to actual reentry peak (real
# Starship heating is multi-MW/m² but constant-k silica is an approximation
# above ~1500 K). Use this when you want to push the model.

_STARSHIP_FLIP_COMMON = {
    **STARSHIP_TILE,
    # Transient on by default for these presets.
    "transient": True,
    "duration": 600.0,
    "n_steps": 120,
    "rho": 144.0,
    "cp": 1200.0,
    "q_profile": "gaussian",
    "q_t0": 300.0,        # peak heating ~halfway through reentry
    "q_fwhm": 250.0,
    "angle_profile": "sweep",
    "angle_t0": 100.0,
    "angle_t1": 500.0,
}

STARSHIP_FLIP_CONSERVATIVE = {
    **_STARSHIP_FLIP_COMMON,
    # q0 the user passes is the Gaussian amplitude. Conservative target.
    # Sweep ±15° about a 75° belly-flop attitude.
    "angle_start": 60.0,
    "angle_end": 90.0,
}

STARSHIP_FLIP_REALISTIC = {
    **_STARSHIP_FLIP_COMMON,
    # Wider attitude sweep, otherwise identical. The user supplies a larger
    # --q0 (e.g. 1e6).
    "angle_start": 50.0,
    "angle_end": 90.0,
}


# Showcase variant: wider 3D sweep that gives a more visible arrow trajectory
# in ParaView. Polar 20°→75° emulates a vehicle pitching up from near-vertical
# into the belly-flop, and a 0°→60° azimuth sweep emulates a bank reversal
# during the heat pulse (physically motivated by cross-range control). Same
# q0 amplitude as the conservative preset.
STARSHIP_FLIP_SHOWCASE = {
    **_STARSHIP_FLIP_COMMON,
    "angle_start": 20.0,
    "angle_end": 75.0,
    "azimuth_start": 0.0,
    "azimuth_end": 60.0,
}


PRESETS: dict[str, dict] = {
    "starship": STARSHIP_TILE,
    "starship-flip-conservative": STARSHIP_FLIP_CONSERVATIVE,
    "starship-flip-realistic": STARSHIP_FLIP_REALISTIC,
    "starship-flip-showcase": STARSHIP_FLIP_SHOWCASE,
}

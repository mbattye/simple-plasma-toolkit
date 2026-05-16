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


PRESETS: dict[str, dict] = {
    "starship": STARSHIP_TILE,
}

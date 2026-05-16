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
# scalar value (constant-k simplification — see v2.1 for k(T)).
#
# Back boundary: tile bonded via a strain-isolation pad (SIP / felt) to the
# 304L stainless steel hull. We model that combined SIP + stud + skin path as
# a Robin BC with an effective contact conductance h and a representative
# steel skin temperature T_inf. Sides default to adiabatic, leaving room for
# the `--neighbors hex6` option to add explicit tile-to-tile shadowing.
STARSHIP_TILE = {
    "k": 0.1,                     # W/m/K
    "bc_unheated": "adiabatic-back-robin",
    "back_h": 100.0,              # W/m^2/K effective skin coupling
    "back_T_inf": 400.0,          # K  cool-side steel temperature
    "back_tol_deg": 30.0,
    "back_axis": "0,0,-1",        # tile back is -z, independent of beam angle
}


PRESETS: dict[str, dict] = {
    "starship": STARSHIP_TILE,
}

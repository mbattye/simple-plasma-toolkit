#!/usr/bin/env bash
# Starship belly-flop reentry: conservative q(t) pulse + slow attitude sweep.
#
# Physics:
#   - 600 s reentry window with a Gaussian heating pulse peaking at t=300s
#     (FWHM 250 s). Peak q0 = 1e5 W/m² is well inside LI-900 limits.
#   - Attitude sweep from 60° to 90° polar (about -z) between t=100 and 500 s,
#     mimicking active body-lift modulation during belly-flop.
#   - Front-face radiation re-emits the bulk of the absorbed heat each
#     timestep — without it the steady answer would blow up (see v3 commit).
#
# Run the realistic variant by switching the preset and bumping q0:
#   --preset starship-flip-realistic --q0 1e6
# That pushes peak T into the 1500-1800 K range, closer to the material
# capability but well into where constant-k silica is an approximation.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."

uv run heatstl \
    --stl examples/heat_shield_tile.stl \
    --q0 1e5 \
    --preset starship-flip-conservative \
    --neighbors hex6 \
    --unit mm \
    --out examples/out/starship_flip.xdmf \
    --report examples/out/starship_flip.json \
    --vtu-frames "examples/out/flip_frames/flip_{:04d}.vtu" \
    --verbose

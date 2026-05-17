#!/usr/bin/env bash
# Starship belly-flop "showcase" run: same Gaussian heat pulse as the
# conservative preset, but with a wider 3D attitude sweep so the beam-direction
# arrow visibly traces a curved path through ParaView.
#
# Physics:
#   - Polar θ: 20° → 75°  (vehicle pitching up into the belly-flop)
#   - Azimuth φ: 0° → 60° (bank reversal during the heat pulse — physically
#                          motivated by cross-range control)
#   - Same q(t) Gaussian (peak 1e5 W/m² at t=300 s) as the conservative preset
#   - Front-face radiation on, six hex neighbours, dropped as a ghost STL
#
# Output:
#   examples/out/starship_flip_showcase.xdmf          — tile T(x,t)
#   examples/out/starship_flip_showcase_arrow.xdmf    — animated beam vector
#   examples/out/starship_flip_showcase_neighbors.stl — ghost neighbour tiles
#       (open in ParaView, set Opacity ~0.3, pick a contrasting solid colour)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."

uv run heatstl \
    --stl examples/heat_shield_tile.stl \
    --q0 1e5 \
    --preset starship-flip-showcase \
    --neighbors hex6 \
    --unit mm \
    --out examples/out/starship_flip_showcase.xdmf \
    --report examples/out/starship_flip_showcase.json \
    --verbose

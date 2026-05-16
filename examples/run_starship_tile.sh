#!/usr/bin/env bash
# Starship hexagonal heat-shield tile: realistic-ish preset with neighbour
# shadowing, plus an oblique beam steep enough to actually exercise it.
#
# Two important caveats up front:
#   1. v2 is a STEADY LINEAR model. Real reentry tiles radiate strongly
#      (q_rad = εσT⁴), and that's what keeps the surface near ~1700 K. With
#      radiation out of scope, the steady answer here just reflects how much
#      heat the back face can dump via Robin coupling to the steel skin — so
#      we pick a modest q0 so peak T stays in a useful range. Don't read
#      these temperatures as flight predictions.
#   2. Shadowing only kicks in for grazing beams when all tiles are coplanar:
#      with a steep beam, rays from the central tile's top face go up *over*
#      the neighbours. θ ≳ 70° illuminates the side facets, and *then* the
#      neighbours occlude them.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."

uv run heatstl \
    --stl examples/heat_shield_tile.stl \
    --q0 5e4 \
    --angle-deg 75 \
    --azimuth-deg 0 \
    --preset starship \
    --neighbors hex6 \
    --unit mm \
    --out examples/out/starship_tile.vtu \
    --report examples/out/starship_tile.json \
    --verbose

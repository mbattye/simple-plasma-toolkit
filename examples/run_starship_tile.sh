#!/usr/bin/env bash
# Starship hexagonal heat-shield tile: realistic-ish preset with neighbour
# shadowing, oblique beam, and v3 front-face radiation.
#
# v3 turns εσ(T⁴-T_env⁴) re-radiation on for this preset, which is the
# dominant cooling channel for a low-k silica TPS tile. Without it (v2)
# the steady answer blew up to ~10⁴ K because energy could only escape
# through 25 mm of k=0.1 W/m·K material via the back-face Robin term.
# With radiation on, ~90% of the absorbed beam re-emits off the hot face
# and peak T sits in the right physical band for the chosen q0.
#
# Shadowing only kicks in for grazing beams when all tiles are coplanar:
# with a steep beam, rays from the central tile's top face go up *over*
# the neighbours. θ ≳ 70° illuminates the side facets, and *then* the
# neighbours occlude them.
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

#!/usr/bin/env bash
# Sample heatstl run on the example heat-shield tile.
#
# Tile geometry: ~150 mm wide hex-like tile, 25 mm thick (z in [-12.5, +12.5] mm).
# Loading:
#   q0 = 5 MW/m^2 (representative of a divertor-class plasma-facing tile).
#   direction = (0, 0, -1)  → beam pointing down, hitting the +z face.
#   T_cool = 300 K, k = 150 W/m/K (tungsten-ish).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."

uv run heatstl \
    --stl examples/heat_shield_tile.stl \
    --q0 5e6 \
    --direction 0,0,-1 \
    --T-cool 300 \
    --k 150 \
    --unit mm \
    --out examples/out/heat_shield_tile.vtu \
    --report examples/out/heat_shield_tile.json \
    --verbose

# simple-plasma-toolkit / `heatstl`

A minimal CLI that takes a watertight STL, applies a prescribed incident heat
flux on the exposed surfaces, and solves the **steady-state** heat conduction
problem inside the solid. Output is a `.vtu` temperature field plus a JSON
diagnostics report.

Deliberately small. Not HEAT. No plasma physics, no field-line tracing, no
transient, no `k(T)`, no radiation, no ablation. The flow is:

```
STL → volume mesh → q(n̂, p̂) on exposed facets → linear FEM solve → T(x)
```

## Install

```bash
uv sync --extra dev
```

## Two example runs

**v1 simple case** — high-conductivity tile, all non-heated faces held at `T_cool`:

```bash
bash examples/run_tile.sh
# peak T ≈ 1133 K, energy residual ≈ 0%
```

**Starship preset** — silica TPS tile, Robin coupling on auto-detected back
face, six hex neighbours that shadow the central tile at grazing beams:

```bash
bash examples/run_starship_tile.sh
```

The Starship demo deliberately uses a modest `q0` and a grazing beam
(θ=75°): see [the radiation caveat](#radiation-and-the-starship-preset)
below.

## CLI overview

```bash
uv run heatstl --help
```

### Flux

| Flag | Default | Meaning |
|---|---|---|
| `--stl PATH` | required | Watertight STL. |
| `--q0 FLOAT` | required | Peak incident heat flux, W/m². |
| `--direction x,y,z` | `0,0,-1` | Unit vector p̂ pointing **toward** the surface. |
| `--angle-deg θ` | — | Alternative to `--direction`: polar angle from straight-down. |
| `--azimuth-deg φ` | `0` | Azimuth in xy-plane, used with `--angle-deg`. |
| `--mode {oblique,normal}` | `oblique` | Oblique: `q = q0·max(0, -p̂·n̂)`. Normal: `q0` on every exposed facet. |

`--direction` and `--angle-deg/--azimuth-deg` are mutually exclusive.

### Material

| Flag | Default | Meaning |
|---|---|---|
| `--k` | `150` (or preset) | Thermal conductivity, W/m·K. |

### Boundary conditions on non-heated facets

`--bc-unheated` selects the BC mode:

| Mode | Meaning | Extra flags |
|---|---|---|
| `dirichlet` (default) | `T = T_cool` everywhere except heated facets | `--T-cool` |
| `robin` | `-k ∂T/∂n = h(T − T_inf)` everywhere except heated facets | `--h`, `--T-inf` |
| `adiabatic-back-dirichlet` | Sides adiabatic; Dirichlet on auto-detected back face | `--T-cool`, `--back-tol-deg`, `--back-axis` |
| `adiabatic-back-robin` | Sides adiabatic; Robin on auto-detected back face | `--back-h`, `--back-T-inf`, `--back-tol-deg`, `--back-axis` |

**Back-face detection.** Facets whose outward normal satisfies
`n̂ · b̂ > cos(back_tol_deg)` are flagged as the back face. `b̂` is taken
from `--back-axis` if provided, otherwise it falls back to `--direction`
(so a normal-incidence beam picks the right face automatically). For
oblique beams, pin `--back-axis` to the tile's geometric symmetry axis
(e.g. `--back-axis 0,0,-1`) — the Starship preset does this for you.

### Hex neighbours and shadowing

| Flag | Default | Meaning |
|---|---|---|
| `--neighbors {none,hex6}` | `none` | `hex6`: surround the central tile with six copies on the hex lattice. |
| `--tile-pitch FLOAT` | auto | Centre-to-centre pitch in STL units. Auto = 2 × projected half-extent of the tile. |
| `--tile-gap FLOAT` | `0` | Extra gap added to the pitch. |

When `--neighbors hex6` is set, for each boundary facet of the central tile
the tool casts a single ray from the facet centroid in `-p̂` toward the
beam source. If it hits any neighbour triangle, that facet's incident `q`
is set to 0 (shadowed). Neighbours are placed in the plane perpendicular
to the **tile axis** (`--back-axis`), not the beam — so the lattice stays
flat even for grazing beams.

### Presets

| `--preset` | What it sets |
|---|---|
| `starship` | `k=0.1 W/m·K` (silica-fibre TPS), `bc-unheated=adiabatic-back-robin`, `back-h=100 W/m²/K`, `back-T-inf=400 K`, `back-axis=0,0,-1` |

Presets are *partial* defaults: any flag you set explicitly wins.

### Output

| Flag | Default |
|---|---|
| `--out PATH` | `result.vtu` |
| `--report PATH` | `result.json` |
| `--unit {mm,m}` | `mm` |
| `--mesh-size FLOAT` | auto (`bbox_diag/30`) |

## Physics (v2)

- Steady linear heat equation `∇·(k ∇T) = 0` with constant `k`.
- **Heated** facets: Neumann `-k ∂T/∂n = q` where
  `q = q0·max(0, -p̂·n̂)` (oblique) or `q = q0` (normal).
- **Robin** facets: `-k ∂T/∂n = h (T − T_inf)`.
- **Dirichlet** facets: `T = T_D`.
- **Adiabatic** facets: zero flux (natural BC, no action).

## Radiation and the Starship preset

The Starship preset is deliberately *realistic-in-material-and-attachment*
but **steady-state and non-radiative**. In flight, the tile surface
radiates strongly (`q_rad = εσT⁴`), and that is what keeps the surface
near ~1700 K under multi-MW/m² heating. With radiation out of scope, you
will see unphysically high steady temperatures if you push `q0` to a real
peak-reentry value: the model has no way to dump that energy other than
through the back face, which is rate-limited by `back_h` and the tile's
thermal resistance `L/k`.

For sensible steady demos, keep `q0` modest (the example uses
`5e4 W/m²`) or switch to a high-conductivity material (the v1 example
uses tungsten-ish `k=150 W/m·K`). v3 will add a radiation BC.

## Tests

```bash
uv run pytest               # 18 fast tests
uv run pytest -m slow       # 2 slab analytic comparisons (Dirichlet, Robin)
```

Covers:
- Per-facet flux formulae (normal-incidence, oblique, edge cases).
- Direction parsing (`--direction` and `--angle-deg/--azimuth-deg`).
- Back-face classification under straight and oblique beams.
- Hex neighbour offset placement (perpendicular plane, distances, angular spacing).
- Shadow ray-cast against a known occluder.
- 1D slab analytic comparison for both `dirichlet` and `adiabatic-back-robin` BCs.

## Out of scope for v2

Radiation BC (`q_rad = εσ(T⁴ − T_∞⁴)`), transient simulation, temperature-
dependent `k`, sub-sampled shadowing, multi-beam, ablation, parallel solve,
GUI.

## License

MIT.

# simple-plasma-toolkit / `heatstl`

A minimal CLI that takes a watertight STL, applies a prescribed incident heat
flux on the exposed surfaces, and solves the **steady-state** heat conduction
problem inside the solid. Output is a `.vtu` temperature field plus a JSON
diagnostics report (peak `T`, total heat in, energy-balance residual).

This is deliberately small. It is not HEAT. There is no plasma physics, no
field-line tracing, no transient, no `k(T)`, no shadowing. The goal is:

```
STL → volume mesh → q(n̂, p̂) on exposed facets → linear FEM solve → T(x)
```

## Install

```bash
uv sync --extra dev
```

## Usage

```bash
uv run heatstl \
    --stl examples/heat_shield_tile.stl \
    --q0 5e6 \
    --direction 0,0,-1 \
    --T-cool 300 \
    --k 150 \
    --unit mm \
    --out result.vtu
```

Open `result.vtu` in [ParaView](https://www.paraview.org/) to view `T` and
per-facet `q_face`.

### Flags

| Flag | Default | Meaning |
|---|---|---|
| `--stl PATH` | required | Watertight STL file. |
| `--q0 FLOAT` | required | Peak incident heat flux, W/m². |
| `--direction x,y,z` | `0,0,-1` | Unit vector p̂ pointing **toward** the surface. |
| `--mode {oblique,normal}` | `oblique` | Oblique uses `q = q0·max(0, -p̂·n̂)`; normal applies `q0` to every exposed facet. |
| `--T-cool FLOAT` | `300` | Cool-side Dirichlet temperature, K. |
| `--k FLOAT` | `150` | Thermal conductivity, W/m·K. |
| `--unit {mm,m}` | `mm` | STL units — converted to SI internally. |
| `--mesh-size FLOAT` | auto | gmsh target element size, in STL units. |
| `--out PATH` | `result.vtu` | Output VTU. |
| `--report PATH` | `result.json` | Diagnostics report. |

## Physics (v1)

- Steady linear heat equation: `∇·(k∇T) = 0` with constant `k`.
- **Heated** facets: Neumann `-k ∂T/∂n = q` where
  `q = q0·max(0, -p̂·n̂)` (oblique) or `q = q0` (normal).
- **Non-heated** facets: Dirichlet `T = T_cool` (back-cooled tile default).

Future BC modes (Robin/convection, adiabatic + back-face Dirichlet) are
scoped for v2.

## Validation

`tests/` includes:

- Pure-function tests for the per-facet flux formula.
- A 1D slab analytic comparison.
- An energy-balance sanity check on the example tile.

Run with `uv run pytest`.

## Out of scope for v1

Tokamak exhaust modelling (EFIT/Eich/HEAT), magnetic tracing, transients,
`k(T)`, shadowing, multi-beam, radiation, ablation, parallel solve, GUI.

## License

MIT.

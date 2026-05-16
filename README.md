# simple-plasma-toolkit / `heatstl`

A minimal CLI that takes a watertight STL, applies a prescribed incident heat
flux on the exposed surfaces, and solves the **steady-state** heat conduction
problem inside the solid. Output is a `.vtu` temperature field plus a JSON
diagnostics report.

Deliberately small. Not HEAT. No plasma physics, no field-line tracing, no
`k(T)`, no ablation. v3 added Stefan–Boltzmann front-face radiation
(Newton-solved). **v4 adds transient time-stepping** with time-varying
heat flux and beam direction (Starship-flip-style demo included). The flow is:

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

**Starship steady** — silica TPS tile, Robin back, six hex neighbours,
front-face radiation:

```bash
bash examples/run_starship_tile.sh
# peak T ≈ 1030 K, Q_rad/Q_in ≈ 94%, energy residual < 1%
```

**Starship-flip transient (v4)** — 600 s belly-flop with a Gaussian `q(t)`
pulse peaking at t=300 s and a slow 60°→90° attitude sweep:

```bash
bash examples/run_starship_flip.sh
# peak T ≈ 1250 K at t = 305 s, transient energy balance < 2% during the
# heating window, 120 timesteps in ~20 s wall time
```

Open `examples/out/starship_flip.xdmf` in ParaView to scrub through time.
A companion `examples/out/starship_flip_arrow.xdmf` is written too — load
it alongside, apply **Filters → Glyph** with `Vectors = incident_scaled`
and `Glyph Type = Arrow` to see an animated arrow indicating the beam
direction and pulse magnitude.

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

### Transient mode (v4)

| Flag | Default | Meaning |
|---|---|---|
| `--transient / --steady` | steady (presets may override) | Solve `ρc_p ∂T/∂t = ∇·(k∇T)` with backward Euler. |
| `--duration` | required if transient | Simulation time, s. |
| `--n-steps` | required if transient | Number of timesteps. |
| `--rho`, `--cp` | required if transient | Material density and specific heat. |
| `--T-initial` | `300` | Initial temperature, K. |
| `--q-profile {constant,ramp,gaussian,piecewise}` | `constant` | `q0(t)` shape. |
| `--q-csv PATH` | — | For `piecewise`: 2-col CSV `(t, q)`. |
| `--q-ramp-t` | `1.0` | Time to reach `q0` in `ramp`. |
| `--q-t0`, `--q-fwhm` | — | Gaussian peak time and FWHM. |
| `--angle-profile {constant,sweep,piecewise}` | `constant` | `p̂(t)` shape. |
| `--angle-start`, `--angle-end`, `--angle-t0`, `--angle-t1` | — | Sweep parameters (polar deg, s). |
| `--angle-csv PATH` | — | For `piecewise`: 3-col CSV `(t, θ_deg, φ_deg)`. |
| `--vtu-frames PATTERN` | — | Also write numbered VTUs (e.g. `out/flip_{:04d}.vtu`). |

In transient mode, output defaults to `result.xdmf` (mesh + time series in
one file pair). Per-step diagnostics — `peak_T`, `Q_in`, `Q_radiated`,
`Q_conducted_out`, `dU_dt_W`, and the transient energy balance
`residual_transient_rel = (Q_in − Q_out − ∂U/∂t)/Q_in` — go in the JSON
report.

### Presets

| `--preset` | What it sets |
|---|---|
| `starship` | Steady. `k=0.1 W/m·K` silica TPS, `adiabatic-back-robin`, `back-h=100`, `back-T-inf=400`, `back-axis=0,0,-1`, radiation on with `ε=0.89`, `T-env=300`. |
| `starship-flip-conservative` | Transient. 600 s, 120 steps, ρ=144, c_p=1200, Gaussian `q(t)` peaking at t=300 s with FWHM 250 s, attitude sweep 60°→90° between 100 and 500 s. |
| `starship-flip-realistic` | As above but wider attitude sweep (50°→90°). Drive with a larger `--q0` (e.g. `1e6`); peak T pushes 1500–1800 K, into the constant-k approximation limit. |

Presets are *partial* defaults: any flag you set explicitly wins.

### Output

| Flag | Default |
|---|---|
| `--out PATH` | `result.vtu` |
| `--report PATH` | `result.json` |
| `--unit {mm,m}` | `mm` |
| `--mesh-size FLOAT` | auto (`bbox_diag/30`) |

## Physics (v4)

- Steady: `∇·(k ∇T) = 0`.
- Transient: `ρ c_p ∂T/∂t = ∇·(k ∇T)`, backward Euler in time. The
  radiation Newton from v3 becomes the inner loop of each time step.
  Time-invariant assemblies (`K`, `M`, back-side Robin, radiation set) are
  built once outside the time loop; heated load and the Newton-linearised
  radiation matrix update each step.
- **Heated** facets: Neumann `-k ∂T/∂n = q` where
  `q = q0·max(0, -p̂·n̂)` (oblique) or `q = q0` (normal).
- **Robin** facets: `-k ∂T/∂n = h (T − T_inf)`.
- **Dirichlet** facets: `T = T_D`.
- **Radiation** facets (v3): `k ∂T/∂n = q − εσ(T⁴ − T_env⁴)`. Nonlinear;
  solved by Newton-linearisation `T⁴ ≈ 4T_k³T − 3T_k⁴`, which adds a
  Robin-like mass `4εσT_k³` to the LHS and a constant `3εσT_k⁴ + εσT_env⁴`
  to the RHS each iteration. Initial guess is the pure-radiation
  equilibrium `T = (q_avg/(εσ))^¼`; typically converges in 4–8 iterations.
- **Adiabatic** facets: zero flux (natural BC, no action).

### Front-face radiation flags

| Flag | Default | Meaning |
|---|---|---|
| `--front-radiation / --no-front-radiation` | off (presets may override) | Enable Stefan–Boltzmann on outward-facing facets. |
| `--emissivity` | `0.89` | ε, silica TPS-class. |
| `--T-env` | `300` | Radiative sink temperature, K. |
| `--newton-tol` | `1e-4` | Relative tolerance on Newton update. |
| `--newton-max-iter` | `50` | Hard cap. |

Radiation attaches to every boundary facet whose outward normal points
forward (`n̂ · (−back_axis) > 0.05`), so a hot side facet still re-radiates
when the beam misses it. Without this fix the v2 Starship preset blew up
to ~10⁴ K because energy could only escape through 25 mm of `k=0.1` TPS.

## Tests

```bash
uv run pytest               # fast unit tests
uv run pytest -m slow       # slab analytics + Starship regression
```

Covers:
- Per-facet flux formulae (normal-incidence, oblique, edge cases).
- Direction parsing (`--direction` and `--angle-deg/--azimuth-deg`).
- Back-face classification under straight and oblique beams.
- Hex neighbour offset placement (perpendicular plane, distances, angular spacing).
- Shadow ray-cast against a known occluder.
- 1D slab analytic comparison for `dirichlet` and `adiabatic-back-robin` BCs.
- 1D pure-radiation slab matches `(q/(εσ) + T_env⁴)^¼`.
- 1D radiation + Robin-back 1D balance via brentq.
- Starship demo regression: peak T < 1500 K, residual < 2%, `Q_rad/Q_in > 0.7`.

## Out of scope for v4

Temperature-dependent `k(T)`, sub-sampled shadowing, multi-beam, ablation,
parallel solve, GUI. Next planned: Cloud Run engine for Analog.

## License

MIT.

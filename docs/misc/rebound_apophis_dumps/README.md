# Apophis-flyby REBOUND/ASSIST reproducibility bundle

Raw IAS15 dumps and acceleration evaluations from REBOUND/ASSIST that
back the Apophis-flyby IAS15 investigation writeup. The companion
notebook (`docs/tutorials/.../apophis_investigation.ipynb`) reads
`outputs/*/parsed.npz` directly; you only need to regenerate this
bundle's contents if you want to re-run the C-side dumps from scratch.

## What's in here

```
rebound_apophis_dumps/
├── README.md                            (this file)
├── .gitignore                           (excludes ephemeris/ from git)
├── rebound_patches/
│   ├── integrator_ias15.c.patch         unified diff vs vanilla REBOUND 4.5.1
│   ├── integrator_ias15.c               full instrumented file (fallback)
│   └── README_patch.md                  what the patch adds and why
├── run_simulations.py                   one entry-point with subcommands
├── parse_dumps.py                       raw stdout log -> structured NPZ
├── ephemeris/                           ~830 MB; gitignored
└── outputs/
    ├── postflyby_ic.npz                 day-105 cached state for sims 3-5
    ├── sim1_yearlong_global/            year-long, GLOBAL controller
    ├── sim2_yearlong_prs23/             year-long, PRS23 controller
    ├── sim3_singlestep_dt0.001/         post-flyby IC, dt=0.001
    ├── sim4_singlestep_dt0.1/           post-flyby IC, dt=0.1
    └── sim5_acceleration_peel/          post-flyby IC, no integration
```

The five simulations are described in detail below under "Simulations".

## Quickstart: just use the data

Every `outputs/sim*/parsed.npz` is loadable with `numpy.load`. The
schema is documented inline in `parse_dumps.py`. There is no need to
regenerate anything just to read these files.

```python
import numpy as np
sim3 = np.load("outputs/sim3_singlestep_dt0.001/parsed.npz")
# sim3["sub_at"] : (n_iters, 7, N3=3)  per-substep accelerations
# sim3["final_b"]: (7, N3)             converged b coefficients
```

## Regenerating the data

This is the path that requires building a patched REBOUND.

### Prerequisites

- macOS or Linux (tested on macOS Darwin 25.4 / arm64)
- `git`, `make`, a C compiler
- `uv` (Python package manager), or `pip` + a venv tool of your choice
- ~5 GB free disk
- The four ephemeris files in `ephemeris/`. They are not committed to
  git (see `.gitignore`). To populate the directory:

  | File                       | Source                                                                                       | Size   |
  |----------------------------|----------------------------------------------------------------------------------------------|--------|
  | `linux_p1550p2650.440`     | NAIF: <https://ssd.jpl.nasa.gov/ftp/eph/planets/Linux/de440/>                                 | 98 MB  |
  | `de440.bsp`                | NAIF: <https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp>             | 114 MB |
  | `sb441-n16.bsp`            | NAIF: <https://ssd.jpl.nasa.gov/ftp/eph/small_bodies/asteroids_de441/sb441-n16.bsp>          | 616 MB |
  | `sb-99942-220.bsp`         | JPL Horizons SPK for asteroid 99942 Apophis (also at `paper/data/sb-99942-220.bsp` in this repo) | 826 KB |

### 1. Build a patched REBOUND

```bash
# Pick any directory; we'll use /tmp here.
git clone --depth 1 --branch 4.5.1 https://github.com/hannorein/rebound.git /tmp/rebound
cd /tmp/rebound
git apply /path/to/rebound_apophis_dumps/rebound_patches/integrator_ias15.c.patch
cd src && make
```

`make` produces `src/librebound.so` (the file is `.so` even on macOS in
this build chain).

If `git apply` fails ("patch does not apply"), fall back to:

```bash
cp /path/to/rebound_apophis_dumps/rebound_patches/integrator_ias15.c \
   /tmp/rebound/src/integrator_ias15.c
cd /tmp/rebound/src && make
```

### 2. Set up a Python environment with REBOUND 4.5.1 + ASSIST

The pip-installed `rebound==4.5.1` and `assist==1.1.10` are ABI-compatible
with the patched library; we just swap the shared library in.

```bash
cd /path/to/rebound_apophis_dumps   # the bundle directory
uv venv .repro-venv
source .repro-venv/bin/activate     # bash/zsh
uv pip install "rebound==4.5.1" "assist==1.1.10" \
               numpy astropy jplephem
```

Now replace pip's vanilla librebound with the patched build:

```bash
# macOS (note the ".cpython-3*-darwin.so" naming)
cp /tmp/rebound/src/librebound.so \
   .repro-venv/lib/python3.13/site-packages/librebound.cpython-313-darwin.so

# Linux
cp /tmp/rebound/src/librebound.so \
   .repro-venv/lib/python*/site-packages/librebound.cpython-*-linux-gnu.so
```

Adjust `python3.13` / `cpython-313` for your Python minor version. The
file name on disk is reported by:

```bash
python -c "import rebound, os; print(os.path.dirname(rebound.clibrebound.__file__) if hasattr(rebound.clibrebound,'__file__') else rebound.clibrebound._name)"
```

### 3. Run the sims

The script self-relaunches with `IAS15_DUMP_PC=1` set, so you don't
have to pass it manually. On macOS you must set `DYLD_LIBRARY_PATH` to
point at the venv's `site-packages` so ASSIST's shared lib can find
`librebound`:

```bash
# macOS
export DYLD_LIBRARY_PATH=$(pwd)/.repro-venv/lib/python3.13/site-packages
# Linux
export LD_LIBRARY_PATH=$(pwd)/.repro-venv/lib/python3.13/site-packages
```

Then:

```bash
# All five plus the cached IC, in dependency order:
python run_simulations.py all

# Or one at a time:
python run_simulations.py compute_postflyby_ic
python run_simulations.py sim1
python run_simulations.py sim2
python run_simulations.py sim3
python run_simulations.py sim4
python run_simulations.py sim5
```

Each run writes:
- `outputs/<sim>/raw_stdout.log` — the captured C-side dump stream (raw,
  human-readable).
- `outputs/<sim>/parsed.npz` — structured arrays from `parse_dumps.py`.
- For sims 3 and 4: `meta.json` with PC iteration count and IC reference.
- For sim 5: `component_order.json` documenting the cumulative subset
  ordering.

Expected runtime on a recent laptop: sim1 takes ~30-60 s, sim2 ~3-5 s,
sims 3/4/5 each ~2 s. Total under ~2 minutes.

Expected step counts:
- sim1 (year-long, GLOBAL): **2117** steps
- sim2 (year-long, PRS23): **~133–137** steps
- sim3 (single step, dt=0.001): **2 PC iters**
- sim4 (single step, dt=0.1): **4 PC iters**

## Simulations

All sims share these settings:

- **Initial condition**: Apophis at 2029-01-01 TDB (JD 2462137.5),
  computed by reading heliocentric position from `sb-99942-220.bsp` and
  Sun barycentric position from `de440.bsp`, summed.
- **Force model**: 7 components — Sun, planets (Newtonian), asteroids
  (Newtonian), GR_EIH (Einstein-Infeld-Hoffmann post-Newtonian for the
  first 11 perturbers), Sun harmonics (J2 only), Earth harmonics
  (J2/J3/J4), Marsden non-gravitational
  (`A1=4.999999873689e-13, A2=-2.901085508711e-14, A3=0`).
- **`extras.gr_eih_sources = 11`** — Sun + 8 planets + Moon + Pluto.
- **Coordinates**: barycentric ICRS, AU and AU/day, TDB time.

### sim1 — year-long, GLOBAL controller

Integrate from 2029-01-01 TDB to 2030-01-01 TDB (Δt = 365.25 d).
`adaptive_mode = "global"`, `epsilon = 1e-9`, `min_dt = 0.001`. The
`min_dt = 0.001` is critical — it matches the
[ASSIST online Apophis tutorial](https://assist.readthedocs.io/en/latest/jupyter_examples/Apophis/);
without it the controller grinds down to ~1e-7-day steps near closest
approach and the run takes hours instead of seconds.

Output records every accepted step's
- `[IAS15_INIT]` step-start state (x0, v0, a0, csx, csv, csa0, b, csb, g)
- `[IAS15_SUB]` per-substep state for each PC iteration (x, v, at,
  g_pre, g, b, csb)
- `[IAS15_PC]` per-iteration convergence error
- `[IAS15_FINAL]` converged b, csb, g, e

`parsed.npz` contains object arrays of length `n_steps`, one entry per
step, plus a top-level `n_steps` scalar. Schema details in
`parse_dumps.py`.

### sim2 — year-long, PRS23 controller

Same span/IC/forces, `adaptive_mode = 2` (REBOUND's PRS23 mode),
`epsilon = 1e-9`, `min_dt = 0.001`. Same output schema.

### sim3 — single step at post-flyby IC, dt = 0.001

IC from `outputs/postflyby_ic.npz` — produced by
`compute_postflyby_ic`, which runs sim1's setup forward by 105 days
(just past closest approach at day ~102.4) and dumps the final state.
This IC is the "post-flyby trap" point used in the investigation's
Phase 11/12.

`dt = 0.001`, `epsilon = 0.0`, `min_dt = 0.0` (adaptivity disabled,
step locked to the requested `dt`). `adaptive_mode` set to "global"
but irrelevant.

`parsed.npz` is a flat-array NPZ (single step). `meta.json` records
`n_pc_iters` and the IC values. Reference: `n_pc_iters = 2`,
`max|b6| ≈ 6.05e-14`.

### sim4 — single step at post-flyby IC, dt = 0.1

Same IC, `dt = 0.1`. Reference: `n_pc_iters = 4`,
`max|b6| ≈ 7.54e-14`.

### sim5 — acceleration peel at post-flyby IC

For each of seven cumulative force-model subsets, build a fresh
`rebound.Simulation` + `assist.Extras`, set
`extras.forces = subset`, and call
`rebound.clibrebound.reb_simulation_update_acceleration(byref(sim))` to
populate `particles[0].a{x,y,z}` at the IC. **No integration step is
taken.**

Cumulative subset order (each row is a superset of the prior):

| idx | label                | forces                                                                   |
|-----|----------------------|--------------------------------------------------------------------------|
| 0   | `01_sun_newt`        | `["SUN"]`                                                                |
| 1   | `02_planets_newt`    | `+ "PLANETS"`                                                            |
| 2   | `03_planets_gr`      | `+ "GR_EIH"`                                                             |
| 3   | `04_+asteroids`      | `+ "ASTEROIDS"`                                                          |
| 4   | `05_+sun_harmonics`  | `+ "SUN_HARMONICS"`                                                      |
| 5   | `06_+earth_harmonics`| `+ "EARTH_HARMONICS"`                                                    |
| 6   | `07_full`            | `+ "NON_GRAVITATIONAL"` (= full force model)                             |

`parsed.npz` contains `a_per_subset[7, 3]` plus the IC.
`component_order.json` is the human-readable manifest.

## Caveats / known gotchas

- **REBOUND 4.5.1 is mandatory.** The C struct `reb_simulation` layout
  differs between 4.5.1 and 4.6.0; using a different tag will likely
  break the C struct layout when ASSIST tries to use the patched
  librebound. Confirm by `python -c "import rebound; print(rebound.__version__)"`
  and check it matches.
- **macOS `DYLD_LIBRARY_PATH`** is required when ASSIST is loaded.
  Linux uses `LD_LIBRARY_PATH`.
- **`min_dt = 0.001` is mandatory for GLOBAL** to match the ASSIST
  online Apophis tutorial; without it ASSIST grinds to ~1e-7-day steps
  near closest approach and the run takes hours.
- **Compiler flags matter.** REBOUND's pip-installed `librebound` is
  built without `-O3`, with `-g`. The bundle's reference output was
  generated with the same flags. Don't change `OPT` in the Makefile
  unless you're prepared for FP-level drift in the comparison.
- **Bit-faithful reproduction is expected for sims 3–5** (single-step,
  deterministic) given the same compiler version. Year-long sims 1–2
  are also expected to be bit-faithful in step count and per-step
  values, but tiny FP drift across rebuilds with different libc /
  fenv defaults is possible.

## What was instrumented in REBOUND

See `rebound_patches/README_patch.md`. TL;DR: four diagnostic line
types are added to `reb_integrator_ias15_step` and `IAS15_DUMP_PC=1`
turns them on. Off-path overhead is one env-var read per process.

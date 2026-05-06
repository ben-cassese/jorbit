"""Generate the five Apophis-flyby REBOUND dumps backing the IAS15 writeup.

Subcommands:
    compute_postflyby_ic   — ASSIST-integrate 2029-01-01 -> day 105, cache state
    sim1                   — year-long, GLOBAL controller
    sim2                   — year-long, PRS23 controller
    sim3                   — single step at post-flyby IC, dt=0.001
    sim4                   — single step at post-flyby IC, dt=0.1
    sim5                   — acceleration peel at post-flyby IC (no integration)
    all                    — all of the above in order

All sims that emit [IAS15_*] dumps run with IAS15_DUMP_PC=1 against the
patched librebound built per the bundle README. The script tee's REBOUND's
C-side stdout into outputs/sim*/raw_stdout.log and post-processes via
parse_dumps.py into outputs/sim*/parsed.npz.

Run with:
    DYLD_LIBRARY_PATH=/path/to/venv/lib/.../site-packages \
      uv run python run_simulations.py <subcommand>

(The script self-relaunches with IAS15_DUMP_PC=1 if not already set, so the
caller does not need to pass it.)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EPHEM_DIR = HERE / "ephemeris"
OUTPUTS_DIR = HERE / "outputs"
POSTFLYBY_IC_NPZ = OUTPUTS_DIR / "postflyby_ic.npz"


def _relaunch_with_dump_env() -> None:
    """Re-exec self with IAS15_DUMP_PC=1 if it isn't already set."""
    if os.environ.get("IAS15_DUMP_PC") != "1":
        env = dict(os.environ)
        env["IAS15_DUMP_PC"] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


# ============================================================================
# Subcommand-aware import of REBOUND/ASSIST. sim5 doesn't need IAS15_DUMP_PC
# since it reads accelerations directly via ctypes; everything else does.
# ============================================================================
def _setup_env(needs_dump: bool) -> None:
    if needs_dump:
        _relaunch_with_dump_env()


# Force-model and physical constants (match docs/tutorials/.../apophis_flyby.ipynb)
T_START_ISO = "2029-01-01"
DAYS_IN_YEAR = 365.25
DAYS_TO_POSTFLYBY = 105.0  # post-closest-approach floor-trap region (day 102.4 = CA)

NONGRAV = (4.999999873689e-13, -2.901085508711e-14, 0.0)
FULL_FORCES = [
    "SUN",
    "PLANETS",
    "ASTEROIDS",
    "GR_EIH",
    "SUN_HARMONICS",
    "EARTH_HARMONICS",
    "NON_GRAVITATIONAL",
]
GR_EIH_SOURCES = 11

# Component-peel cumulative subsets for sim5
PEEL_SUBSETS = [
    ("01_sun_newt", ["SUN"]),
    ("02_planets_newt", ["SUN", "PLANETS"]),
    ("03_planets_gr", ["SUN", "PLANETS", "GR_EIH"]),
    ("04_+asteroids", ["SUN", "PLANETS", "GR_EIH", "ASTEROIDS"]),
    ("05_+sun_harmonics", ["SUN", "PLANETS", "GR_EIH", "ASTEROIDS", "SUN_HARMONICS"]),
    (
        "06_+earth_harmonics",
        ["SUN", "PLANETS", "GR_EIH", "ASTEROIDS", "SUN_HARMONICS", "EARTH_HARMONICS"],
    ),
    ("07_full", FULL_FORCES),
]

AU_KM = 1.495978707e8


# ============================================================================
# Initial conditions: Apophis at 2029-01-01 TDB barycentric
# ============================================================================
def build_apophis_ic_2029() -> tuple:
    """Return (bary_x, bary_v, t_start_jd_tdb).

    Reads Apophis's heliocentric position from the JPL #220 SPK and the
    Sun's barycentric position from DE440, sums to get the barycentric
    Apophis state.
    """
    import numpy as np
    from astropy.time import Time
    from jplephem.spk import SPK

    apophis_spk = EPHEM_DIR / "sb-99942-220.bsp"
    de440_spk = EPHEM_DIR / "de440.bsp"
    if not apophis_spk.exists() or not de440_spk.exists():
        raise FileNotFoundError(
            f"Need {apophis_spk} and {de440_spk}. See README.md, 'Ephemeris files'."
        )

    t_start = float(Time(T_START_ISO, scale="tdb").tdb.jd)

    # Apophis heliocentric position
    spk = SPK.open(str(apophis_spk))
    starts = np.array([seg.start_jd for seg in spk.segments])
    order = starts.argsort()
    seg_idx = int(np.searchsorted(starts[order], t_start) - 1)
    real_seg = spk.segments[order[seg_idx]]
    pos_km, vel_km_per_day = real_seg.compute_and_differentiate(t_start)
    helio_x = pos_km / AU_KM
    helio_v = vel_km_per_day / AU_KM

    # Sun barycentric position (target=10 around solar-system-barycenter=0)
    de440 = SPK.open(str(de440_spk))
    sun_seg = next(s for s in de440.segments if s.target == 10 and s.center == 0)
    sun_pos_km, sun_vel_km_per_day = sun_seg.compute_and_differentiate(t_start)
    sun_x = sun_pos_km / AU_KM
    sun_v = sun_vel_km_per_day / AU_KM

    bary_x = (helio_x + sun_x).astype(np.float64)
    bary_v = (helio_v + sun_v).astype(np.float64)
    return bary_x, bary_v, t_start


# ============================================================================
# REBOUND/ASSIST sim builder
# ============================================================================
def _build_sim(x, v, t_abs_jd, forces):
    """Build a REBOUND.Simulation + assist.Extras for Apophis at (x, v, t)."""
    import numpy as np
    import rebound
    import assist

    sim = rebound.Simulation()
    eph = assist.Ephem(
        str(EPHEM_DIR / "linux_p1550p2650.440"),
        str(EPHEM_DIR / "sb441-n16.bsp"),
    )
    sim.add(
        rebound.Particle(
            x=float(x[0]),
            y=float(x[1]),
            z=float(x[2]),
            vx=float(v[0]),
            vy=float(v[1]),
            vz=float(v[2]),
        )
    )
    sim.t = float(t_abs_jd) - eph.jd_ref
    extras = assist.Extras(sim, eph)
    extras.forces = forces
    extras.gr_eih_sources = GR_EIH_SOURCES
    extras.particle_params = np.array(NONGRAV, dtype=np.float64)
    return sim, eph, extras


# ============================================================================
# stdout-to-file capture
# ============================================================================
class _ReroutedStdout:
    """Redirect both Python's sys.stdout and the C-side fd 1 to `path`.

    Necessary because the IAS15_DUMP_PC printf's are emitted by C code; a
    pure Python `with open(...) as sys.stdout` won't catch them.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._saved_fd: int | None = None
        self._fh = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", buffering=1)
        sys.stdout.flush()
        self._saved_fd = os.dup(1)
        os.dup2(self._fh.fileno(), 1)
        return self

    def __exit__(self, *exc):
        sys.stdout.flush()
        try:
            os.fsync(1)
        except OSError:
            pass
        if self._saved_fd is not None:
            os.dup2(self._saved_fd, 1)
            os.close(self._saved_fd)
        if self._fh is not None:
            self._fh.close()


# ============================================================================
# compute_postflyby_ic — produces outputs/postflyby_ic.npz
# ============================================================================
def cmd_compute_postflyby_ic() -> None:
    import numpy as np

    bary_x, bary_v, t_start = build_apophis_ic_2029()
    print(f"[IC build] Apophis at JD {t_start} TDB")
    print(f"  bary_x = {bary_x}")
    print(f"  bary_v = {bary_v}")

    print(
        f"[postflyby] integrating with full forces, GLOBAL+min_dt=0.001 -> day {DAYS_TO_POSTFLYBY}"
    )
    sim, eph, extras = _build_sim(bary_x, bary_v, t_start, FULL_FORCES)
    sim.dt = 0.001
    sim.ri_ias15.epsilon = 1e-9
    sim.ri_ias15.min_dt = 0.001
    sim.ri_ias15.adaptive_mode = "global"

    t_target = t_start + DAYS_TO_POSTFLYBY
    extras.integrate_or_interpolate(t_target - eph.jd_ref)

    p = sim.particles[0]
    ic_x = np.array([p.x, p.y, p.z], dtype=np.float64)
    ic_v = np.array([p.vx, p.vy, p.vz], dtype=np.float64)
    print(f"[postflyby] state at JD {t_target}")
    print(f"  x = {ic_x}")
    print(f"  v = {ic_v}")
    print(f"  |x| = {np.linalg.norm(ic_x):.6f} AU")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        POSTFLYBY_IC_NPZ,
        x=ic_x,
        v=ic_v,
        t=np.array(t_target),
        bary_x_2029=bary_x,
        bary_v_2029=bary_v,
        t_start_2029=np.array(t_start),
        days_past_2029=np.array(DAYS_TO_POSTFLYBY),
    )
    print(f"saved {POSTFLYBY_IC_NPZ}")


def _load_postflyby_ic():
    import numpy as np

    if not POSTFLYBY_IC_NPZ.exists():
        raise FileNotFoundError(
            f"{POSTFLYBY_IC_NPZ} not found. Run: python run_simulations.py compute_postflyby_ic"
        )
    z = np.load(POSTFLYBY_IC_NPZ)
    return z["x"], z["v"], float(z["t"])


# ============================================================================
# Year-long sims (sim1, sim2)
# ============================================================================
def _yearlong(sim_label: str, adaptive_mode: str) -> None:
    """Year-long Apophis integration with IAS15_DUMP_PC=1 stream capture.

    adaptive_mode: "global" or "prs23" (i.e., 1 or 2 in REBOUND's enum).
    """
    import numpy as np

    out_dir = OUTPUTS_DIR / sim_label
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "raw_stdout.log"
    npz_path = out_dir / "parsed.npz"

    bary_x, bary_v, t_start = build_apophis_ic_2029()
    print(f"[{sim_label}] year-long Apophis 2029 -> 2030, controller={adaptive_mode}")

    sim, eph, extras = _build_sim(bary_x, bary_v, t_start, FULL_FORCES)
    sim.dt = 0.001
    sim.ri_ias15.epsilon = 1e-9
    sim.ri_ias15.min_dt = 0.001
    sim.ri_ias15.adaptive_mode = adaptive_mode

    t_target = t_start + DAYS_IN_YEAR

    print(f"[{sim_label}] capturing C-side stdout -> {log_path}")
    with _ReroutedStdout(log_path):
        extras.integrate_or_interpolate(t_target - eph.jd_ref)

    print(f"[{sim_label}] integration finished. parsing log...")
    from parse_dumps import parse_log, save_npz

    steps = parse_log(log_path)
    save_npz(steps, npz_path)
    print(f"[{sim_label}] {len(steps)} steps -> {npz_path}")
    if len(steps) > 0:
        dts = np.array([s["dt"] for s in steps])
        n_iters = np.array([s["n_iters"] for s in steps])
        print(f"  dt: min={dts.min():.6g}  max={dts.max():.6g}  mean={dts.mean():.6g}")
        print(
            f"  PC iters: min={n_iters.min()}  max={n_iters.max()}  mean={n_iters.mean():.2f}"
        )


def cmd_sim1() -> None:
    _yearlong("sim1_yearlong_global", "global")


def cmd_sim2() -> None:
    _yearlong("sim2_yearlong_prs23", 2)  # REBOUND's "prs23" enum int


# ============================================================================
# Single-step sims at post-flyby IC (sim3, sim4)
# ============================================================================
def _single_step(sim_label: str, dt: float) -> None:
    import numpy as np

    out_dir = OUTPUTS_DIR / sim_label
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "raw_stdout.log"
    npz_path = out_dir / "parsed.npz"
    meta_path = out_dir / "meta.json"

    ic_x, ic_v, ic_t = _load_postflyby_ic()
    print(f"[{sim_label}] one IAS15 step at post-flyby IC, dt={dt}")
    sim, eph, extras = _build_sim(ic_x, ic_v, ic_t, FULL_FORCES)
    sim.dt = dt
    sim.ri_ias15.epsilon = 0.0  # disable adaptivity
    sim.ri_ias15.min_dt = 0.0
    sim.ri_ias15.adaptive_mode = "global"

    print(f"[{sim_label}] capturing -> {log_path}")
    with _ReroutedStdout(log_path):
        sim.steps(1)

    from parse_dumps import parse_log, save_npz

    steps = parse_log(log_path)
    if len(steps) != 1:
        raise RuntimeError(f"expected 1 step, parsed {len(steps)}")
    save_npz(steps, npz_path)

    s = steps[0]
    meta = {
        "sim_label": sim_label,
        "dt": dt,
        "ic_x": ic_x.tolist(),
        "ic_v": ic_v.tolist(),
        "ic_t_jd_tdb": ic_t,
        "n_pc_iters": int(s["n_iters"]),
        "max_abs_b6": float(np.max(np.abs(s["final_b"][6]))),
        "forces": FULL_FORCES,
        "nongrav_a1_a2_a3": list(NONGRAV),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[{sim_label}] n_pc_iters={s['n_iters']}  max|b6|={meta['max_abs_b6']:.6e}")
    print(f"[{sim_label}] -> {npz_path}, {meta_path}")


def cmd_sim3() -> None:
    _single_step("sim3_singlestep_dt0.001", 0.001)


def cmd_sim4() -> None:
    _single_step("sim4_singlestep_dt0.1", 0.1)


# ============================================================================
# Acceleration peel (sim5) — no integration, just _update_accel per subset
# ============================================================================
def cmd_sim5() -> None:
    import numpy as np
    import rebound
    from ctypes import POINTER, byref

    sim_label = "sim5_acceleration_peel"
    out_dir = OUTPUTS_DIR / sim_label
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "raw_stdout.log"
    npz_path = out_dir / "parsed.npz"
    order_path = out_dir / "component_order.json"

    update_accel = rebound.clibrebound.reb_simulation_update_acceleration
    update_accel.restype = None
    update_accel.argtypes = [POINTER(rebound.Simulation)]

    ic_x, ic_v, ic_t = _load_postflyby_ic()
    print(f"[{sim_label}] acceleration peel at post-flyby IC, no integration")

    a_per_subset = np.zeros((len(PEEL_SUBSETS), 3), dtype=np.float64)
    log_lines: list[str] = []
    log_lines.append(f"# IC: x={ic_x.tolist()}  v={ic_v.tolist()}  t_jd_tdb={ic_t}")
    log_lines.append("# subset_idx  label  ax  ay  az")
    for idx, (label, forces) in enumerate(PEEL_SUBSETS):
        sim, eph, extras = _build_sim(ic_x, ic_v, ic_t, forces)
        # Make sure the ASSIST forces hook is registered before _update_accel
        sim.t = float(ic_t) - eph.jd_ref
        update_accel(byref(sim))
        p = sim.particles[0]
        a = np.array([p.ax, p.ay, p.az], dtype=np.float64)
        a_per_subset[idx] = a
        line = f"{idx}  {label}  {a[0]:.17e}  {a[1]:.17e}  {a[2]:.17e}"
        log_lines.append(line)
        print(f"  [{idx}] {label:24s}  ||a|| = {np.linalg.norm(a):.6e}")

    log_path.write_text("\n".join(log_lines) + "\n")

    np.savez_compressed(
        npz_path,
        ic_x=ic_x,
        ic_v=ic_v,
        ic_t=np.array(ic_t),
        a_per_subset=a_per_subset,
        labels=np.array([lab for lab, _ in PEEL_SUBSETS]),
    )

    order_path.write_text(
        json.dumps(
            {
                "description": (
                    "Cumulative force-model peel. Each entry's `forces` list is "
                    "passed to assist.Extras.forces, then "
                    "rebound.clibrebound.reb_simulation_update_acceleration is "
                    "invoked to populate particles[0].a{x,y,z} at the IC "
                    "without taking any step."
                ),
                "ic": {"x": ic_x.tolist(), "v": ic_v.tolist(), "t_jd_tdb": ic_t},
                "subsets": [
                    {"idx": i, "label": lab, "forces": fl}
                    for i, (lab, fl) in enumerate(PEEL_SUBSETS)
                ],
            },
            indent=2,
        )
    )
    print(f"[{sim_label}] -> {npz_path}, {log_path}, {order_path}")


# ============================================================================
# main / dispatch
# ============================================================================
SUBCOMMANDS = {
    "compute_postflyby_ic": (cmd_compute_postflyby_ic, False),
    "sim1": (cmd_sim1, True),
    "sim2": (cmd_sim2, True),
    "sim3": (cmd_sim3, True),
    "sim4": (cmd_sim4, True),
    "sim5": (cmd_sim5, False),  # uses _update_accel directly
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "subcommand",
        choices=list(SUBCOMMANDS) + ["all"],
        help="Which simulation to run.",
    )
    args = parser.parse_args()

    if args.subcommand == "all":
        # Determine if any subcommand needs the dump env; if so, set it now.
        if any(needs for (_, needs) in SUBCOMMANDS.values()):
            _setup_env(needs_dump=True)
        for name in [
            "compute_postflyby_ic",
            "sim1",
            "sim2",
            "sim3",
            "sim4",
            "sim5",
        ]:
            print(f"\n{'=' * 70}\n  >>> {name}\n{'=' * 70}")
            SUBCOMMANDS[name][0]()
        return

    fn, needs_dump = SUBCOMMANDS[args.subcommand]
    _setup_env(needs_dump=needs_dump)
    fn()


if __name__ == "__main__":
    main()

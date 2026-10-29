"""Compare synchronized IAS15 primal/Jacobian timings and finite differences.

Run from the repository root, before and after a change, in separate processes::

    uv run python tools/benchmark_ias15_autodiff.py --output /tmp/before.json
    uv run python tools/benchmark_ias15_autodiff.py --freeze-controller \
        --output /tmp/frozen.json

Compilation is timed separately. Every execution waits for its result; raw samples,
environment information, and numerical comparisons are saved in JSON. No random
numbers or Horizons queries are used. Ephemeris kernels must be cached or downloaded.
"""

import argparse
import json
import platform
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import astropy.units as u
import jax.numpy as jnp
import numpy as np
from astropy.time import Time
from jax.tree_util import Partial

from jorbit import Ephemeris
from jorbit.accelerations import nongrav_acceleration
from jorbit.accelerations.create_acc_funcs import _default_ephemeris_acc
from jorbit.astrometry.sky_projection import tangent_plane_projection
from jorbit.data.constants import SPEED_OF_LIGHT
from jorbit.integrators import (
    ias15_evolve,
    initialize_ias15_integrator_state,
    next_proposed_dt_global,
)
from jorbit.particle.ias15_dense import _ephem_ias15
from jorbit.utils.states import KeplerianState, SystemState

KM_AU = 1 / 1.495978707e8
MMS_AUD = 1e-6 * 86400 * KM_AU


def _frozen_controller(*args: jnp.ndarray) -> jnp.ndarray:
    return jax.lax.stop_gradient(next_proposed_dt_global(*args))


def _column_error(actual: np.ndarray, reference: np.ndarray) -> list[float]:
    n = actual.shape[-1]
    return (
        np.linalg.norm((actual - reference).reshape(-1, n), axis=0)
        / np.linalg.norm(reference.reshape(-1, n), axis=0)
    ).tolist()


def _measure(fn: Callable, arg: jnp.ndarray, repeats: int) -> tuple:
    start = time.perf_counter()
    executable = jax.jit(fn).lower(arg).compile()
    compile_seconds = time.perf_counter() - start
    for _ in range(3):
        jax.block_until_ready(executable(arg))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(executable(arg))
        samples.append(time.perf_counter() - start)
    return executable, {
        "compile_seconds": compile_seconds,
        "median_seconds": float(np.median(samples)),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "samples_seconds": samples,
    }


def main() -> None:
    """Run the synthetic NEO workload in the currently checked-out implementation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-controller", action="store_true")
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    epoch = Time("2000-01-01", scale="tdb")
    eph = Ephemeris(
        ssos="default solar system",
        earliest_time=epoch - 30 * u.day,
        latest_time=epoch + 3682 * u.day,
    )
    kw = {"c2": SPEED_OF_LIGHT**2}
    elements = {
        "semi": 2.08,
        "ecc": 0.5,
        "inc": 8.0,
        "Omega": 60.0,
        "omega": 120.0,
        "nu": 270.0,
    }
    nominal = KeplerianState(
        **{key: jnp.array([value]) for key, value in elements.items()},
        acceleration_func_kwargs=kw,
        time_reference=jnp.array(epoch.tdb.jd),
    ).to_cartesian()
    scheduler = Partial(
        _frozen_controller if args.freeze_controller else next_proposed_dt_global
    )

    def acceleration(a2: jnp.ndarray, state: SystemState) -> jnp.ndarray:
        ng = state.replace(
            acceleration_func_kwargs={
                **kw,
                "a1": jnp.zeros(1),
                "a2": a2,
                "a3": jnp.zeros(1),
            }
        )
        return _default_ephemeris_acc(eph.processor, state) + nongrav_acceleration(ng)

    def setup(theta: jnp.ndarray) -> tuple:
        state = nominal.replace(
            x=nominal.x + theta[:3][None] * KM_AU,
            v=nominal.v + theta[3:6][None] * MMS_AUD,
        )
        acc = Partial(acceleration, theta[6:] * 1e-14)
        return state, acc, initialize_ias15_integrator_state(acc(state.to_system()))

    theta0 = jnp.zeros(7)
    results = {
        "jax": jax.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "devices": str(jax.devices()),
        "x64": jax.config.x64_enabled,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_diff": subprocess.check_output(["git", "diff", "--", "src"], text=True),
        "freeze_controller": args.freeze_controller,
        "repeats": args.repeats,
        "cases": {},
    }
    for name, days in [
        ("1yr_4times", [30.0, 100.0, 200.0, 365.0]),
        ("10yr_4times", [30.0, 365.0, 1461.0, 3652.0]),
        ("10yr_256times", np.linspace(30, 3652, 256)),
    ]:
        days = jnp.array(days)
        earth = jnp.array(
            eph.state(epoch + np.asarray(days) * u.day)["earth"]["x"].to_value(u.au)
        )

        def xyz(theta: jnp.ndarray, days: jnp.ndarray = days) -> jnp.ndarray:
            state, acc, integrator_state = setup(theta)
            return (
                ias15_evolve(state.to_system(), acc, days, integrator_state, scheduler)[
                    0
                ][:, 0]
                / KM_AU
            )

        def radec(
            theta: jnp.ndarray, days: jnp.ndarray = days, earth: jnp.ndarray = earth
        ) -> tuple:
            state, acc, integrator_state = setup(theta)
            ra, dec, covered = _ephem_ias15(
                days,
                state,
                acc,
                integrator_state,
                earth,
                jnp.arange(len(days)),
                scheduler,
            )
            return ra, dec, covered

        ra0, dec0, covered = radec(theta0)
        assert np.all(covered)

        def sky(
            theta: jnp.ndarray,
            ra0: jnp.ndarray = ra0,
            dec0: jnp.ndarray = dec0,
            radec_fn: Callable = radec,
        ) -> jnp.ndarray:
            ra, dec, _ = radec_fn(theta)
            return jax.vmap(tangent_plane_projection)(ra0, dec0, ra, dec)

        for kind, fn in [("xyz", xyz), ("sky", sky)]:
            case = f"{name}_{kind}"
            primal, primal_times = _measure(fn, theta0, args.repeats)
            jacobian, jac_times = _measure(jax.jacfwd(fn), theta0, args.repeats)
            ad = np.asarray(jacobian(theta0))
            fd = []
            for h in (1.0, 0.1):
                fd.append(
                    np.stack(
                        [
                            np.asarray((primal(h * e) - primal(-h * e)) / (2 * h))
                            for e in np.eye(7)
                        ],
                        axis=-1,
                    )
                )
            results["cases"][case] = {
                "primal": primal_times,
                "jacfwd": jac_times,
                "ad_vs_fd_h1_column_error": _column_error(ad, fd[0]),
                "fd_h01_vs_h1_column_error": _column_error(fd[1], fd[0]),
                "nominal": np.asarray(primal(theta0)).tolist(),
                "jacobian": ad.tolist(),
            }
            args.output.write_text(json.dumps(results, indent=2) + "\n")
            print(
                case,
                "primal",
                primal_times["median_seconds"],
                "jacfwd",
                jac_times["median_seconds"],
                "error",
                results["cases"][case]["ad_vs_fd_h1_column_error"],
                flush=True,
            )
        jax.clear_caches()
    state, acc, _ = setup(theta0)
    a_of_t = lambda t: acc(state.to_system().replace(relative_time=t))
    da_dt = jax.jvp(a_of_t, (jnp.array(10.0),), (jnp.array(1.0),))[1]
    results["acceleration_time_jvp"] = np.asarray(da_dt).tolist()
    results["acceleration_time_fd"] = np.asarray(
        (a_of_t(10.001) - a_of_t(9.999)) / 0.002
    ).tolist()
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()

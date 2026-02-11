"""Solar system round-trip tests for the DoubleDouble precision IAS15 integrator."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.time import Time

from jorbit.accelerations.gr_dd import _ppn_gravity_dd_core
from jorbit.data.constants import (
    SPEED_OF_LIGHT,
)
from jorbit.ephemeris.ephemeris import Ephemeris
from jorbit.integrators.iasnn_dd_prec import (
    setup_iasnn_integrator,
    step,
)
from jorbit.utils.doubledouble import DoubleDouble, dd_exp, dd_zeros


def _to_dd(x: jnp.ndarray) -> DoubleDouble:
    """Convert a plain jnp.ndarray to DoubleDouble (lo = 0)."""
    return DoubleDouble(x, jnp.zeros_like(x))


def _integrate_n_steps(x, v, b, dt, precomputed, acc_func, n_steps) -> tuple:
    """Helper to integrate forward n_steps."""
    for _ in range(n_steps):
        x, v, b = step(x, v, b, dt, precomputed, acc_func)
    return x, v, b


def _get_solar_system_state(ssos="default planets") -> tuple:
    """Get positions, velocities, and log_gms for solar system bodies at J2000.

    Args:
        ssos: "default planets" for 11 bodies, "default solar system" for 27.

    Returns:
        Tuple of (positions, velocities, log_gms) as float64 arrays.
    """
    ephem = Ephemeris(
        ssos=ssos,
        earliest_time=Time("1999-01-01"),
        latest_time=Time("2001-01-01"),
    )
    state = ephem.state(Time("2000-01-01 12:00:00", scale="tdb"))

    names = ephem._combined_names
    N = len(names)

    positions = np.zeros((N, 3))
    velocities = np.zeros((N, 3))
    log_gms = np.zeros(N)

    for i, name in enumerate(names):
        positions[i] = state[name]["x"].to("au").value
        velocities[i] = state[name]["v"].to("au/day").value
        log_gms[i] = float(state[name]["log_gm"])

    return (
        jnp.array(positions),
        jnp.array(velocities),
        jnp.array(log_gms),
    )


@pytest.fixture(scope="module")
def planets_setup() -> tuple:
    """Set up 11-planet solar system for DD PPN roundtrip tests."""
    positions, velocities, log_gms = _get_solar_system_state("default planets")
    N = positions.shape[0]

    x0_dd = _to_dd(positions)
    v0_dd = _to_dd(velocities)
    gms_dd = dd_exp(_to_dd(log_gms))
    c2_dd = DoubleDouble(jnp.float64(SPEED_OF_LIGHT**2))

    def acc_fn(x_dd, v_dd) -> DoubleDouble:
        return _ppn_gravity_dd_core(x_dd, v_dd, gms_dd, c2_dd)

    acc_func = jax.tree_util.Partial(acc_fn)
    precomputed = setup_iasnn_integrator(7)

    return x0_dd, v0_dd, N, precomputed, acc_func


@pytest.fixture(scope="module")
def full_solar_system_setup() -> tuple:
    """Set up 27-body solar system (planets + asteroids) for DD PPN roundtrip tests."""
    positions, velocities, log_gms = _get_solar_system_state("default solar system")
    N = positions.shape[0]

    x0_dd = _to_dd(positions)
    v0_dd = _to_dd(velocities)
    gms_dd = dd_exp(_to_dd(log_gms))
    c2_dd = DoubleDouble(jnp.float64(SPEED_OF_LIGHT**2))

    def acc_fn(x_dd, v_dd) -> DoubleDouble:
        return _ppn_gravity_dd_core(x_dd, v_dd, gms_dd, c2_dd)

    acc_func = jax.tree_util.Partial(acc_fn)
    precomputed = setup_iasnn_integrator(7)

    return x0_dd, v0_dd, N, precomputed, acc_func


class TestSolarSystemPPNRoundtrip:
    """Round-trip tests for the full solar system with PPN gravity in DD precision.

    Uses 11 planets (sun, mercury, venus, earth, moon, mars, jupiter, saturn,
    uranus, neptune, pluto) with DD PPN gravity.
    """

    @pytest.mark.parametrize(
        "duration_days,dt_days,tol",
        [
            (1, 0.5, 1e-18),
            (1, 1.0, 1e-18),
            (7, 0.5, 1e-16),
            (7, 1.0, 1e-16),
            (30, 0.5, 1e-14),
            (30, 1.0, 1e-14),
            (30, 4.0, 1e-14),
        ],
    )
    def test_planets_roundtrip(
        self, planets_setup, duration_days, dt_days, tol
    ) -> None:
        """Roundtrip test: integrate forward then backward, check return to start."""
        x0_dd, v0_dd, N, precomputed, acc_func = planets_setup
        n_steps = int(duration_days / dt_days)
        dt = DoubleDouble(jnp.float64(dt_days))

        # Forward integration
        b = dd_zeros((7, N, 3))
        x, v, b = _integrate_n_steps(
            x0_dd, v0_dd, b, dt, precomputed, acc_func, n_steps
        )

        # Backward integration (negate velocity, reset b)
        b = dd_zeros((7, N, 3))
        v = -v
        x, v, b = _integrate_n_steps(x, v, b, dt, precomputed, acc_func, n_steps)
        v = -v

        # Per-body position error
        dx = x - x0_dd
        per_body_err = jnp.sqrt(jnp.sum((dx.hi + dx.lo) ** 2, axis=1))  # (N,) float64
        max_err = float(jnp.max(per_body_err))
        print(max_err)

        # Roundtrip error should be very small for DD arithmetic.
        # Even the loosest tolerance here (1e-14 AU ~ 1.5 m) is many orders
        # of magnitude better than float64 integration would achieve.
        assert max_err < tol, (
            f"Roundtrip error {max_err:.2e} exceeds tolerance {tol:.2e} "
            f"for {duration_days}d @ dt={dt_days}d"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "duration_days,dt_days,tol",
        [
            (365, 1.0, 1e-11),
            (365, 4.0, 1e-11),
        ],
    )
    def test_planets_roundtrip_long(
        self, planets_setup, duration_days, dt_days, tol
    ) -> None:
        """Long-duration roundtrip tests (marked slow)."""
        x0_dd, v0_dd, N, precomputed, acc_func = planets_setup
        n_steps = int(duration_days / dt_days)
        dt = DoubleDouble(jnp.float64(dt_days))

        b = dd_zeros((7, N, 3))
        x, v, b = _integrate_n_steps(
            x0_dd, v0_dd, b, dt, precomputed, acc_func, n_steps
        )

        b = dd_zeros((7, N, 3))
        v = -v
        x, v, b = _integrate_n_steps(x, v, b, dt, precomputed, acc_func, n_steps)
        v = -v

        dx = x - x0_dd
        per_body_err = jnp.sqrt(jnp.sum((dx.hi + dx.lo) ** 2, axis=1))
        max_err = float(jnp.max(per_body_err))

        assert max_err < tol, (
            f"Roundtrip error {max_err:.2e} exceeds tolerance {tol:.2e} "
            f"for {duration_days}d @ dt={dt_days}d"
        )

    @pytest.mark.slow
    def test_planets_roundtrip_decade(self, planets_setup) -> None:
        """Decade-long roundtrip test (very slow)."""
        x0_dd, v0_dd, N, precomputed, acc_func = planets_setup
        duration_days = 3652
        dt_days = 4.0
        n_steps = int(duration_days / dt_days)
        dt = DoubleDouble(jnp.float64(dt_days))

        b = dd_zeros((7, N, 3))
        x, v, b = _integrate_n_steps(
            x0_dd, v0_dd, b, dt, precomputed, acc_func, n_steps
        )

        b = dd_zeros((7, N, 3))
        v = -v
        x, v, b = _integrate_n_steps(x, v, b, dt, precomputed, acc_func, n_steps)
        v = -v

        dx = x - x0_dd
        per_body_err = jnp.sqrt(jnp.sum((dx.hi + dx.lo) ** 2, axis=1))
        max_err = float(jnp.max(per_body_err))

        # 1e-9 AU ~ 150 m, still far below float64 integration precision
        tol = 1e-9
        assert (
            max_err < tol
        ), f"Decade roundtrip error {max_err:.2e} exceeds tolerance {tol:.2e}"

    def test_smaller_dt_gives_smaller_error(self, planets_setup) -> None:
        """Verify that smaller dt reduces roundtrip error for same total duration."""
        x0_dd, v0_dd, N, precomputed, acc_func = planets_setup
        duration_days = 7

        errors = []
        for dt_days in [1.0, 0.5]:
            n_steps = int(duration_days / dt_days)
            dt = DoubleDouble(jnp.float64(dt_days))

            b = dd_zeros((7, N, 3))
            x, v, b = _integrate_n_steps(
                x0_dd, v0_dd, b, dt, precomputed, acc_func, n_steps
            )
            b = dd_zeros((7, N, 3))
            v = -v
            x, v, b = _integrate_n_steps(x, v, b, dt, precomputed, acc_func, n_steps)

            dx = x - x0_dd
            max_err = float(jnp.max(jnp.sqrt(jnp.sum((dx.hi + dx.lo) ** 2, axis=1))))
            errors.append(max_err)

        # Smaller dt should give smaller roundtrip error
        assert errors[1] < errors[0], (
            f"Smaller dt didn't reduce error: dt=1.0 -> {errors[0]:.2e}, "
            f"dt=0.5 -> {errors[1]:.2e}"
        )


class TestSolarSystemWithAsteroidsRoundtrip:
    """Round-trip tests for the full 27-body solar system (planets + asteroids)."""

    @pytest.mark.parametrize(
        "duration_days,dt_days,tol",
        [
            (1, 0.5, 1e-18),
            (1, 1.0, 1e-18),
            (7, 1.0, 1e-16),
        ],
    )
    def test_full_system_roundtrip(
        self, full_solar_system_setup, duration_days, dt_days, tol
    ) -> None:
        """Roundtrip test with 27 bodies (11 planets + 16 asteroids)."""
        x0_dd, v0_dd, N, precomputed, acc_func = full_solar_system_setup
        n_steps = int(duration_days / dt_days)
        dt = DoubleDouble(jnp.float64(dt_days))

        b = dd_zeros((7, N, 3))
        x, v, b = _integrate_n_steps(
            x0_dd, v0_dd, b, dt, precomputed, acc_func, n_steps
        )

        b = dd_zeros((7, N, 3))
        v = -v
        x, v, b = _integrate_n_steps(x, v, b, dt, precomputed, acc_func, n_steps)
        v = -v

        dx = x - x0_dd
        per_body_err = jnp.sqrt(jnp.sum((dx.hi + dx.lo) ** 2, axis=1))
        max_err = float(jnp.max(per_body_err))

        assert max_err < tol, (
            f"Roundtrip error {max_err:.2e} exceeds tolerance {tol:.2e} "
            f"for {duration_days}d @ dt={dt_days}d (27 bodies)"
        )

    @pytest.mark.slow
    def test_full_system_roundtrip_month(self, full_solar_system_setup) -> None:
        """Month-long roundtrip with 27 bodies (slow)."""
        x0_dd, v0_dd, N, precomputed, acc_func = full_solar_system_setup
        duration_days = 30
        dt_days = 4.0
        n_steps = int(duration_days / dt_days)
        dt = DoubleDouble(jnp.float64(dt_days))

        b = dd_zeros((7, N, 3))
        x, v, b = _integrate_n_steps(
            x0_dd, v0_dd, b, dt, precomputed, acc_func, n_steps
        )

        b = dd_zeros((7, N, 3))
        v = -v
        x, v, b = _integrate_n_steps(x, v, b, dt, precomputed, acc_func, n_steps)
        v = -v

        dx = x - x0_dd
        per_body_err = jnp.sqrt(jnp.sum((dx.hi + dx.lo) ** 2, axis=1))
        max_err = float(jnp.max(per_body_err))

        # 1e-14 AU ~ 1.5 m, well below float64 precision
        tol = 1e-14
        assert (
            max_err < tol
        ), f"Month roundtrip error {max_err:.2e} exceeds tolerance {tol:.2e} (27 bodies)"

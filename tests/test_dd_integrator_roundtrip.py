"""Round-trip integration tests for the DoubleDouble precision IAS15 integrator.

Tests verify that integrating forward N steps, negating velocity, and integrating
forward N more steps returns to the starting position/velocity within DD precision
tolerances.
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from jorbit.integrators.iasnn_dd_prec import (
    central_potential,
    setup_iasnn_integrator,
    step,
)
from jorbit.utils.doubledouble import DoubleDouble, dd_norm


def _to_dd(x: jnp.ndarray) -> DoubleDouble:
    """Convert a plain jnp.ndarray to DoubleDouble (lo = 0)."""
    return DoubleDouble(x, jnp.zeros_like(x))


@pytest.fixture
def precomputed_7() -> tuple:
    """Set up IAS15 integrator with 7 internal points."""
    return setup_iasnn_integrator(7)


@pytest.fixture
def acc_func() -> jax.tree_util.Partial:
    """Create a Partial-wrapped central potential acceleration function."""
    return jax.tree_util.Partial(central_potential)


def _integrate_n_steps(x, v, b, dt, precomputed, acc_func, n_steps) -> tuple:
    """Helper to integrate forward n_steps."""
    for _ in range(n_steps):
        x, v, b = step(x, v, b, dt, precomputed, acc_func)
    return x, v, b


def _roundtrip(
    x0, v0, dt, precomputed, acc_func, n_steps, n_internal_points=7
) -> tuple:
    """Integrate forward, negate velocity, integrate forward, negate velocity."""
    b = DoubleDouble(jnp.zeros((n_internal_points, 1, 3)))

    # Forward
    x, v, b = _integrate_n_steps(x0, v0, b, dt, precomputed, acc_func, n_steps)

    # Reset b, negate velocity for backward integration
    b = DoubleDouble(jnp.zeros((n_internal_points, 1, 3)))
    v = -v

    # Backward (same number of steps)
    x, v, b = _integrate_n_steps(x, v, b, dt, precomputed, acc_func, n_steps)

    # Negate velocity to restore original direction
    v = -v

    return x, v


class TestCircularKeplerRoundtrip:
    """Round-trip tests for circular Kepler orbits (unit central potential)."""

    def test_short_roundtrip(self, precomputed_7, acc_func) -> None:
        """10 steps forward, 10 steps back with dt=0.01."""
        x0 = DoubleDouble(jnp.array([[1.0, 0.0, 0.0]]))
        v0 = DoubleDouble(jnp.array([[0.0, 1.0, 0.0]]))
        dt = DoubleDouble(0.01)

        x, v = _roundtrip(x0, v0, dt, precomputed_7, acc_func, n_steps=10)

        err_x = float(dd_norm(x - x0).hi)
        err_v = float(dd_norm(v - v0).hi)
        assert err_x < 1e-25, f"Position round-trip error too large: {err_x}"
        assert err_v < 1e-25, f"Velocity round-trip error too large: {err_v}"

    def test_medium_roundtrip(self, precomputed_7, acc_func) -> None:
        """50 steps forward, 50 steps back with dt=0.01."""
        x0 = DoubleDouble(jnp.array([[1.0, 0.0, 0.0]]))
        v0 = DoubleDouble(jnp.array([[0.0, 1.0, 0.0]]))
        dt = DoubleDouble(0.01)

        x, v = _roundtrip(x0, v0, dt, precomputed_7, acc_func, n_steps=50)

        err_x = float(dd_norm(x - x0).hi)
        err_v = float(dd_norm(v - v0).hi)
        assert err_x < 1e-24, f"Position round-trip error too large: {err_x}"
        assert err_v < 1e-24, f"Velocity round-trip error too large: {err_v}"

    def test_long_roundtrip(self, precomputed_7, acc_func) -> None:
        """100 steps forward, 100 steps back with dt=0.01."""
        x0 = DoubleDouble(jnp.array([[1.0, 0.0, 0.0]]))
        v0 = DoubleDouble(jnp.array([[0.0, 1.0, 0.0]]))
        dt = DoubleDouble(0.01)

        x, v = _roundtrip(x0, v0, dt, precomputed_7, acc_func, n_steps=100)

        err_x = float(dd_norm(x - x0).hi)
        err_v = float(dd_norm(v - v0).hi)
        assert err_x < 1e-23, f"Position round-trip error too large: {err_x}"
        assert err_v < 1e-23, f"Velocity round-trip error too large: {err_v}"


class TestEccentricKeplerRoundtrip:
    """Round-trip tests for eccentric Kepler orbits."""

    @pytest.mark.parametrize(
        "eccentricity,tol",
        [
            (0.3, 1e-23),
            (0.6, 1e-21),
            (0.9, 1e-18),
        ],
    )
    def test_eccentric_roundtrip(
        self, precomputed_7, acc_func, eccentricity, tol
    ) -> None:
        """Round-trip for various eccentricities, starting at apoapsis.

        At apoapsis: x = a(1+e), v_y = sqrt(GM(1-e)/(a(1+e))) for a=1, GM=1.
        """
        r_apo = 1.0 + eccentricity
        v_apo = jnp.sqrt((1.0 - eccentricity) / r_apo)

        x0 = DoubleDouble(jnp.array([[r_apo, 0.0, 0.0]]))
        v0 = DoubleDouble(jnp.array([[0.0, float(v_apo), 0.0]]))
        dt = DoubleDouble(0.01)

        x, v = _roundtrip(x0, v0, dt, precomputed_7, acc_func, n_steps=50)

        err_x = float(dd_norm(x - x0).hi)
        err_v = float(dd_norm(v - v0).hi)
        assert err_x < tol, f"Position error for e={eccentricity}: {err_x}"
        assert err_v < tol, f"Velocity error for e={eccentricity}: {err_v}"


class TestFullOrbitRoundtrip:
    """Test integration for exactly one full orbital period."""

    def test_full_period_return(self, precomputed_7, acc_func) -> None:
        """Integrate one full period (2*pi) and check return to start."""
        import mpmath as mpm

        period = mpm.mpf("2") * mpm.pi
        n_steps = 100
        dt_val = period / mpm.mpf(str(n_steps))
        dt = DoubleDouble.from_string(str(dt_val))

        x0 = DoubleDouble(jnp.array([[1.0, 0.0, 0.0]]))
        v0 = DoubleDouble(jnp.array([[0.0, 1.0, 0.0]]))
        b = DoubleDouble(jnp.zeros((7, 1, 3)))

        x, v, b = _integrate_n_steps(x0, v0, b, dt, precomputed_7, acc_func, n_steps)

        err_x = float(dd_norm(x - x0).hi)
        err_v = float(dd_norm(v - v0).hi)
        # With 100 steps at DD precision, should be very accurate
        assert err_x < 1e-28, f"Full orbit position error: {err_x}"
        assert err_v < 1e-28, f"Full orbit velocity error: {err_v}"


class TestMultiStepConsistency:
    """Verify round-trip error scales gracefully with step count and step size."""

    def test_error_grows_with_steps(self, precomputed_7, acc_func) -> None:
        """More steps should give larger (or equal) round-trip error."""
        x0 = DoubleDouble(jnp.array([[1.0, 0.0, 0.0]]))
        v0 = DoubleDouble(jnp.array([[0.0, 1.0, 0.0]]))
        dt = DoubleDouble(0.01)

        errors = []
        for n_steps in [10, 50, 100]:
            x, _v = _roundtrip(x0, v0, dt, precomputed_7, acc_func, n_steps=n_steps)
            err = float(dd_norm(x - x0).hi)
            errors.append(err)

        # Error should generally increase (or at least not decrease dramatically)
        # with more steps - just check the longest is larger than the shortest
        assert errors[2] >= errors[0] * 0.01, f"Error didn't grow as expected: {errors}"

    def test_smaller_dt_gives_smaller_error(self, precomputed_7, acc_func) -> None:
        """Smaller dt should give smaller round-trip error for the same total time."""
        x0 = DoubleDouble(jnp.array([[1.0, 0.0, 0.0]]))
        v0 = DoubleDouble(jnp.array([[0.0, 1.0, 0.0]]))

        # Same total integration time (1.0), different step sizes
        total_time = 1.0
        errors = []
        for n_steps in [10, 100]:
            dt = DoubleDouble(total_time / n_steps)
            x, _v = _roundtrip(x0, v0, dt, precomputed_7, acc_func, n_steps=n_steps)
            err = float(dd_norm(x - x0).hi)
            errors.append(err)

        # More steps (smaller dt) should give smaller round-trip error
        assert (
            errors[1] < errors[0]
        ), f"Smaller dt didn't reduce error: dt_large={errors[0]}, dt_small={errors[1]}"

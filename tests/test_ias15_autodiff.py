"""Physical derivatives through adaptive IAS15 with moving ephemeris perturbers."""

import jax

jax.config.update("jax_enable_x64", True)
import astropy.units as u
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.time import Time
from jax.tree_util import Partial

from jorbit import Ephemeris
from jorbit.accelerations.create_acc_funcs import _default_ephemeris_acc
from jorbit.accelerations.gr import ppn_gravity
from jorbit.data.constants import SPEED_OF_LIGHT
from jorbit.integrators import (
    ias15_evolve,
    initialize_ias15_integrator_state,
    next_proposed_dt_global,
)
from jorbit.utils.states import KeplerianState

KM_AU = 1 / 1.495978707e8
MMS_AUD = 1e-6 * 86400 * KM_AU
ARC = 180.0  # days; long enough that the 1.7.2 bug gave ~1e-3 relative errors


@pytest.fixture(scope="module")
def orbit() -> tuple:
    """Synthetic eccentric NEO; no fitted state or remote Horizons query is needed."""
    epoch = Time("2000-01-01", scale="tdb")
    eph = Ephemeris(
        ssos="default solar system",
        earliest_time=epoch - 30 * u.day,
        latest_time=epoch + (ARC + 30) * u.day,
    )
    elements = {
        "semi": 2.08,
        "ecc": 0.5,
        "inc": 8.0,
        "Omega": 60.0,
        "omega": 120.0,
        "nu": 270.0,
    }
    state = KeplerianState(
        **{key: jnp.array([value]) for key, value in elements.items()},
        acceleration_func_kwargs={"c2": SPEED_OF_LIGHT**2},
        time_reference=jnp.array(epoch.tdb.jd),
    ).to_cartesian()
    return epoch, eph, state


def test_ephemeris_acceleration_time_derivative(orbit: tuple) -> None:
    """Moving perturbers must contribute to da/dt even at fixed target x and v."""
    _, eph, state = orbit

    def acceleration(t: jnp.ndarray) -> jnp.ndarray:
        return _default_ephemeris_acc(
            eph.processor, state.to_system().replace(relative_time=t)
        )

    actual = jax.jvp(acceleration, (jnp.array(10.0),), (jnp.array(1.0),))[1]
    fd = (acceleration(10.01) - acceleration(9.99)) / 0.02
    assert np.linalg.norm(fd) > 1e-10
    np.testing.assert_allclose(actual, fd, rtol=2e-6, atol=1e-15)


def test_ppn_perturber_velocity_derivative(orbit: tuple) -> None:
    """Check the smaller PPN velocity term independently of Newtonian gravity."""
    _, eph, state = orbit
    x, v = eph.processor.state(state.time_reference + 10.0)
    system = state.to_system().replace(
        fixed_perturber_positions=x[:11],
        fixed_perturber_velocities=v[:11],
        fixed_perturber_log_gms=eph.processor.log_gms[:11],
        acceleration_func_kwargs={"c2": 100.0},
    )

    def acceleration(offset: jnp.ndarray) -> jnp.ndarray:
        # A reduced c^2 makes the relativistic velocity derivative measurable
        # without subtracting two nearly identical Newtonian accelerations.
        return ppn_gravity(
            system.replace(fixed_perturber_velocities=v[:11].at[0, 0].add(offset))
        )

    actual = jax.jvp(acceleration, (jnp.array(0.0),), (jnp.array(1.0),))[1]
    fd = (acceleration(1e-4) - acceleration(-1e-4)) / 2e-4
    assert np.linalg.norm(fd) > 1e-10
    np.testing.assert_allclose(actual, fd, rtol=2e-5, atol=1e-14)


def test_long_arc_jacobian(orbit: tuple) -> None:
    """A forward-mode directional derivative matches a central finite difference.

    The direction mixes 1 km position and 1 mm/s velocity offsets. In 1.7.2 the
    adaptive controller's state-dependent proposals moved the substep times while
    the perturbers were frozen, giving relative errors of ~1e-3 here; now ~1e-9.
    """
    _, eph, nominal = orbit
    acc = Partial(_default_ephemeris_acc, eph.processor)

    @jax.jit
    def model(theta: jnp.ndarray) -> jnp.ndarray:
        state = nominal.replace(
            x=nominal.x + theta[:3][None] * KM_AU,
            v=nominal.v + theta[3:][None] * MMS_AUD,
        ).to_system()
        x, v, _, _, _ = ias15_evolve(
            state,
            acc,
            jnp.array([ARC]),
            initialize_ias15_integrator_state(acc(state)),
            Partial(next_proposed_dt_global),
        )
        return jnp.concatenate([x[0, 0] / KM_AU, v[0, 0] / MMS_AUD])

    direction = jnp.ones(6) / jnp.sqrt(6.0)
    ad = jax.jvp(model, (jnp.zeros(6),), (direction,))[1]
    h = 10.0
    fd = (model(h * direction) - model(-h * direction)) / (2 * h)
    assert np.linalg.norm(ad - fd) / np.linalg.norm(fd) < 1e-6
    jax.clear_caches()

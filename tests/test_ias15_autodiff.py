"""Physical derivatives through adaptive IAS15 with moving ephemeris perturbers."""

from collections.abc import Callable

import jax

jax.config.update("jax_enable_x64", True)
import astropy.units as u
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.time import Time
from jax.tree_util import Partial

from jorbit import Ephemeris
from jorbit.accelerations import nongrav_acceleration
from jorbit.accelerations.create_acc_funcs import (
    _default_ephemeris_acc,
    _gr_ephemeris_acc,
    _newtonian_ephemeris_acc,
)
from jorbit.accelerations.gr import ppn_gravity
from jorbit.astrometry.sky_projection import tangent_plane_projection
from jorbit.data.constants import SPEED_OF_LIGHT
from jorbit.integrators import (
    ias15_evolve,
    ias15_evolve_forced_landing,
    initialize_ias15_integrator_state,
    next_proposed_dt_global,
    next_proposed_dt_PRS23,
)
from jorbit.particle.ias15_dense import _ephem_ias15, _ephem_ias15_with_cov
from jorbit.utils.states import KeplerianState, SystemState

KM_AU = 1 / 1.495978707e8
MMS_AUD = 1e-6 * 86400 * KM_AU
DAYS = jnp.array([30.0, 365.0, 1461.0, 3652.0])


@pytest.fixture(scope="module")
def orbit() -> tuple:
    """Synthetic eccentric NEO; no fitted state or remote Horizons query is needed."""
    epoch = Time("2000-01-01", scale="tdb")
    eph = Ephemeris(
        ssos="default solar system",
        earliest_time=epoch - 3682 * u.day,
        latest_time=epoch + 3682 * u.day,
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


@pytest.mark.parametrize(
    "gravity", [_newtonian_ephemeris_acc, _gr_ephemeris_acc, _default_ephemeris_acc]
)
def test_ephemeris_acceleration_time_derivative(
    orbit: tuple, gravity: Callable
) -> None:
    """Moving perturbers must contribute to da/dt even at fixed target x and v."""
    _, eph, state = orbit

    def acceleration(t: jnp.ndarray) -> jnp.ndarray:
        return gravity(eph.processor, state.to_system().replace(relative_time=t))

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


def _column_error(actual: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.linalg.norm((actual - reference).reshape(-1, 7), axis=0) / np.linalg.norm(
        reference.reshape(-1, 7), axis=0
    )


@pytest.mark.parametrize(
    ("mode", "controller", "direction"),
    [
        ("position", next_proposed_dt_global, 1),
        ("position", next_proposed_dt_PRS23, 1),
        ("position", next_proposed_dt_global, -1),
        ("forced", next_proposed_dt_global, 1),
        ("sky", next_proposed_dt_global, 1),
    ],
)
def test_multiyear_jacobian(
    orbit: tuple, mode: str, controller: Callable, direction: int
) -> None:
    """All seven physical sensitivities agree with converged finite differences.

    Parameters are position offsets in km, velocity offsets in mm/s, and transverse
    nongravitational acceleration A2 in 1e-14 au/day^2. Column norms include all four
    output epochs and coordinates. Two FD steps distinguish AD errors from roundoff.
    """
    epoch, eph, nominal = orbit
    days = DAYS * direction
    scheduler = Partial(controller)

    def acceleration(a2: jnp.ndarray, state: SystemState) -> jnp.ndarray:
        ng = state.replace(
            acceleration_func_kwargs={
                **nominal.acceleration_func_kwargs,
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
    state, acc, integrator_state = setup(theta0)
    if mode == "sky":
        earth = jnp.array(
            eph.state(epoch + np.asarray(days) * u.day)["earth"]["x"].to_value(u.au)
        )
        ra0, dec0, covered = _ephem_ias15(
            days, state, acc, integrator_state, earth, jnp.arange(4), scheduler
        )
        assert np.all(covered)

    @jax.jit
    def model(theta: jnp.ndarray) -> jnp.ndarray:
        state, acc, integrator_state = setup(theta)
        if mode == "sky":
            ra, dec, _ = _ephem_ias15(
                days, state, acc, integrator_state, earth, jnp.arange(4), scheduler
            )
            return jax.vmap(tangent_plane_projection)(ra0, dec0, ra, dec)
        evolve = ias15_evolve_forced_landing if mode == "forced" else ias15_evolve
        x, v, _, _, _steps = evolve(
            state.to_system(), acc, days, integrator_state, scheduler
        )
        return jnp.concatenate([x[:, 0] / KM_AU, v[:, 0] / MMS_AUD], axis=-1)

    ad = np.asarray(jax.jit(jax.jacfwd(model))(theta0))
    fds = [
        np.stack(
            [np.asarray((model(h * e) - model(-h * e)) / (2 * h)) for e in np.eye(7)],
            axis=-1,
        )
        for h in (10.0, 1.0)
    ]
    components = [slice(None)] if mode == "sky" else [slice(0, 3), slice(3, 6)]
    for component in components:
        assert np.all(_column_error(fds[0][:, component], fds[1][:, component]) < 2e-4)
        assert np.all(_column_error(ad[:, component], fds[0][:, component]) < 2e-4)

    if mode == "sky":
        # Exercise the public covariance helper's six-state Jacobian separately.
        # A diagonal initial covariance corresponds to 1 km and 1 mm/s errors.
        cov = jnp.diag(jnp.array([KM_AU**2] * 3 + [MMS_AUD**2] * 3))
        _, _, propagated, covered = _ephem_ias15_with_cov(
            days, state, acc, earth, jnp.arange(4), scheduler, cov
        )
        assert np.all(covered)
        # The covariance helper uses RA, Dec; model uses RA*cos(Dec), Dec.
        jac = fds[0][..., :6].copy()
        jac[:, 0] /= np.cos(np.asarray(dec0))[:, None]
        expected = np.einsum("nik,njk->nij", jac, jac)
        np.testing.assert_allclose(propagated, expected, rtol=3e-4, atol=1e-12)
    jax.clear_caches()

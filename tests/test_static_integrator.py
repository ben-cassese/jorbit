"""Test the static version of the ias15 integrator."""

import jax

jax.config.update("jax_enable_x64", True)
import astropy.units as u
import jax.numpy as jnp
from astropy.time import Time

from jorbit import Ephemeris, Particle
from jorbit.accelerations import (
    create_default_ephemeris_acceleration_func,
    create_static_default_acceleration_func,
)
from jorbit.accelerations.gr import precompute_perturber_ppn
from jorbit.accelerations.static_helpers import (
    get_natural_dynamic_dts,
    precompute_perturber_positions,
)
from jorbit.astrometry.sky_projection import on_sky, sky_sep
from jorbit.data.constants import SPEED_OF_LIGHT
from jorbit.integrators import (
    ias15_evolve_forced_landing,
    ias15_static_evolve,
    initialize_ias15_integrator_state,
    interpolate_from_dense_output,
    next_proposed_dt_PRS23,
    precompute_interpolation_indices,
)
from jorbit.utils.horizons import get_observer_positions

eph = Ephemeris(ssos="default solar system")
# t_ref_jd=0 means this func expects absolute JDs in state.time; used only by
# _test_agreement's on_sky calls where we pass absolute JD obs times.
acc_func_dynamic_abs = create_default_ephemeris_acceleration_func(eph.processor)
acc_func_static = create_static_default_acceleration_func()


def _get_dynamic_positions(asteroid: str, times: Time) -> jnp.ndarray:
    p = Particle.from_horizons(asteroid, times[0])
    state = p.keplerian_state.to_system()
    acc_func_dynamic = create_default_ephemeris_acceleration_func(
        eph.processor, t_ref_jd=p._t_ref_jd
    )
    offsets = p._times_to_offsets(times)

    a0_dynamic = acc_func_dynamic(state)
    integrator_init = initialize_ias15_integrator_state(a0_dynamic)
    integrator_init.dt = jnp.diff(offsets)[0]

    dynamic_x, dynamic_v, _dynamic_state, _dynamic_integrator_state, _steps = (
        ias15_evolve_forced_landing(
            initial_system_state=state,
            acceleration_func=acc_func_dynamic,
            times=offsets,
            initial_integrator_state=integrator_init,
            step_scheduler=jax.tree_util.Partial(next_proposed_dt_PRS23),
        )
    )

    return dynamic_x, dynamic_v


def _get_static_positions(asteroid: str, times: Time) -> jnp.ndarray:
    p = Particle.from_horizons(asteroid, times[0])
    state = p.keplerian_state.to_system()
    acc_func_dynamic = create_default_ephemeris_acceleration_func(
        eph.processor, t_ref_jd=p._t_ref_jd
    )
    offsets = p._times_to_offsets(times)
    t0 = state.time

    a0_dynamic = acc_func_dynamic(state)
    integrator_init = initialize_ias15_integrator_state(a0_dynamic)
    integrator_init.dt = jnp.diff(offsets)[0]

    # Use natural adaptive steps (no forced landings on observation times)
    dts = get_natural_dynamic_dts(
        initial_system_state=state,
        acceleration_func=acc_func_dynamic,
        final_time=offsets[-1],
        initial_integrator_state=integrator_init,
        step_scheduler=jax.tree_util.Partial(next_proposed_dt_PRS23),
    )

    t_step_starts = t0 + jnp.concatenate([jnp.array([0.0]), jnp.cumsum(dts[:-1])])

    planet_pos, planet_vel, asteroid_pos, asteroid_vel, gms = (
        precompute_perturber_positions(
            t0=p._t_ref_astropy, dts=dts, de_ephemeris_version="440"
        )
    )
    perturber_pos = jnp.concatenate((planet_pos, asteroid_pos), axis=2)
    perturber_vel = jnp.concatenate((planet_vel, asteroid_vel), axis=2)

    # Pre-compute perturber-perturber PPN quantities
    planet_gms_linear = jnp.exp(gms[:11])
    pp_a2, pp_a_newt, pp_a_gr = jax.vmap(
        jax.vmap(precompute_perturber_ppn, in_axes=(0, 0, None, None)),
        in_axes=(0, 0, None, None),
    )(planet_pos, planet_vel, planet_gms_linear, SPEED_OF_LIGHT**2)

    state.fixed_perturber_positions = perturber_pos[0, 0]
    state.fixed_perturber_velocities = perturber_vel[0, 0]
    state.fixed_perturber_log_gms = gms
    state.acceleration_func_kwargs = {
        **state.acceleration_func_kwargs,
        "pp_a2": pp_a2[0, 0],
        "pp_a_newt": pp_a_newt[0, 0],
        "pp_a_gr": pp_a_gr[0, 0],
    }

    a0_static = acc_func_static(state)
    integrator_init = initialize_ias15_integrator_state(a0_static)
    integrator_init.dt = dts[0]

    (b_all, a0_all, x0_all, v0_all), _static_state, _static_integrator_state = (
        ias15_static_evolve(
            initial_system_state=state,
            acceleration_func=acc_func_static,
            dts=dts,
            initial_integrator_state=integrator_init,
            perturber_positions=perturber_pos,
            perturber_velocities=perturber_vel,
            perturber_log_gms=gms,
            pp_a2=pp_a2,
            pp_a_newt=pp_a_newt,
            pp_a_gr=pp_a_gr,
        )
    )

    # Interpolate at observation times (in the same rebased offset frame)
    step_indices, h_values = precompute_interpolation_indices(
        t_step_starts, dts, offsets
    )
    interp_x, interp_v = interpolate_from_dense_output(
        b_all, a0_all, x0_all, v0_all, dts, step_indices, h_values
    )
    return interp_x, interp_v


def _test_agreement(asteroid: str, times: Time) -> None:
    dynamic_x, dynamic_v = _get_dynamic_positions(asteroid, times)
    static_x, static_v = _get_static_positions(asteroid, times)

    pos_diff = jnp.linalg.norm(static_x[:, 0] - dynamic_x[:, 0], axis=1)
    assert jnp.max(pos_diff) * u.au.to(u.m) < 500.0

    obspos = get_observer_positions(
        observatories="kitt peak",
        times=times,
        de_ephemeris_version="440",
    )

    dynamic_ra, dynamic_dec = jax.vmap(on_sky, in_axes=(0, 0, 0, 0, None))(
        dynamic_x[:, 0],
        dynamic_v[:, 0],
        times.tdb.jd,
        obspos,
        acc_func_dynamic_abs,
    )

    static_ra, static_dec = jax.vmap(on_sky, in_axes=(0, 0, 0, 0, None))(
        static_x[:, 0], static_v[:, 0], times.tdb.jd, obspos, acc_func_dynamic_abs
    )

    ang_diff = jax.vmap(sky_sep)(
        dynamic_ra, dynamic_dec, static_ra, static_dec
    ) * u.arcsec.to(u.mas)
    assert jnp.max(ang_diff) < 0.2


def test_static_integrator() -> None:
    """Test whether the static integrator agrees with the dynamic one.

    'Agreement' here means within 500 m and 0.2 mas for time spans of up to 10 years.
    Most of test cases do much better than this, but for high-precision work, probably
    stick with the dynamic integrator or DoubleDouble precision.

    """
    # n_tests = 100
    n_tests = 10  # worked for 100 but took ~x min locally
    n_subtimes = 20

    asteroids = jax.random.randint(jax.random.PRNGKey(0), (n_tests,), 1, 500_000)
    asteroids = [str(a) for a in asteroids]

    t0s = (
        Time("2026-01-01")
        + jax.random.uniform(
            jax.random.PRNGKey(1), (n_tests,), jnp.float64, -365 * 10, 365 * 10
        )
        * u.day
    )
    max_time_spans = jax.random.uniform(
        jax.random.PRNGKey(2), (n_tests,), jnp.float64, 1, 3650
    )

    times = []
    for i, t in enumerate(t0s):
        rands = jax.random.uniform(
            jax.random.PRNGKey(3 + i), (n_subtimes,), jnp.float64, 0, 1
        )
        rands = jnp.sort(rands)
        subtimes = t + (rands * max_time_spans[i]) * u.day
        times.append(subtimes)

    for i in range(n_tests):
        _test_agreement(asteroids[i], times[i])

"""Test the dense-output / interpolation variant of the dynamic IAS15 evolve.

Two comparisons:
  1. New ias15_evolve (interpolation) vs _ias15_evolve_forced_landing (forced
     landing, same live-ephemeris acceleration function). Tight tolerance since
     both paths share the accelerations.
  2. New ias15_evolve vs the existing static-path interpolation via
     ias15_static_evolve + Chebyshev-fit perturbers. Looser tolerance because
     the static path has its own perturber-fit error floor.

Plus a jacfwd smoke test to confirm forward-mode AD still works through the
while_loop-based dense-output integrator.
"""

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
    ias15_evolve,
    ias15_static_evolve,
    initialize_ias15_integrator_state,
    interpolate_from_dense_output,
    precompute_interpolation_indices,
)
from jorbit.integrators.ias15 import _ias15_evolve_forced_landing
from jorbit.utils.horizons import get_observer_positions

eph = Ephemeris(ssos="default solar system")
acc_func_dynamic = create_default_ephemeris_acceleration_func(eph.processor)
acc_func_static = create_static_default_acceleration_func()


def _get_forced_landing_positions(asteroid: str, times: Time) -> tuple:
    p = Particle.from_horizons(asteroid, times[0])
    state = p.keplerian_state.to_system()

    a0 = acc_func_dynamic(state)
    integrator_init = initialize_ias15_integrator_state(a0)
    integrator_init.dt = jnp.diff(times.tdb.jd)[0]

    x, v, _, _ = _ias15_evolve_forced_landing(
        initial_system_state=state,
        acceleration_func=acc_func_dynamic,
        times=times.tdb.jd,
        initial_integrator_state=integrator_init,
    )
    return x, v


def _get_dynamic_interp_positions(asteroid: str, times: Time) -> tuple:
    p = Particle.from_horizons(asteroid, times[0])
    state = p.keplerian_state.to_system()

    a0 = acc_func_dynamic(state)
    integrator_init = initialize_ias15_integrator_state(a0)
    integrator_init.dt = jnp.diff(times.tdb.jd)[0]

    x, v, _, _ = ias15_evolve(
        initial_system_state=state,
        acceleration_func=acc_func_dynamic,
        times=times.tdb.jd,
        initial_integrator_state=integrator_init,
    )
    return x, v


def _get_static_interp_positions(asteroid: str, times: Time) -> tuple:
    p = Particle.from_horizons(asteroid, times[0])
    state = p.keplerian_state.to_system()
    t0 = times[0].tdb.jd

    a0_dynamic = acc_func_dynamic(state)
    integrator_init = initialize_ias15_integrator_state(a0_dynamic)
    integrator_init.dt = jnp.diff(times.tdb.jd)[0]

    dts = get_natural_dynamic_dts(
        initial_system_state=state,
        acceleration_func=acc_func_dynamic,
        final_time=times[-1].tdb.jd,
        initial_integrator_state=integrator_init,
    )
    t_step_starts = t0 + jnp.concatenate([jnp.array([0.0]), jnp.cumsum(dts[:-1])])

    planet_pos, planet_vel, asteroid_pos, asteroid_vel, gms = (
        precompute_perturber_positions(t0=times[0], dts=dts, de_ephemeris_version="440")
    )
    perturber_pos = jnp.concatenate((planet_pos, asteroid_pos), axis=2)
    perturber_vel = jnp.concatenate((planet_vel, asteroid_vel), axis=2)

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
    integrator_init_static = initialize_ias15_integrator_state(a0_static)
    integrator_init_static.dt = dts[0]

    (b_all, a0_all, x0_all, v0_all), _, _ = ias15_static_evolve(
        initial_system_state=state,
        acceleration_func=acc_func_static,
        dts=dts,
        initial_integrator_state=integrator_init_static,
        perturber_positions=perturber_pos,
        perturber_velocities=perturber_vel,
        perturber_log_gms=gms,
        pp_a2=pp_a2,
        pp_a_newt=pp_a_newt,
        pp_a_gr=pp_a_gr,
    )
    step_indices, h_values = precompute_interpolation_indices(
        t_step_starts, dts, times.tdb.jd
    )
    return interpolate_from_dense_output(
        b_all, a0_all, x0_all, v0_all, dts, step_indices, h_values
    )


def _check_sky_agreement(
    x_a: jnp.ndarray,
    v_a: jnp.ndarray,
    x_b: jnp.ndarray,
    v_b: jnp.ndarray,
    times: Time,
    pos_tol_m: float,
    sky_tol_mas: float,
) -> None:
    pos_diff = jnp.linalg.norm(x_a[:, 0] - x_b[:, 0], axis=1)
    assert jnp.max(pos_diff) * u.au.to(u.m) < pos_tol_m

    obspos = get_observer_positions(
        observatories="kitt peak", times=times, de_ephemeris_version="440"
    )
    ra_a, dec_a = jax.vmap(on_sky, in_axes=(0, 0, 0, 0, None))(
        x_a[:, 0], v_a[:, 0], times.tdb.jd, obspos, acc_func_dynamic
    )
    ra_b, dec_b = jax.vmap(on_sky, in_axes=(0, 0, 0, 0, None))(
        x_b[:, 0], v_b[:, 0], times.tdb.jd, obspos, acc_func_dynamic
    )
    ang_diff = jax.vmap(sky_sep)(ra_a, dec_a, ra_b, dec_b) * u.arcsec.to(u.mas)
    assert jnp.max(ang_diff) < sky_tol_mas


def _random_test_cases(n_tests: int, n_subtimes: int) -> tuple:
    """Same harness as test_static_integrator: random asteroids, random time spans."""
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

    times_list = []
    for i, t in enumerate(t0s):
        rands = jax.random.uniform(
            jax.random.PRNGKey(3 + i), (n_subtimes,), jnp.float64, 0, 1
        )
        rands = jnp.sort(rands)
        subtimes = t + (rands * max_time_spans[i]) * u.day
        times_list.append(subtimes)
    return asteroids, times_list


# def test_dynamic_interp_matches_forced_landing() -> None:
#     """

#     Interp vs forced-landing share the same acceleration function, so the only
#     source of difference is the polynomial-evaluation and slight adaptive-step
#     divergence. Threshold is tight: 10 m / 10 microarcseconds.
#     """
#     n_tests = 10
#     n_subtimes = 20
#     asteroids, times_list = _random_test_cases(n_tests, n_subtimes)

#     for i in range(n_tests):
#         forced_x, forced_v = _get_forced_landing_positions(asteroids[i], times_list[i])
#         interp_x, interp_v = _get_dynamic_interp_positions(asteroids[i], times_list[i])
#         _check_sky_agreement(
#             interp_x,
#             interp_v,
#             forced_x,
#             forced_v,
#             times_list[i],
#             pos_tol_m=10.0,
#             sky_tol_mas=0.01,
#         )


# def test_dynamic_interp_matches_static_interp() -> None:
#     """Dynamic-interp (live ephemeris) vs static-interp (Chebyshev-fit perturbers).
#     Looser threshold: 500 m / 0.2 mas, matching the existing static-vs-dynamic test.
#     """
#     n_tests = 10
#     n_subtimes = 20
#     asteroids, times_list = _random_test_cases(n_tests, n_subtimes)

#     for i in range(n_tests):
#         interp_x, interp_v = _get_dynamic_interp_positions(asteroids[i], times_list[i])
#         static_x, static_v = _get_static_interp_positions(asteroids[i], times_list[i])
#         _check_sky_agreement(
#             interp_x,
#             interp_v,
#             static_x,
#             static_v,
#             times_list[i],
#             pos_tol_m=500.0,
#             sky_tol_mas=0.2,
#         )


# def test_dynamic_interp_jacfwd_runs() -> None:
#     """Smoke test: jax.jacfwd through the new ias15_evolve produces a finite
#     Jacobian of the expected shape.
#     """
#     p = Particle.from_horizons("274301", Time("2022-01-14"))
#     state = p.keplerian_state.to_system()
#     t0 = state.time
#     a0 = acc_func_dynamic(state)
#     integrator_init = initialize_ias15_integrator_state(a0)
#     integrator_init.dt = 10.0

#     times = jnp.array([t0 + 10.0, t0 + 30.0, t0 + 100.0])

#     def f(cartesian_state):
#         ss = cartesian_state.to_system()
#         ig = initialize_ias15_integrator_state(acc_func_dynamic(ss))
#         ig.dt = 10.0
#         positions, _, _, _ = ias15_evolve(ss, acc_func_dynamic, times, ig)
#         return positions[:, 0, :]

#     jac = jax.jacfwd(f)(p.cartesian_state)
#     # Output is positions[:, 0, :] shape (3 times, 3 coords); CartesianState.x
#     # shape is (1, 3), so jac.x shape is output_shape + input_shape = (3, 3, 1, 3).
#     assert jnp.all(jnp.isfinite(jac.x))
#     assert jnp.all(jnp.isfinite(jac.v))
#     assert jac.x.shape == (3, 3, 1, 3)
#     assert jac.v.shape == (3, 3, 1, 3)

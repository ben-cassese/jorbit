"""Helper functions to set up rapid likelihood evaluations using cached pertubers."""
from __future__ import annotations
import jax

jax.config.update("jax_enable_x64", True)

import astropy.units as u
import jax.numpy as jnp
from astropy.time import Time

from jorbit import Ephemeris
from jorbit.accelerations import (
    create_default_ephemeris_acceleration_func,
    create_static_default_acceleration_func,
    create_static_default_on_sky_acc_func,
)
from jorbit.accelerations.static_helpers import (
    generate_perturber_chebyshev_coeffs,
    get_all_dynamic_intermediate_dts,
    precompute_perturber_positions,
)
from jorbit.astrometry.sky_projection import on_sky, tangent_plane_projection
from jorbit.integrators import ias15_static_evolve, initialize_ias15_integrator_state
from jorbit.utils.states import CartesianState, KeplerianState


def precompute_likelihood_data(p: "Particle") -> tuple:  # noqa: F821
    """Given a nearly correct orbit, precompute needed data for further fast likelihood evaluations.

    Args:
        p (Particle):
            A Particle object with an associated Observations object. The particle's
            state should be approximately correct in the sense that the same "optimal"
            time steps taken by the dynamic IAS15 integrator can be reused for slightly
            different initial conditions.

    Returns:
        tuple:
            A tuple containing the following precomputed data:
            - cheby_info: A dictionary containing Chebyshev coefficients for perturber
            positions and velocities.
            - dts: A JAX array of time steps for the dynamic integrator.
            - inds: A JAX array of indices corresponding to observation times in the
            dynamic integration.
            - perturber_pos: A JAX array of precomputed perturber positions at each
            intermediate step (and each IAS15 substep).
            - perturber_vel: A JAX array of precomputed perturber velocities at
            each intermediate step (and each IAS15 substep).
            - gms: A JAX array of gravitational parameters for the perturbers.
            - observer_positions: A JAX array of observer positions at each observation
            time.
            - times: An Astropy Time object containing the times of all intermediate
            steps between observations.
            - obs_ras: A JAX array of observed right ascensions.
            - obs_decs: A JAX array of observed declinations.
    """
    obs_times = p.observations.times

    ephem = Ephemeris(
        earliest_time=Time(jnp.min(obs_times), format="jd", scale="tdb") - 10 * u.day,
        latest_time=Time(jnp.max(obs_times), format="jd", scale="tdb") + 10 * u.day,
        ssos="default solar system",
        de_ephemeris_version="440",
    )

    # precompute Chebyshev coefficients to compute the states of the perturbers
    # around each observation time for light travel time corrections
    perturber_pos_chebys, perturber_vel_chebys = [], []
    for t in obs_times:
        pos, vel = generate_perturber_chebyshev_coeffs(
            obs_time=Time(t, format="jd", scale="tdb"),
            ephem=ephem,
        )
        perturber_pos_chebys.append(pos)
        perturber_vel_chebys.append(vel)
    perturber_pos_chebys = jnp.array(perturber_pos_chebys)
    perturber_vel_chebys = jnp.array(perturber_vel_chebys)

    # constants chosen to match
    # jorbit.accelerations.static_helpers.generate_perturber_chebyshev_coeffs
    # which in turn were chosen to allow for light travel time out to ~1000 au
    # (the +2 is chosen to keep the observation times away from the edges of the
    # Chebyshev interval, where the approximation is worst)
    cheby_info = {
        "perturber_position_cheby_coeffs": perturber_pos_chebys,
        "perturber_velocity_cheby_coeffs": perturber_vel_chebys,
        "cheby_t0": obs_times - 6.0,
        "cheby_t1": obs_times + 2.0,
    }

    # get the times of all the intermediate steps between observations
    times = Time(
        jnp.concatenate([jnp.array([p.keplerian_state.time]), obs_times]),
        format="jd",
        scale="tdb",
    )
    state = p.keplerian_state.to_system()
    dynamic_acc_func = create_default_ephemeris_acceleration_func(ephem.processor)
    a0_dynamic = dynamic_acc_func(state)
    integrator_init = initialize_ias15_integrator_state(a0_dynamic)
    integrator_init.dt = jnp.diff(times.tdb.jd)[0]
    dts, inds = get_all_dynamic_intermediate_dts(
        initial_system_state=state,
        acceleration_func=dynamic_acc_func,
        times=times,
        initial_integrator_state=integrator_init,
    )

    # precompute the positions and velocities of all perturbers at each intermediate
    # step, and at each ias15 substep
    planet_pos, planet_vel, asteroid_pos, asteroid_vel, gms = (
        precompute_perturber_positions(
            t0=Time(p.cartesian_state.time, format="jd", scale="tdb"),
            dts=dts,
            de_ephemeris_version="440",
        )
    )
    perturber_pos = jnp.concatenate((planet_pos, asteroid_pos), axis=2)
    perturber_vel = jnp.concatenate((planet_vel, asteroid_vel), axis=2)

    observer_positions = p.observations.observer_positions

    return (
        cheby_info,
        dts,
        inds,
        perturber_pos,
        perturber_vel,
        gms,
        observer_positions,
        times,
        p.observations.ra,
        p.observations.dec,
    )


def create_default_static_residuals_func(inputs: tuple) -> jax.tree_util.Partial:
    """Create a function to compute residuals using static integration and precomputed perturber positions.

    The returned function uses the "default" dynamical model, meaning GR effects for the
    Sun+planets, Newtonian gravity for the asteroids, and no non-gravitational
    or gravitational harmonic effects.

    Args:
        inputs (tuple):
            The outputs of the `precompute_likelihood_data` function, containing all
            necessary precomputed data for the static integration.

    Returns:
        jax.tree_util.Partial:
            A JIT-compiled function that takes a system state as input and returns the
            residuals between the observed and model right ascensions and declinations
            projected onto the tangent plane.
    """
    (
        cheby_info,
        dts,
        inds,
        perturber_pos,
        perturber_vel,
        log_gms,
        observer_positions,
        obs_times,
        obs_ras,
        obs_decs,
    ) = inputs

    static_acc_func = create_static_default_acceleration_func()
    on_sky_acc_func = create_static_default_on_sky_acc_func()

    times = obs_times[1:].tdb.jd

    def static_residuals_func(state: CartesianState | KeplerianState) -> jnp.ndarray:
        state = state.to_system()
        state.fixed_perturber_positions = perturber_pos[0, 0]
        state.fixed_perturber_velocities = perturber_vel[0, 0]
        state.fixed_perturber_log_gms = log_gms
        a0 = static_acc_func(state)
        integrator_init = initialize_ias15_integrator_state(a0)
        integrator_init.dt = dts[0]

        static_x, static_v, _static_state, _static_integrator_state = (
            ias15_static_evolve(
                initial_system_state=state,
                acceleration_func=static_acc_func,
                dts=dts,
                initial_integrator_state=integrator_init,
                perturber_positions=perturber_pos,
                perturber_velocities=perturber_vel,
                perturber_log_gms=log_gms,
            )
        )

        obs_xs = static_x[inds, 0, :][
            1:
        ]  # cut out the initial state, which we tacked on for a0
        obs_vs = static_v[inds, 0, :][1:]

        model_ras, model_decs = jax.vmap(on_sky, in_axes=(0, 0, 0, 0, None, 0))(
            obs_xs,
            obs_vs,
            times,
            observer_positions,
            on_sky_acc_func,
            cheby_info,
        )

        xis_etas = jax.vmap(tangent_plane_projection)(
            obs_ras, obs_decs, model_ras, model_decs
        )

        return xis_etas

    return jax.jit(jax.tree_util.Partial(static_residuals_func))

"""Helper functions to set up rapid likelihood evaluations using cached pertubers."""

from collections.abc import Callable

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
from jorbit.accelerations.gr import precompute_perturber_ppn
from jorbit.accelerations.static_helpers import (
    generate_perturber_chebyshev_coeffs,
    get_natural_dynamic_dts,
    precompute_perturber_positions,
)
from jorbit.astrometry.sky_projection import on_sky, tangent_plane_projection
from jorbit.data.constants import SPEED_OF_LIGHT
from jorbit.integrators import (
    ias15_static_evolve,
    initialize_ias15_integrator_state,
    precompute_interpolation_indices,
)
from jorbit.integrators.ias15 import make_ltt_propagator
from jorbit.utils.states import CartesianState, KeplerianState


def precompute_likelihood_data(
    p: "Particle", step_scheduler: Callable  # noqa: F821
) -> tuple:
    """Given a nearly correct orbit, precompute needed data for further fast likelihood evaluations.

    Args:
        p (Particle):
            A Particle object with an associated Observations object. The particle's
            state should be approximately correct in the sense that the same "optimal"
            time steps taken by the dynamic IAS15 integrator can be reused for slightly
            different initial conditions.
        step_scheduler (Callable):
            The step scheduler used by the underlying ``ias15_step`` to choose the
            next proposed step size when computing the natural step sizes.

    Returns:
        tuple:
            A tuple containing the following precomputed data:
            - cheby_info: A dictionary containing Chebyshev coefficients for perturber
            positions and velocities.
            - dts: A JAX array of time steps for the dynamic integrator.
            - t_step_starts: A JAX array of the start time of each integration step,
            used with searchsorted for interpolation.
            - perturber_pos: A JAX array of precomputed perturber positions at each
            intermediate step (and each IAS15 substep).
            - perturber_vel: A JAX array of precomputed perturber velocities at
            each intermediate step (and each IAS15 substep).
            - gms: A JAX array of gravitational parameters for the perturbers.
            - observer_positions: A JAX array of observer positions at each observation
            time.
            - obs_times: A JAX array of observation times (JD TDB).
            - obs_ras: A JAX array of observed right ascensions.
            - obs_decs: A JAX array of observed declinations.
    """
    # Absolute-JD astropy times: used to size the Ephemeris window and to generate
    # Chebyshev coefficients (both inherently operate in absolute JD).
    obs_times_astropy = p.observations.times_astropy
    if obs_times_astropy is None:
        obs_times_astropy = Time(p.observations.times, format="jd", scale="tdb")
    # Offsets from the particle's t_ref: used everywhere JAX touches time.
    obs_times = p._times_to_offsets(obs_times_astropy)

    ephem = Ephemeris(
        earliest_time=obs_times_astropy.min() - 10 * u.day,
        latest_time=obs_times_astropy.max() + 10 * u.day,
        ssos="default solar system",
        de_ephemeris_version="440",
    )

    # precompute Chebyshev coefficients to compute the states of the perturbers
    # around each observation time for light travel time corrections
    perturber_pos_chebys, perturber_vel_chebys = [], []
    for t_ast in obs_times_astropy:
        pos, vel = generate_perturber_chebyshev_coeffs(
            obs_time=t_ast,
            ephem=ephem,
        )
        perturber_pos_chebys.append(pos)
        perturber_vel_chebys.append(vel)
    perturber_pos_chebys = jnp.array(perturber_pos_chebys)
    perturber_vel_chebys = jnp.array(perturber_vel_chebys)

    # cheby_t0/cheby_t1 live in the same offset frame as inputs.time inside
    # on_sky_acc_func, so the normalization 2*(t - cheby_t0)/(cheby_t1 - cheby_t0) - 1
    # is frame-agnostic as long as the three quantities are consistent.
    cheby_info = {
        "perturber_position_cheby_coeffs": perturber_pos_chebys,
        "perturber_velocity_cheby_coeffs": perturber_vel_chebys,
        "cheby_t0": obs_times - 6.0,
        "cheby_t1": obs_times + 2.0,
    }

    # Use natural adaptive steps from start past the last observation time.
    # state.time is 0.0 in the rebased frame; obs_times are offsets too.
    state = p.keplerian_state.to_system()
    t0 = state.time
    dynamic_acc_func = create_default_ephemeris_acceleration_func(
        ephem.processor, t_ref_jd=p._t_ref_jd
    )
    a0_dynamic = dynamic_acc_func(state)
    integrator_init = initialize_ias15_integrator_state(a0_dynamic)
    integrator_init.dt = obs_times[0] - t0 if len(obs_times) > 0 else 10.0
    dts = get_natural_dynamic_dts(
        initial_system_state=state,
        acceleration_func=dynamic_acc_func,
        final_time=jnp.max(obs_times),
        initial_integrator_state=integrator_init,
        step_scheduler=step_scheduler,
    )

    # Compute the start time of each step and precompute interpolation indices.
    # In the rebased frame t0 == 0.0, so the cumulative sum is the offset itself.
    t_step_starts = t0 + jnp.concatenate([jnp.array([0.0]), jnp.cumsum(dts[:-1])])
    step_indices, h_values = precompute_interpolation_indices(
        t_step_starts, dts, obs_times
    )

    # precompute the positions and velocities of all perturbers at each intermediate
    # step, and at each ias15 substep. This routine operates in absolute JD internally.
    planet_pos, planet_vel, asteroid_pos, asteroid_vel, gms = (
        precompute_perturber_positions(
            t0=p._t_ref_astropy,
            dts=dts,
            de_ephemeris_version="440",
        )
    )
    perturber_pos = jnp.concatenate((planet_pos, asteroid_pos), axis=2)
    perturber_vel = jnp.concatenate((planet_vel, asteroid_vel), axis=2)

    # Pre-compute perturber-perturber PPN quantities (P=11 GR perturbers only)
    planet_gms = jnp.exp(gms[:11])
    pp_a2, pp_a_newt, pp_a_gr = jax.vmap(
        jax.vmap(precompute_perturber_ppn, in_axes=(0, 0, None, None)),
        in_axes=(0, 0, None, None),
    )(planet_pos, planet_vel, planet_gms, SPEED_OF_LIGHT**2)

    observer_positions = p.observations.observer_positions

    return (
        cheby_info,
        dts,
        step_indices,
        h_values,
        perturber_pos,
        perturber_vel,
        gms,
        observer_positions,
        obs_times,
        p.observations.ra,
        p.observations.dec,
        pp_a2,
        pp_a_newt,
        pp_a_gr,
    )


def create_default_static_residuals_func(inputs: tuple) -> jax.tree_util.Partial:
    """Create a function to compute residuals using static integration and precomputed perturber positions.

    The returned function uses the "default" dynamical model, meaning GR effects for the
    Sun+planets, Newtonian gravity for the asteroids, and no non-gravitational
    or gravitational harmonic effects. Positions at observation times are obtained by
    interpolating within IAS15 steps using stored polynomial coefficients, rather than
    forcing steps to land on observation times.

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
        step_indices,
        h_values,
        perturber_pos,
        perturber_vel,
        log_gms,
        observer_positions,
        obs_times,
        obs_ras,
        obs_decs,
        pp_a2,
        pp_a_newt,
        pp_a_gr,
    ) = inputs

    static_acc_func = create_static_default_acceleration_func()
    on_sky_acc_func = create_static_default_on_sky_acc_func()

    def static_residuals_func(state: CartesianState | KeplerianState) -> jnp.ndarray:
        state = state.to_system()
        state.fixed_perturber_positions = perturber_pos[0, 0]
        state.fixed_perturber_velocities = perturber_vel[0, 0]
        state.fixed_perturber_log_gms = log_gms
        state.acceleration_func_kwargs = {
            **state.acceleration_func_kwargs,
            "pp_a2": pp_a2[0, 0],
            "pp_a_newt": pp_a_newt[0, 0],
            "pp_a_gr": pp_a_gr[0, 0],
        }
        a0 = static_acc_func(state)
        integrator_init = initialize_ias15_integrator_state(a0)
        integrator_init.dt = dts[0]

        (b_all, a0_all, x0_all, v0_all), _static_state, _static_integrator_state = (
            ias15_static_evolve(
                initial_system_state=state,
                acceleration_func=static_acc_func,
                dts=dts,
                initial_integrator_state=integrator_init,
                perturber_positions=perturber_pos,
                perturber_velocities=perturber_vel,
                perturber_log_gms=log_gms,
                pp_a2=pp_a2,
                pp_a_newt=pp_a_newt,
                pp_a_gr=pp_a_gr,
            )
        )

        # Gather per-obs dense-output slices for the single tracer (index 0).
        # Shapes: b_per_obs (n_obs, 7, 3); a0/x0/v0_per_obs (n_obs, 3);
        # dt_per_obs (n_obs,).
        b_per_obs = b_all[step_indices][:, :, 0, :]
        a0_per_obs = a0_all[step_indices][:, 0, :]
        x0_per_obs = x0_all[step_indices][:, 0, :]
        v0_per_obs = v0_all[step_indices][:, 0, :]
        dt_per_obs = dts[step_indices]

        def per_obs_on_sky(
            b_step: jnp.ndarray,
            a0_step: jnp.ndarray,
            x0_step: jnp.ndarray,
            v0_step: jnp.ndarray,
            dt_step: jnp.ndarray,
            h_obs: jnp.ndarray,
            time: jnp.ndarray,
            observer_pos: jnp.ndarray,
            cheby_info_obs: dict[str, jnp.ndarray],
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            # Build a closure that evaluates the IAS15 polynomial at light-travel-
            # delayed times within this step. on_sky uses it instead of the
            # constant-acceleration Taylor expansion.
            propagator = make_ltt_propagator(
                b_step, a0_step, x0_step, v0_step, dt_step, h_obs
            )
            # x_obs (=position at h_obs) is what on_sky uses to seed the LTT
            # loop and to compute the geometric distance to the observer.
            # When ltt_position_fn is provided, on_sky doesn't use the
            # velocity argument, so we pass zeros to keep the signature.
            x_obs = propagator(jnp.array(0.0))
            return on_sky(
                x_obs,
                jnp.zeros(3),
                time,
                observer_pos,
                on_sky_acc_func,
                cheby_info_obs,
                ltt_position_fn=propagator,
            )

        model_ras, model_decs = jax.vmap(
            per_obs_on_sky, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0)
        )(
            b_per_obs,
            a0_per_obs,
            x0_per_obs,
            v0_per_obs,
            dt_per_obs,
            h_values,
            obs_times,
            observer_positions,
            cheby_info,
        )

        xis_etas = jax.vmap(tangent_plane_projection)(
            obs_ras, obs_decs, model_ras, model_decs
        )

        return xis_etas

    return jax.jit(jax.tree_util.Partial(static_residuals_func))

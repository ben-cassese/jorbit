"""IAS15 dense-output ephemeris helpers (the ``interpolate=True`` path)."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from jorbit.astrometry.sky_projection import on_sky
from jorbit.integrators import (
    ias15_evolve_with_dense_output,
    initialize_ias15_integrator_state,
    make_ltt_propagator,
    stitched_per_query_gather,
)
from jorbit.particle.covariance import (
    _cov_from_jacobian,
    _state_to_vec,
    _state_vec_to_xv,
)
from jorbit.utils.states import CartesianState, IAS15IntegratorState, KeplerianState


@jax.jit
def _dense_ltt_radec(
    b_per_obs: jnp.ndarray,
    a0_per_obs: jnp.ndarray,
    x0_per_obs: jnp.ndarray,
    v0_per_obs: jnp.ndarray,
    dt_per_obs: jnp.ndarray,
    h_per_obs: jnp.ndarray,
    obs_times: jnp.ndarray,
    observer_positions: jnp.ndarray,
    acc_func: Callable,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-observation dense-output light-travel-time ``on_sky`` for a single particle.

    The ``on_sky`` light-travel-time correction defaults to a 2nd-order Taylor with a
    constant acceleration. For IAS15 we already have the converged 7th-order polynomial
    per step (the "dense output"), so this evaluates that polynomial at the
    light-travel-delayed time instead. Inputs are already gathered per observation:
    ``b_per_obs`` is ``(n, 7, 3)``; ``a0/x0/v0_per_obs`` are ``(n, 3)``;
    ``dt/h_per_obs/obs_times`` are ``(n,)``; ``observer_positions`` is ``(n, 3)``.
    """

    def per_obs_on_sky(
        b_step: jnp.ndarray,
        a0_step: jnp.ndarray,
        x0_step: jnp.ndarray,
        v0_step: jnp.ndarray,
        dt_step: jnp.ndarray,
        h_obs: jnp.ndarray,
        time: jnp.ndarray,
        observer_pos: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        propagator = make_ltt_propagator(
            b_step, a0_step, x0_step, v0_step, dt_step, h_obs
        )
        x_obs = propagator(jnp.array(0.0))
        return on_sky(
            x_obs,
            jnp.zeros(3),
            time,
            observer_pos,
            acc_func,
            ltt_position_fn=propagator,
        )

    return jax.vmap(per_obs_on_sky, in_axes=(0, 0, 0, 0, 0, 0, 0, 0))(
        b_per_obs,
        a0_per_obs,
        x0_per_obs,
        v0_per_obs,
        dt_per_obs,
        h_per_obs,
        obs_times,
        observer_positions,
    )


@jax.jit
def _ephem_ias15(
    times: jnp.ndarray,
    particle_state: CartesianState | KeplerianState,
    acc_func: Callable,
    integrator_state: IAS15IntegratorState,
    observer_positions: jnp.ndarray,
    relevant_inds: jnp.ndarray,
    step_scheduler: Callable,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Single-chunk IAS15 dense-output ephemeris (used inside the autodiff cov path).

    The truncation-proof nominal ephemeris uses :func:`_ephem_ias15_stitched` instead;
    this single-:func:`ias15_evolve_with_dense_output` version is retained because it is
    fully JIT-able and so can be wrapped by ``jax.jacfwd`` in
    :func:`_ephem_ias15_with_cov`.
    """
    state = particle_state.to_system()
    t0 = state.relative_time

    times_fwd = jnp.where(times >= t0, times, t0)
    times_bwd = jnp.where(times < t0, times, t0)

    out_fwd = ias15_evolve_with_dense_output(
        state,
        acc_func,
        times_fwd,
        integrator_state,
        step_scheduler,
    )
    (
        b_buf_fwd,
        a0_buf_fwd,
        x0_buf_fwd,
        v0_buf_fwd,
        dts_buf_fwd,
        _,
        step_indices_fwd,
        h_values_fwd,
    ) = out_fwd[5:13]

    out_bwd = ias15_evolve_with_dense_output(
        state,
        acc_func,
        times_bwd,
        integrator_state,
        step_scheduler,
    )
    (
        b_buf_bwd,
        a0_buf_bwd,
        x0_buf_bwd,
        v0_buf_bwd,
        dts_buf_bwd,
        _,
        step_indices_bwd,
        h_values_bwd,
    ) = out_bwd[5:13]

    obs_step_indices_fwd = step_indices_fwd[relevant_inds]
    ras_fwd, decs_fwd = _dense_ltt_radec(
        b_buf_fwd[obs_step_indices_fwd][:, :, 0, :],
        a0_buf_fwd[obs_step_indices_fwd][:, 0, :],
        x0_buf_fwd[obs_step_indices_fwd][:, 0, :],
        v0_buf_fwd[obs_step_indices_fwd][:, 0, :],
        dts_buf_fwd[obs_step_indices_fwd],
        h_values_fwd[relevant_inds],
        times_fwd[relevant_inds],
        observer_positions,
        acc_func,
    )

    obs_step_indices_bwd = step_indices_bwd[relevant_inds]
    ras_bwd, decs_bwd = _dense_ltt_radec(
        b_buf_bwd[obs_step_indices_bwd][:, :, 0, :],
        a0_buf_bwd[obs_step_indices_bwd][:, 0, :],
        x0_buf_bwd[obs_step_indices_bwd][:, 0, :],
        v0_buf_bwd[obs_step_indices_bwd][:, 0, :],
        dts_buf_bwd[obs_step_indices_bwd],
        h_values_bwd[relevant_inds],
        times_bwd[relevant_inds],
        observer_positions,
        acc_func,
    )

    is_fwd = times[relevant_inds] >= t0
    ras = jnp.where(is_fwd, ras_fwd, ras_bwd)
    decs = jnp.where(is_fwd, decs_fwd, decs_bwd)
    return ras, decs


def _ephem_ias15_stitched(
    times: jnp.ndarray,
    particle_state: CartesianState | KeplerianState,
    acc_func: Callable,
    integrator_state: IAS15IntegratorState,
    observer_positions: jnp.ndarray,
    relevant_inds: jnp.ndarray,
    step_scheduler: Callable,
) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    """Truncation-proof IAS15 dense-output ephemeris (nominal ``interpolate=True``).

    Host-side wrapper that stitches as many dense-output chunks as the span requires
    (see :func:`jorbit.integrators.stitched_per_query_gather`) before the same per-obs
    dense-LTT ``on_sky`` evaluation as :func:`_ephem_ias15`.
    """
    state = particle_state.to_system()
    b_q, a0_q, x0_q, v0_q, dt_q, h_q, steps = stitched_per_query_gather(
        state, acc_func, times, integrator_state, step_scheduler
    )
    # Single tracer at index 0; restrict to observation times.
    ras, decs = _dense_ltt_radec(
        b_q[relevant_inds][:, :, 0, :],
        a0_q[relevant_inds][:, 0, :],
        x0_q[relevant_inds][:, 0, :],
        v0_q[relevant_inds][:, 0, :],
        dt_q[relevant_inds],
        h_q[relevant_inds],
        times[relevant_inds],
        observer_positions,
        acc_func,
    )
    return ras, decs, steps


@jax.jit
def _ephem_ias15_with_cov(
    times: jnp.ndarray,
    particle_state: CartesianState | KeplerianState,
    acc_func: Callable,
    observer_positions: jnp.ndarray,
    relevant_inds: jnp.ndarray,
    step_scheduler: Callable,
    cov: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """IAS15 dense-output ephemeris with sky-plane covariance via forward-mode AD."""
    is_keplerian_param = isinstance(particle_state, KeplerianState)

    def radec_fn(state_vec: jnp.ndarray) -> jnp.ndarray:
        x, v = _state_vec_to_xv(state_vec, is_keplerian_param)
        state = CartesianState(
            x=x,
            v=v,
            relative_time=particle_state.relative_time,
            time_reference=particle_state.time_reference,
            acceleration_func_kwargs=particle_state.acceleration_func_kwargs,
        )
        a0 = acc_func(state.to_system())
        integrator_state = initialize_ias15_integrator_state(a0)
        ras, decs = _ephem_ias15(
            times,
            state,
            acc_func,
            integrator_state,
            observer_positions,
            relevant_inds,
            step_scheduler,
        )
        return jnp.stack([ras, decs], axis=1).flatten()

    nominal_vec = _state_to_vec(particle_state)
    return _cov_from_jacobian(radec_fn, nominal_vec, cov, relevant_inds.shape[0])

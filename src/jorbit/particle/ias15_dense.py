"""IAS15 dense-output ephemeris helpers (the ``interpolate=True`` path)."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp

from jorbit.integrators import (
    DenseOutput,
    apply_ltt_seed_floor,
    dense_ltt_radec,
    ias15_evolve_with_dense_output,
    initialize_ias15_integrator_state,
    ltt_backward_times,
    ltt_coverage_mask,
    ltt_seed_floor,
    stitched_dense_buffers,
)
from jorbit.particle.covariance import (
    _cov_from_jacobian,
    _state_to_vec,
    _state_vec_to_xv,
)
from jorbit.utils.states import CartesianState, IAS15IntegratorState, KeplerianState


@partial(jax.jit, static_argnames=["max_steps"])
def _ephem_ias15(
    times: jnp.ndarray,
    particle_state: CartesianState | KeplerianState,
    acc_func: Callable,
    integrator_state: IAS15IntegratorState,
    observer_positions: jnp.ndarray,
    relevant_inds: jnp.ndarray,
    step_scheduler: Callable,
    max_steps: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single-chunk IAS15 dense-output ephemeris (used inside the autodiff cov path).

    The truncation-proof nominal ephemeris uses :func:`_ephem_ias15_stitched` instead; this
    single-:func:`ias15_evolve_with_dense_output` version is retained because it is fully
    JIT-able and so can be wrapped by ``jax.jacfwd`` in :func:`_ephem_ias15_with_cov`.

    Being traced, it cannot extend its buffers or raise when a retarded time falls outside
    them, so it returns a per-observation ``ltt_covered`` mask instead and leaves the
    decision to its host-side caller.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            ``(ras, decs, ltt_covered)``, each of length ``len(relevant_inds)``.
    """
    state = particle_state.to_system()
    t0 = state.relative_time

    integrator_state = apply_ltt_seed_floor(
        integrator_state, state.tracer_positions, observer_positions
    )
    # Run the backward pass past the earliest requested time (and past the epoch, even when
    # nothing precedes it) by twice the largest light travel time, so that every retarded
    # time the LTT iteration asks for lands inside an integrated step.
    pad = ltt_seed_floor(state.tracer_positions, observer_positions)

    out_fwd = ias15_evolve_with_dense_output(
        state,
        acc_func,
        jnp.where(times >= t0, times, t0),
        integrator_state,
        step_scheduler,
        max_steps,
    )
    out_bwd = ias15_evolve_with_dense_output(
        state,
        acc_func,
        ltt_backward_times(times, t0, pad),
        integrator_state,
        step_scheduler,
        max_steps,
    )

    obs_times = times[relevant_inds]
    ras, decs, x_obs = dense_ltt_radec(
        DenseOutput(*out_fwd[5:11]),
        DenseOutput(*out_bwd[5:11]),
        t0,
        obs_times,
        observer_positions,
        acc_func,
    )
    ltt_covered = ltt_coverage_mask(
        x_obs,
        obs_times,
        observer_positions,
        out_bwd[2].relative_time,
        out_fwd[2].relative_time,
        t0=t0,
    )
    return ras[0], decs[0], ltt_covered


def _ephem_ias15_stitched(
    times: jnp.ndarray,
    particle_state: CartesianState | KeplerianState,
    acc_func: Callable,
    integrator_state: IAS15IntegratorState,
    observer_positions: jnp.ndarray,
    relevant_inds: jnp.ndarray,
    step_scheduler: Callable,
    max_steps: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    """Truncation-proof IAS15 dense-output ephemeris (nominal ``interpolate=True``).

    Host-side wrapper that stitches as many dense-output chunks as the span requires and
    extends the backward pass until it covers every retarded time (see
    :func:`jorbit.integrators.stitched_dense_buffers`), before the same per-obs dense-LTT
    ``on_sky`` evaluation as :func:`_ephem_ias15`.
    """
    state = particle_state.to_system()
    t0 = state.relative_time
    obs_times = times[relevant_inds]
    fwd, bwd, steps = stitched_dense_buffers(
        state,
        acc_func,
        times,
        integrator_state,
        step_scheduler,
        max_steps,
        obs_times=obs_times,
        observer_positions=observer_positions,
    )
    ras, decs, _x_obs = dense_ltt_radec(
        fwd, bwd, t0, obs_times, observer_positions, acc_func
    )
    return ras[0], decs[0], steps


@partial(jax.jit, static_argnames=["max_steps"])
def _ephem_ias15_with_cov(
    times: jnp.ndarray,
    particle_state: CartesianState | KeplerianState,
    acc_func: Callable,
    observer_positions: jnp.ndarray,
    relevant_inds: jnp.ndarray,
    step_scheduler: Callable,
    cov: jnp.ndarray,
    max_steps: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """IAS15 dense-output ephemeris with sky-plane covariance via forward-mode AD.

    Returns ``(ras, decs, cov_radec, ltt_covered)``; the caller is expected to raise on a
    false entry in the mask, since this path cannot extend its own buffers.
    """
    is_keplerian_param = isinstance(particle_state, KeplerianState)

    def radec_fn(state_vec: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
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
        ras, decs, ltt_covered = _ephem_ias15(
            times,
            state,
            acc_func,
            integrator_state,
            observer_positions,
            relevant_inds,
            step_scheduler,
            max_steps,
        )
        return jnp.stack([ras, decs], axis=1).flatten(), ltt_covered

    nominal_vec = _state_to_vec(particle_state)
    return _cov_from_jacobian(
        radec_fn, nominal_vec, cov, relevant_inds.shape[0], has_aux=True
    )

"""IAS15 dense-output ephemeris helpers (the ``interpolate=True`` path)."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp

from jorbit.astrometry.sky_projection import on_sky
from jorbit.integrators import (
    DenseOutput,
    apply_ltt_seed_floor,
    assert_ltt_span_covered,
    ias15_evolve_with_dense_output,
    initialize_ias15_integrator_state,
    ltt_seed_floor,
    make_ltt_propagator,
    stitched_dense_buffers,
)
from jorbit.particle.covariance import (
    _cov_from_jacobian,
    _state_to_vec,
    _state_vec_to_xv,
)
from jorbit.utils.states import CartesianState, IAS15IntegratorState, KeplerianState


@jax.jit
def _dense_ltt_radec(
    fwd: DenseOutput,
    bwd: DenseOutput,
    t0: jnp.ndarray,
    obs_times: jnp.ndarray,
    observer_positions: jnp.ndarray,
    acc_func: Callable,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-observation dense-output light-travel-time ``on_sky`` for a single particle.

    The ``on_sky`` light-travel-time correction defaults to a 2nd-order Taylor with a
    constant acceleration. For IAS15 we already have the converged 7th-order polynomial
    per step (the "dense output"), so this evaluates that polynomial at the
    light-travel-delayed time instead. The whole buffers are passed in (``in_axes=None``
    under the vmap): the retarded time of an observation routinely falls in an earlier
    step than the observation itself, so :func:`make_ltt_propagator` has to look it up.
    ``obs_times`` and ``observer_positions`` are per observation, shapes ``(n,)`` and
    ``(n, 3)``; ``t0`` is the integration epoch separating the two passes.
    """

    def per_obs_on_sky(
        time: jnp.ndarray, observer_pos: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        propagator = make_ltt_propagator(fwd, bwd, t0, time)
        x_obs = propagator(jnp.array(0.0))
        return on_sky(
            x_obs,
            jnp.zeros(3),
            time,
            observer_pos,
            acc_func,
            ltt_position_fn=propagator,
        )

    return jax.vmap(per_obs_on_sky)(obs_times, observer_positions)


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
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Single-chunk IAS15 dense-output ephemeris (used inside the autodiff cov path).

    The truncation-proof nominal ephemeris uses :func:`_ephem_ias15_stitched` instead;
    this single-:func:`ias15_evolve_with_dense_output` version is retained because it is
    fully JIT-able and so can be wrapped by ``jax.jacfwd`` in
    :func:`_ephem_ias15_with_cov`.
    """
    state = particle_state.to_system()
    t0 = state.relative_time

    integrator_state = apply_ltt_seed_floor(
        integrator_state, state.tracer_positions, observer_positions
    )
    # Run the backward pass past the earliest requested time (and past the epoch, even
    # when nothing precedes it) by twice the largest light travel time, so that every
    # retarded time the LTT iteration asks for lands inside an integrated step.
    pad = ltt_seed_floor(state.tracer_positions, observer_positions)

    times_fwd = jnp.where(times >= t0, times, t0)
    times_bwd = jnp.where(times < t0, times, t0)
    times_bwd = jnp.minimum(times_bwd, jnp.min(times_bwd) - pad)

    out_fwd = ias15_evolve_with_dense_output(
        state,
        acc_func,
        times_fwd,
        integrator_state,
        step_scheduler,
        max_steps,
    )
    out_bwd = ias15_evolve_with_dense_output(
        state,
        acc_func,
        times_bwd,
        integrator_state,
        step_scheduler,
        max_steps,
    )

    return _dense_ltt_radec(
        DenseOutput(*out_fwd[5:11]),
        DenseOutput(*out_bwd[5:11]),
        t0,
        times[relevant_inds],
        observer_positions,
        acc_func,
    )


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

    Host-side wrapper that stitches as many dense-output chunks as the span requires
    (see :func:`jorbit.integrators.stitched_dense_buffers`) before the same per-obs
    dense-LTT ``on_sky`` evaluation as :func:`_ephem_ias15`.
    """
    state = particle_state.to_system()
    t0 = state.relative_time
    integrator_state = apply_ltt_seed_floor(
        integrator_state, state.tracer_positions, observer_positions
    )
    pad = ltt_seed_floor(state.tracer_positions, observer_positions)
    fwd, bwd, steps = stitched_dense_buffers(
        state,
        acc_func,
        times,
        integrator_state,
        step_scheduler,
        max_steps,
        backward_pad=float(pad),
    )
    obs_times = times[relevant_inds]
    assert_ltt_span_covered(fwd, bwd, float(t0), obs_times, observer_positions)
    ras, decs = _dense_ltt_radec(fwd, bwd, t0, obs_times, observer_positions, acc_func)
    return ras, decs, steps


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
            max_steps,
        )
        return jnp.stack([ras, decs], axis=1).flatten()

    nominal_vec = _state_to_vec(particle_state)
    return _cov_from_jacobian(radec_fn, nominal_vec, cov, relevant_inds.shape[0])

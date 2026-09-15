"""IAS15 dense-output ephemeris helpers for the whole System (``interpolate=True``)."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp

from jorbit.astrometry.sky_projection import on_sky
from jorbit.data.constants import SPEED_OF_LIGHT
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
from jorbit.utils.states import IAS15IntegratorState, SystemState

# Tolerance (days) for "did the integrator reach this obs time" comparisons. Matches
# jorbit.integrators.budgeted._TIME_TOL: ~0.1 ms, far below any meaningful step size.
_TIME_TOL = 1e-9


@jax.jit
def dense_ltt_radec_multi(
    fwd: DenseOutput,
    bwd: DenseOutput,
    t0: jnp.ndarray,
    obs_times: jnp.ndarray,
    observer_positions: jnp.ndarray,
    acc_func: Callable,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-observation, per-particle dense-output light-travel-time ``on_sky``.

    Vmaps the dense-output polynomial-LTT closure over both the observation axis and the
    particle axis; each particle gets its own light-travel-time correction. The dense
    buffers are shared across both axes (``in_axes=None``), since the retarded time of
    an observation generally falls in an earlier step than the observation itself and
    :func:`make_ltt_propagator` has to look it up. ``obs_times`` is ``(n_obs,)`` and
    ``observer_positions`` is ``(n_obs, 3)``; returns ``(ras, decs)`` each shaped
    ``(P, n_obs)``.
    """

    def per_particle_per_obs(
        particle_index: jnp.ndarray,
        time: jnp.ndarray,
        observer_pos: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        propagator = make_ltt_propagator(fwd, bwd, t0, time, particle_index)
        x_obs = propagator(jnp.array(0.0))
        return on_sky(
            x_obs,
            jnp.zeros(3),
            time,
            observer_pos,
            acc_func,
            ltt_position_fn=propagator,
        )

    def for_single_particle(
        particle_index: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jax.vmap(per_particle_per_obs, in_axes=(None, 0, 0))(
            particle_index, obs_times, observer_positions
        )

    return jax.vmap(for_single_particle)(jnp.arange(fwd.x0.shape[1]))


def _ephem_ias15_stitched(
    times: jnp.ndarray,
    state: SystemState,
    acc_func: Callable,
    integrator_state: IAS15IntegratorState,
    observer_positions: jnp.ndarray,
    relevant_inds: jnp.ndarray,
    step_scheduler: Callable,
    max_steps: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    """Truncation-proof IAS15 dense-output ephemeris for the whole system.

    Host-side wrapper that stitches as many dense-output chunks as the span requires
    (see :func:`jorbit.integrators.stitched_dense_buffers`) before the per-obs,
    per-particle dense-LTT ``on_sky`` evaluation in :func:`dense_ltt_radec_multi`.
    """
    all_positions = jnp.concatenate((state.massive_positions, state.tracer_positions))
    integrator_state = apply_ltt_seed_floor(
        integrator_state, all_positions, observer_positions
    )
    # Run the backward pass past the earliest requested time (and past the epoch, even
    # when nothing precedes it) by twice the largest light travel time, so that every
    # retarded time the LTT iteration asks for lands inside an integrated step.
    pad = ltt_seed_floor(all_positions, observer_positions)
    fwd, bwd, steps = stitched_dense_buffers(
        state,
        acc_func,
        times,
        integrator_state,
        step_scheduler,
        max_steps,
        backward_pad=float(pad),
    )
    # Restrict to observation times (drops any intermediate landing times). For IAS15
    # relevant_inds is the identity, but keep the indexing uniform with other paths.
    obs_times = times[relevant_inds]
    t0 = state.relative_time
    assert_ltt_span_covered(fwd, bwd, float(t0), obs_times, observer_positions)
    ras, decs = dense_ltt_radec_multi(
        fwd, bwd, t0, obs_times, observer_positions, acc_func
    )
    return ras, decs, steps


@partial(jax.jit, static_argnames=["max_steps"])
def _ephem_ias15_bounded(
    states: jnp.ndarray,
    times_off: jnp.ndarray,
    fwd_mask: jnp.ndarray,
    times_fwd: jnp.ndarray,
    times_bwd: jnp.ndarray,
    observer_positions: jnp.ndarray,
    t_ref_jd: jnp.ndarray,
    acc_func: Callable,
    step_scheduler: Callable,
    max_steps: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fully-jitted, bounded-arc dense-output ephemeris for a ``(P, 6)`` state batch.

    The compile-once / reuse counterpart to :func:`_ephem_ias15_stitched`: it drops the
    host-side stitching loop (which forces device->host syncs every call) in favour of a
    fixed two-pass gather. The observation times are split into a forward pass (offsets
    ``>= 0``, ``times_fwd``) and a backward pass (offsets ``< 0``, ``times_bwd``, with
    the other pass's times clamped to 0). Each pass is a single
    :func:`ias15_evolve_with_dense_output` call, so the whole arc must fit in one dense
    buffer (``max_steps`` accepted steps; None uses ``IAS15_MAX_DYNAMIC_STEPS``);
    ``fwd_mask``/``times_fwd``/``times_bwd`` are
    precomputed once at bind time. Returns ``(ras, decs, reached_mask)`` where ``ras`` and
    ``decs`` are ``(P, n_obs)`` and ``reached_mask`` is ``(n_obs,)`` — ``False`` for any
    observation the (shared-schedule) integration failed to reach before the buffer filled.

    Args:
        states (jnp.ndarray):
            ``(P, 6)`` batch of barycentric equatorial Cartesian states
            ``[x, y, z, vx, vy, vz]`` (AU, AU/day) at the reference epoch.
        times_off (jnp.ndarray):
            ``(n_obs,)`` observation-time offsets (days) from ``t_ref_jd``.
        fwd_mask (jnp.ndarray):
            ``(n_obs,)`` boolean, ``times_off >= 0``.
        times_fwd (jnp.ndarray):
            ``(n_obs,)``, ``times_off`` where ``fwd_mask`` else 0.0.
        times_bwd (jnp.ndarray):
            ``(n_obs,)``, ``times_off`` where not ``fwd_mask`` else 0.0.
        observer_positions (jnp.ndarray):
            ``(n_obs, 3)`` observer positions (AU).
        t_ref_jd (jnp.ndarray):
            Scalar reference epoch (JD, TDB).
        acc_func (Callable):
            The system's acceleration function.
        step_scheduler (Callable):
            The adaptive step-size controller.
        max_steps (int | None):
            Dense-output buffer depth per directional pass (static; None uses
            ``IAS15_MAX_DYNAMIC_STEPS``). Arcs needing more accepted steps than this
            truncate batch-wide and are reported through ``reached_mask``.
    """
    empty3 = jnp.empty((0, 3))
    state = SystemState(
        tracer_positions=states[:, :3],
        tracer_velocities=states[:, 3:],
        massive_positions=empty3,
        massive_velocities=empty3,
        log_gms=jnp.empty((0,)),
        time_reference=jnp.asarray(t_ref_jd),
        relative_time=jnp.asarray(0.0),
        fixed_perturber_positions=empty3,
        fixed_perturber_velocities=empty3,
        fixed_perturber_log_gms=jnp.empty((0,)),
        acceleration_func_kwargs={"c2": SPEED_OF_LIGHT**2},
    )
    a0 = acc_func(state)
    integrator_state = initialize_ias15_integrator_state(a0)
    integrator_state = apply_ltt_seed_floor(
        integrator_state, states[:, :3], observer_positions
    )
    # Extend the backward pass past the earliest observation (and past the epoch, even
    # when every observation follows it) by twice the largest light travel time, so the
    # retarded times the LTT iteration asks for land inside an integrated step.
    # Recomputed from the traced states each call, so the span self-adapts as a sampler
    # explores large topocentric distances. (This fully-jitted hot loop gets no
    # host-side coverage check; the pad is the protection.)
    pad = ltt_seed_floor(states[:, :3], observer_positions)
    times_bwd = jnp.minimum(times_bwd, jnp.min(times_bwd) - pad)

    def gather(times_dir: jnp.ndarray) -> tuple[DenseOutput, jnp.ndarray]:
        out = ias15_evolve_with_dense_output(
            state, acc_func, times_dir, integrator_state, step_scheduler, max_steps
        )
        # out[2] is the final system state: the farthest time this pass reached.
        return DenseOutput(*out[5:11]), out[2].relative_time

    fwd, reached_fwd = gather(times_fwd)
    bwd, reached_bwd = gather(times_bwd)

    ras, decs = dense_ltt_radec_multi(
        fwd, bwd, jnp.asarray(0.0), times_off, observer_positions, acc_func
    )

    # Per-obs reach flag: did the (shared-schedule) integration actually reach each obs
    # time before its dense buffer filled? Forward obs must be within the forward pass's
    # reached time; backward obs within the backward pass's.
    reached_mask = jnp.where(
        fwd_mask,
        times_off <= reached_fwd + _TIME_TOL,
        times_off >= reached_bwd - _TIME_TOL,
    )
    return ras, decs, reached_mask

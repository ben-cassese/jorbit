"""IAS15 dense-output ephemeris helpers for the whole System (``interpolate=True``)."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp

from jorbit.data.constants import SPEED_OF_LIGHT
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
from jorbit.integrators.ias15.interpolation import _TIME_TOL
from jorbit.utils.states import IAS15IntegratorState, SystemState


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

    Host-side wrapper that stitches as many dense-output chunks as the span requires and
    extends the backward pass until it covers every retarded time (see
    :func:`jorbit.integrators.stitched_dense_buffers`), before the per-obs, per-particle
    dense-LTT ``on_sky`` evaluation in :func:`jorbit.integrators.dense_ltt_radec`.
    """
    # Restrict to observation times (drops any intermediate landing times). For IAS15
    # relevant_inds is the identity, but keep the indexing uniform with other paths.
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
        fwd, bwd, state.relative_time, obs_times, observer_positions, acc_func
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
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fully-jitted, bounded-arc dense-output ephemeris for a ``(P, 6)`` state batch.

    The compile-once / reuse counterpart to :func:`_ephem_ias15_stitched`: it drops the
    host-side stitching loop (which forces device->host syncs every call) in favour of a
    fixed two-pass gather. The observation times are split into a forward pass (offsets
    ``>= 0``, ``times_fwd``) and a backward pass (offsets ``< 0``, ``times_bwd``, with the
    other pass's times clamped to 0). Each pass is a single
    :func:`ias15_evolve_with_dense_output` call, so the whole arc must fit in one dense
    buffer (``max_steps`` accepted steps; None uses ``IAS15_MAX_DYNAMIC_STEPS``);
    ``fwd_mask``/``times_fwd``/``times_bwd`` are precomputed once at bind time.

    Returns ``(ras, decs, reached_mask, ltt_covered)``. ``ras`` and ``decs`` are
    ``(P, n_obs)``; the two masks are ``(n_obs,)`` and report the two ways an observation
    can come back unusable. ``reached_mask`` is False where the (shared-schedule)
    integration never reached that time before its buffer filled. ``ltt_covered`` is False
    where it did reach it, but the retarded time ``t_obs - LTT`` falls outside the
    integrated span, so the dense polynomial would have to be extrapolated. Being traced,
    this function can neither extend its buffers nor raise, which is why it reports both
    conditions rather than fixing or refusing them; :func:`_ephem_ias15_stitched` extends
    instead.

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
    # Extend the backward pass past the earliest observation (and past the epoch, even when
    # every observation follows it) by twice the largest light travel time, so the retarded
    # times the LTT iteration asks for land inside an integrated step. Recomputed from the
    # traced states each call, so the span self-adapts as a sampler explores large
    # topocentric distances. The pad is only a guess (it is sized from the epoch distance,
    # not the per-observation one); this path cannot extend it the way the stitched path
    # does, so any residual shortfall comes back in ltt_covered below.
    pad = ltt_seed_floor(states[:, :3], observer_positions)

    def gather(times_dir: jnp.ndarray) -> tuple[DenseOutput, jnp.ndarray]:
        out = ias15_evolve_with_dense_output(
            state, acc_func, times_dir, integrator_state, step_scheduler, max_steps
        )
        # out[2] is the final system state: the farthest time this pass reached.
        return DenseOutput(*out[5:11]), out[2].relative_time

    fwd, reached_fwd = gather(times_fwd)
    bwd, reached_bwd = gather(ltt_backward_times(times_bwd, 0.0, pad))

    ras, decs, x_obs = dense_ltt_radec(
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
    ltt_covered = ltt_coverage_mask(
        x_obs,
        times_off,
        observer_positions,
        reached_bwd,
        reached_fwd,
        reached_mask=reached_mask,
    )
    return ras, decs, reached_mask, ltt_covered

"""IAS15 dense-output ephemeris helpers for the whole System (``interpolate=True``)."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from jorbit.astrometry.sky_projection import on_sky
from jorbit.data.constants import SPEED_OF_LIGHT
from jorbit.integrators import (
    ias15_evolve_with_dense_output,
    initialize_ias15_integrator_state,
    make_ltt_propagator,
    stitched_per_query_gather,
)
from jorbit.utils.states import IAS15IntegratorState, SystemState

# Tolerance (days) for "did the integrator reach this obs time" comparisons. Matches
# jorbit.integrators.budgeted._TIME_TOL: ~0.1 ms, far below any meaningful step size.
_TIME_TOL = 1e-9


@jax.jit
def dense_ltt_radec_multi(
    b_per_obs_all: jnp.ndarray,
    a0_per_obs_all: jnp.ndarray,
    x0_per_obs_all: jnp.ndarray,
    v0_per_obs_all: jnp.ndarray,
    dt_per_obs: jnp.ndarray,
    h_per_obs: jnp.ndarray,
    obs_times: jnp.ndarray,
    observer_positions: jnp.ndarray,
    acc_func: Callable,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-observation, per-particle dense-output light-travel-time ``on_sky``.

    Vmaps the dense-output polynomial-LTT closure over both the observation axis and the
    particle axis; each particle gets its own light-travel-time correction. Inputs are
    already gathered per observation: ``b_per_obs_all`` is ``(n_obs, 7, P, 3)``;
    ``a0/x0/v0_per_obs_all`` are ``(n_obs, P, 3)``; ``dt/h_per_obs/obs_times`` are
    ``(n_obs,)``; ``observer_positions`` is ``(n_obs, 3)``. Returns ``(ras, decs)`` each
    shaped ``(P, n_obs)``.
    """

    def per_particle_per_obs(
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

    def for_single_particle(
        b_obs_p: jnp.ndarray,
        a0_obs_p: jnp.ndarray,
        x0_obs_p: jnp.ndarray,
        v0_obs_p: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        # b_obs_p: (n_obs, 7, 3); a0/v0/x0_obs_p: (n_obs, 3)
        return jax.vmap(per_particle_per_obs, in_axes=(0, 0, 0, 0, 0, 0, 0, 0))(
            b_obs_p,
            a0_obs_p,
            x0_obs_p,
            v0_obs_p,
            dt_per_obs,
            h_per_obs,
            obs_times,
            observer_positions,
        )

    # Vmap over particle axis: 2 in b_per_obs_all (axes are obs/coeff/particle/xyz),
    # 1 in a0/v0/x0_per_obs_all (axes are obs/particle/xyz).
    ras, decs = jax.vmap(for_single_particle, in_axes=(2, 1, 1, 1))(
        b_per_obs_all, a0_per_obs_all, x0_per_obs_all, v0_per_obs_all
    )
    return ras, decs


def _ephem_ias15_stitched(
    times: jnp.ndarray,
    state: SystemState,
    acc_func: Callable,
    integrator_state: IAS15IntegratorState,
    observer_positions: jnp.ndarray,
    relevant_inds: jnp.ndarray,
    step_scheduler: Callable,
) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    """Truncation-proof IAS15 dense-output ephemeris for the whole system.

    Host-side wrapper that stitches as many dense-output chunks as the span requires
    (see :func:`jorbit.integrators.stitched_per_query_gather`) before the per-obs,
    per-particle dense-LTT ``on_sky`` evaluation in :func:`dense_ltt_radec_multi`.
    """
    b_q, a0_q, x0_q, v0_q, dt_q, h_q, steps = stitched_per_query_gather(
        state, acc_func, times, integrator_state, step_scheduler
    )
    # Restrict to observation times (drops any intermediate landing times). For IAS15
    # relevant_inds is the identity, but keep the indexing uniform with other paths.
    ras, decs = dense_ltt_radec_multi(
        b_q[relevant_inds],
        a0_q[relevant_inds],
        x0_q[relevant_inds],
        v0_q[relevant_inds],
        dt_q[relevant_inds],
        h_q[relevant_inds],
        times[relevant_inds],
        observer_positions,
        acc_func,
    )
    return ras, decs, steps


@jax.jit
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
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fully-jitted, bounded-arc dense-output ephemeris for a ``(P, 6)`` state batch.

    The compile-once / reuse counterpart to :func:`_ephem_ias15_stitched`: it drops the
    host-side stitching loop (which forces device->host syncs every call) in favour of a
    fixed two-pass ``jnp.where`` gather. The observation times are split into a forward
    pass (offsets ``>= 0``, ``times_fwd``) and a backward pass (offsets ``< 0``,
    ``times_bwd``, with the other pass's times clamped to 0). Each pass is a single
    :func:`ias15_evolve_with_dense_output` call, so the whole arc must fit in one dense
    buffer (``IAS15_MAX_DYNAMIC_STEPS``); ``fwd_mask``/``times_fwd``/``times_bwd`` are
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

    def gather(times_dir: jnp.ndarray) -> tuple:
        out = ias15_evolve_with_dense_output(
            state, acc_func, times_dir, integrator_state, step_scheduler
        )
        (
            _p,
            _v,
            final_system_state,
            _fis,
            _it,
            b_buf,
            a0_buf,
            x0_buf,
            v0_buf,
            dts_buf,
            _tss,
            step_indices,
            h_values,
        ) = out
        return (
            b_buf[step_indices],  # (n_obs, 7, P, 3)
            a0_buf[step_indices],  # (n_obs, P, 3)
            x0_buf[step_indices],
            v0_buf[step_indices],
            dts_buf[step_indices],  # (n_obs,)
            h_values,  # (n_obs,)
            final_system_state.relative_time,  # scalar: farthest time reached
        )

    bf, af, xf, vf, df, hf, reached_fwd = gather(times_fwd)
    bb, ab, xb, vb, db, hb, reached_bwd = gather(times_bwd)

    # Select each observation from its own direction's gather.
    m_coeff = fwd_mask[:, None, None, None]
    m_vec = fwd_mask[:, None, None]
    b = jnp.where(m_coeff, bf, bb)
    a0p = jnp.where(m_vec, af, ab)
    x0p = jnp.where(m_vec, xf, xb)
    v0p = jnp.where(m_vec, vf, vb)
    dtp = jnp.where(fwd_mask, df, db)
    hp = jnp.where(fwd_mask, hf, hb)

    ras, decs = dense_ltt_radec_multi(
        b, a0p, x0p, v0p, dtp, hp, times_off, observer_positions, acc_func
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

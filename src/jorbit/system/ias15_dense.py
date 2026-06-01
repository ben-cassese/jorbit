"""IAS15 dense-output ephemeris helpers for the whole System (``interpolate=True``)."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from jorbit.astrometry.sky_projection import on_sky
from jorbit.integrators import make_ltt_propagator, stitched_per_query_gather
from jorbit.utils.states import IAS15IntegratorState, SystemState


@jax.jit
def _dense_ltt_radec_multi(
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
    per-particle dense-LTT ``on_sky`` evaluation in :func:`_dense_ltt_radec_multi`.
    """
    b_q, a0_q, x0_q, v0_q, dt_q, h_q, steps = stitched_per_query_gather(
        state, acc_func, times, integrator_state, step_scheduler
    )
    # Restrict to observation times (drops any intermediate landing times). For IAS15
    # relevant_inds is the identity, but keep the indexing uniform with other paths.
    ras, decs = _dense_ltt_radec_multi(
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

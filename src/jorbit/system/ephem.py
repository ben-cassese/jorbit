"""Generic multi-particle integrate/ephemeris helpers (leapfrog + building blocks)."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from jorbit.astrometry.sky_projection import on_sky
from jorbit.utils.states import IAS15IntegratorState, SystemState


@jax.jit
def _integrate(
    times: jnp.ndarray,
    state: SystemState,
    acc_func: Callable,
    integrator_func: Callable,
    integrator_state: IAS15IntegratorState,
    relevant_inds: jnp.ndarray,
    step_scheduler: Callable,
) -> tuple[jnp.ndarray, jnp.ndarray, SystemState, IAS15IntegratorState]:
    positions, velocities, final_system_state, final_integrator_state, _steps = (
        integrator_func(state, acc_func, times, integrator_state, step_scheduler)
    )

    return (
        positions[relevant_inds],
        velocities[relevant_inds],
        final_system_state,
        final_integrator_state,
    )


@jax.jit
def _ephem(
    times: jnp.ndarray,
    state: SystemState,
    acc_func: Callable,
    integrator_func: Callable,
    integrator_state: IAS15IntegratorState,
    observer_positions: jnp.ndarray,
    relevant_inds: jnp.ndarray,
    step_scheduler: Callable,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    positions, velocities, _, _ = _integrate(
        times,
        state,
        acc_func,
        integrator_func,
        integrator_state,
        relevant_inds,
        step_scheduler,
    )

    def interior(px: jnp.ndarray, pv: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        def scan_func(
            carry: None, scan_over: tuple[jnp.ndarray, jnp.ndarray]
        ) -> tuple[None, tuple[jnp.ndarray, jnp.ndarray]]:
            position, velocity, time, observer_position = scan_over
            ra, dec = on_sky(
                position,
                velocity,
                time,
                observer_position,
                acc_func,
                time_reference=state.time_reference,
            )
            return None, (ra, dec)

        _, (ras, decs) = jax.lax.scan(
            scan_func,
            None,
            (px, pv, times, observer_positions),
        )

        return ras, decs

    ras, decs = jax.vmap(interior, in_axes=(1, 1))(positions, velocities)
    return ras, decs

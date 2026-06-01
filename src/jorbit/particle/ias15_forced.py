"""IAS15 forced-landing ephemeris helper (the ``interpolate=False`` path)."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp

from jorbit.integrators import budgeted_forced_landing
from jorbit.particle.ephem import _on_sky_scan
from jorbit.utils.states import CartesianState, IAS15IntegratorState, KeplerianState


def _ephem_forced_budgeted(
    times: jnp.ndarray,
    particle_state: CartesianState | KeplerianState,
    acc_func: Callable,
    integrator_state: IAS15IntegratorState,
    observer_positions: jnp.ndarray,
    relevant_inds: jnp.ndarray,
    step_scheduler: Callable,
) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    """Truncation-proof IAS15 forced-landing ephemeris (nominal ``interpolate=False``).

    Host-side wrapper that inserts dummy landing times as needed (see
    :func:`jorbit.integrators.budgeted_forced_landing`) before the Taylor-LTT
    ``on_sky`` scan used by the non-dense paths.
    """
    state = particle_state.to_system()
    positions, velocities, steps = budgeted_forced_landing(
        state, acc_func, times, integrator_state, step_scheduler
    )
    ras, decs = _on_sky_scan(
        positions[relevant_inds][:, 0, :],
        velocities[relevant_inds][:, 0, :],
        times[relevant_inds],
        observer_positions,
        acc_func,
        particle_state.time_reference,
    )
    return ras, decs, steps

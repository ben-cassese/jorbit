"""Generic single-particle integrate/ephemeris helpers (leapfrog + building blocks)."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from jorbit.astrometry.sky_projection import on_sky
from jorbit.integrators import initialize_ias15_integrator_state
from jorbit.particle.covariance import (
    _cov_from_jacobian,
    _state_to_vec,
    _state_vec_to_xv,
)
from jorbit.utils.states import (
    CartesianState,
    IAS15IntegratorState,
    KeplerianState,
    LeapfrogIntegratorState,
    SystemState,
)


@jax.jit
def _integrate(
    times: jnp.ndarray,
    particle_state: CartesianState | KeplerianState,
    acc_func: Callable,
    integrator_func: Callable,
    integrator_state: IAS15IntegratorState | LeapfrogIntegratorState,
    relevant_inds: jnp.ndarray,
    step_scheduler: Callable,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    SystemState,
    IAS15IntegratorState | LeapfrogIntegratorState,
]:
    state = particle_state.to_system()
    positions, velocities, final_system_state, final_integrator_state, steps = (
        integrator_func(state, acc_func, times, integrator_state, step_scheduler)
    )

    return (
        positions[relevant_inds],
        velocities[relevant_inds],
        final_system_state,
        final_integrator_state,
        steps,
    )


@jax.jit
def _ephem(
    times: jnp.ndarray,
    particle_state: CartesianState | KeplerianState,
    acc_func: Callable,
    integrator_func: Callable,
    integrator_state: IAS15IntegratorState | LeapfrogIntegratorState,
    observer_positions: jnp.ndarray,
    relevant_inds: jnp.ndarray,
    step_scheduler: Callable,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    positions, velocities, _, _, _ = _integrate(
        times,
        particle_state,
        acc_func,
        integrator_func,
        integrator_state,
        relevant_inds,
        step_scheduler,
    )

    def scan_func(carry: None, scan_over: tuple) -> tuple[None, tuple]:
        position, velocity, time, observer_position = scan_over
        ra, dec = on_sky(
            position,
            velocity,
            time,
            observer_position,
            acc_func,
            time_reference=particle_state.time_reference,
        )
        return None, (ra, dec)

    _, (ras, decs) = jax.lax.scan(
        scan_func,
        None,
        (
            positions[:, 0, :],
            velocities[:, 0, :],
            times[relevant_inds],
            observer_positions,
        ),
    )

    return ras, decs


@jax.jit
def _on_sky_scan(
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    times: jnp.ndarray,
    observer_positions: jnp.ndarray,
    acc_func: Callable,
    time_reference: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Scan the (Taylor-LTT) ``on_sky`` over a sequence of single-particle states."""

    def scan_func(carry: None, scan_over: tuple) -> tuple[None, tuple]:
        position, velocity, time, observer_position = scan_over
        ra, dec = on_sky(
            position,
            velocity,
            time,
            observer_position,
            acc_func,
            time_reference=time_reference,
        )
        return None, (ra, dec)

    _, (ras, decs) = jax.lax.scan(
        scan_func, None, (positions, velocities, times, observer_positions)
    )
    return ras, decs


@jax.jit
def _ephem_with_cov(
    times: jnp.ndarray,
    particle_state: CartesianState | KeplerianState,
    acc_func: Callable,
    integrator_func: Callable,
    integrator_state: IAS15IntegratorState | LeapfrogIntegratorState,
    observer_positions: jnp.ndarray,
    relevant_inds: jnp.ndarray,
    step_scheduler: Callable,
    cov: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generic non-dense ephemeris with sky-plane covariance via forward-mode AD.

    Handles both the IAS15 forced-landing path (``interpolate=False`` + IAS15) and
    the leapfrog path. For IAS15, the integrator state must be re-initialized
    inside the AD closure so the initial-acceleration entry tracks the perturbed
    state vector; for leapfrog, ``LeapfrogIntegratorState`` is independent of
    the dynamical state and is reused as-is.
    """
    is_keplerian_param = isinstance(particle_state, KeplerianState)
    reinit_ias15 = isinstance(integrator_state, IAS15IntegratorState)

    def radec_fn(state_vec: jnp.ndarray) -> jnp.ndarray:
        x, v = _state_vec_to_xv(state_vec, is_keplerian_param)
        state = CartesianState(
            x=x,
            v=v,
            relative_time=particle_state.relative_time,
            time_reference=particle_state.time_reference,
            acceleration_func_kwargs=particle_state.acceleration_func_kwargs,
        )
        if reinit_ias15:
            a0 = acc_func(state.to_system())
            local_integrator_state = initialize_ias15_integrator_state(a0)
        else:
            local_integrator_state = integrator_state
        ras, decs = _ephem(
            times,
            state,
            acc_func,
            integrator_func,
            local_integrator_state,
            observer_positions,
            relevant_inds,
            step_scheduler,
        )
        return jnp.stack([ras, decs], axis=1).flatten()

    nominal_vec = _state_to_vec(particle_state)
    return _cov_from_jacobian(radec_fn, nominal_vec, cov, relevant_inds.shape[0])

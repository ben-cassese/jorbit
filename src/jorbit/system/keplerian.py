"""Analytic two-body (Keplerian) integrate/ephemeris helpers for the whole System.

These vmap the single-particle :func:`jorbit.particle.keplerian._keplerian_integrate` /
:func:`jorbit.particle.keplerian._keplerian_on_sky` over all particles in the System.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jorbit.particle.keplerian import _keplerian_integrate, _keplerian_on_sky


@jax.jit
def _keplerian_system_integrate(
    xs: jnp.ndarray,
    vs: jnp.ndarray,
    t0: float,
    times: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    # vmap _keplerian_integrate over particles: (N,T,3) for each
    positions, velocities = jax.vmap(_keplerian_integrate, in_axes=(0, 0, None, None))(
        xs, vs, t0, times
    )
    # transpose to (T,N,3) to match existing convention
    positions = jnp.transpose(positions, (1, 0, 2))
    velocities = jnp.transpose(velocities, (1, 0, 2))
    return positions, velocities


@jax.jit
def _keplerian_system_ephem(
    xs: jnp.ndarray,
    vs: jnp.ndarray,
    t0: float,
    times: jnp.ndarray,
    observer_positions: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    positions, velocities = _keplerian_system_integrate(xs, vs, t0, times)

    # _keplerian_on_sky operates on a single (position, velocity, time, observer)
    # vmap over times (axis 0 of positions[n]), then over particles (axis 1)
    _on_sky_over_times = jax.vmap(_keplerian_on_sky, in_axes=(0, 0, 0, 0))
    _on_sky_over_particles = jax.vmap(_on_sky_over_times, in_axes=(1, 1, None, None))

    ras, decs = _on_sky_over_particles(positions, velocities, times, observer_positions)
    return ras, decs

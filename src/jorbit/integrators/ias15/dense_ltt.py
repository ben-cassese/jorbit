"""Dense-output light-travel-time ``on_sky`` evaluation, shared by the Particle and System paths.

Kept separate from :mod:`jorbit.integrators.ias15.interpolation` only because it needs
``on_sky`` from :mod:`jorbit.astrometry`, which the rest of the interpolation machinery does
not.
"""

from __future__ import annotations

from collections.abc import Callable

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jorbit.astrometry.sky_projection import on_sky
from jorbit.integrators.ias15.interpolation import DenseOutput, make_ltt_propagator


@jax.jit
def dense_ltt_radec(
    fwd: DenseOutput,
    bwd: DenseOutput | None,
    t0: jnp.ndarray,
    obs_times: jnp.ndarray,
    observer_positions: jnp.ndarray,
    acc_func: Callable,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Per-observation, per-particle dense-output light-travel-time ``on_sky``.

    The ``on_sky`` light-travel-time correction defaults to a 2nd-order Taylor expansion
    with a constant acceleration. For IAS15 the converged 7th-order polynomial per step (the
    "dense output") is already in hand, so this evaluates that polynomial at the
    light-travel-delayed time instead.

    Vmaps over both the observation axis and the particle axis; each particle gets its own
    light-travel-time correction. The dense buffers are shared across both axes
    (``in_axes=None``), since the retarded time of an observation generally falls in an
    earlier step than the observation itself and :func:`make_ltt_propagator` has to look it
    up.

    Args:
        fwd (DenseOutput): Dense output of the forward pass.
        bwd (DenseOutput | None): Dense output of the backward pass, if there is one.
        t0 (jnp.ndarray): Integration epoch, the boundary between the two passes.
        obs_times (jnp.ndarray): Observation times, shape (n_obs,).
        observer_positions (jnp.ndarray): Observer positions, shape (n_obs, 3).
        acc_func (Callable): The system's acceleration function.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            ``(ras, decs, x_obs)``, shaped ``(P, n_obs)``, ``(P, n_obs)`` and
            ``(P, n_obs, 3)``. ``x_obs`` is the particle position at each observation time,
            which ``on_sky`` needs anyway to seed its light-travel-time iteration; returning
            it lets callers run a coverage check (see
            :func:`jorbit.integrators.ltt_coverage_mask`) at no extra cost.
    """

    def per_particle_per_obs(
        particle_index: jnp.ndarray,
        time: jnp.ndarray,
        observer_pos: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        propagator = make_ltt_propagator(fwd, bwd, t0, time, particle_index)
        x_obs = propagator(jnp.array(0.0))
        ra, dec = on_sky(
            x_obs,
            jnp.zeros(3),
            time,
            observer_pos,
            acc_func,
            ltt_position_fn=propagator,
        )
        return ra, dec, x_obs

    def for_single_particle(
        particle_index: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return jax.vmap(per_particle_per_obs, in_axes=(None, 0, 0))(
            particle_index, obs_times, observer_positions
        )

    return jax.vmap(for_single_particle)(jnp.arange(fwd.x0.shape[1]))

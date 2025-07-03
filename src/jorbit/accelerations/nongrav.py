"""Marsden-style nongravitational accelerations for asteroids."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.utils.states import SystemState


def nongrav_acceleration(state: SystemState) -> jnp.ndarray:
    """Compute the nongravitational acceleration felt by each particle.

    Args:
        state (SystemState): The instantaneous state of the system. Uses a1, a2, a3 from
            state.acceleration_func_kwargs. Coefficients are assumed zero if not
            provided, and there should be one coefficient for each particle.

    Returns:
        jnp.ndarray:
            The 3D nongravitational acceleration felt by each particle, ordered by
            massive particles first followed by tracer particles.

    """
    x = jnp.concatenate([state.massive_positions, state.tracer_positions])
    v = jnp.concatenate([state.massive_velocities, state.tracer_velocities])
    a1 = state.acceleration_func_kwargs.get("a1", jnp.zeros_like(x))
    a2 = state.acceleration_func_kwargs.get("a2", jnp.zeros_like(x))
    a3 = state.acceleration_func_kwargs.get("a3", jnp.zeros_like(x))

    r = jnp.linalg.norm(x, axis=1)
    r_cross_v = jnp.cross(x, v, axis=1)
    r_cross_v_cross_r = jnp.cross(r_cross_v, x, axis=1)

    g_prefactor = (1 / r) ** 2  # for asteroids only, comets would rely on specific
    # choice of below coefficients

    # r0 = 1.0
    # alpha = 1.0
    # nk = 0.0
    # nm = 2.0
    # nn = 5.093
    # g_prefactor = alpha * (r/r0) ** (-nm) * (1.0 + (r / r0) ** nn) ** (-nk), 1/r**2

    term1 = a1[:, None] * x / r[:, None]
    term2 = (
        a2[:, None]
        * r_cross_v_cross_r
        / jnp.linalg.norm(r_cross_v_cross_r, axis=1)[:, None]
    )
    term3 = a3[:, None] * r_cross_v / jnp.linalg.norm(r_cross_v, axis=1)[:, None]

    return g_prefactor[:, None] * (term1 + term2 + term3)

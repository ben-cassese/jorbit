"""Simple Newtonian gravity acceleration function."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.utils.states import SystemState


@jax.jit
def newtonian_gravity(inputs: SystemState) -> jnp.ndarray:
    """Compute the acceleration felt by each particle due to Newtonian gravity.

    Set up to separate massive and tracer particles, so systems with many tracers
    will not compute useless pairwise interactions.

    Note: We use "stop_gradient" on perturbers that are passed as fixed inputs, so
    any gradients with respect to these perturber quantities will not be correct. To
    track gradients with respect to perturbers, they must be included as "massive"
    particles, not "fixed perturbers".

    Args:
        inputs (SystemState): The instantaneous state of the system.

    Returns:
        jnp.ndarray:
            The 3D acceleration felt by each particle, ordered by massive particles
            first followed by tracer particles.
    """
    M = inputs.massive_positions.shape[0]  # number of massive particles
    gms = jnp.exp(inputs.log_gms)  # (M,)

    # massive particles due to other massive particles
    dx_massive = (
        inputs.massive_positions[:, None, :] - inputs.massive_positions[None, :, :]
    )  # (M,M,3)
    r2_massive = jnp.sum(dx_massive * dx_massive, axis=-1)  # (M,M)
    r3_massive = r2_massive * jnp.sqrt(r2_massive)  # (M,M)

    mask_massive = ~jnp.eye(M, dtype=bool)  # (M,M)
    prefac_massive = jnp.where(mask_massive, 1.0 / r3_massive, 0.0)

    a_massive = -jnp.sum(
        prefac_massive[:, :, None] * dx_massive * gms[None, :, None],
        axis=1,
    )  # (M,3)

    # tracer particles due to massive particles
    dx_tracers = (
        inputs.tracer_positions[:, None, :] - inputs.massive_positions[None, :, :]
    )  # (T,M,3)
    r2_tracers = jnp.sum(dx_tracers * dx_tracers, axis=-1)  # (T,M)
    r3_tracers = r2_tracers * jnp.sqrt(r2_tracers)  # (T,M)

    a_tracers = -jnp.sum(
        (1.0 / r3_tracers)[:, :, None] * dx_tracers * gms[None, :, None],
        axis=1,
    )  # (T,3)

    all_as = jnp.concatenate([a_massive, a_tracers], axis=0)

    # Fixed perturbers acting on all (massive + tracer) particles.
    # stop_gradient since we never need gradients through fixed perturber quantities.
    p_pos = jax.lax.stop_gradient(inputs.fixed_perturber_positions)
    p_gms = jax.lax.stop_gradient(jnp.exp(inputs.fixed_perturber_log_gms))

    dx_perturbers = (
        jnp.concatenate([inputs.massive_positions, inputs.tracer_positions], axis=0)[
            :, None, :
        ]
        - p_pos[None, :, :]
    )  # (N,P,3)
    r2_perturbers = jnp.sum(dx_perturbers * dx_perturbers, axis=-1)  # (N,P)
    r3_perturbers = r2_perturbers * jnp.sqrt(r2_perturbers)  # (N,P)
    a_perturbers = -jnp.sum(
        (1.0 / r3_perturbers)[:, :, None] * dx_perturbers * p_gms[None, :, None],
        axis=1,
    )  # (N,3)

    return all_as + a_perturbers

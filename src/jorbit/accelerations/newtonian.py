import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.utils.states import SystemState


@jax.jit
def newtonian_gravity(inputs: SystemState) -> jnp.ndarray:
    """
    At an instantaneous moment in time, calculate the acceleration on a set of particles
    due to Newtonian gravity.

    Args:
        inputs: AccelerationInputs object containing the positions, velocities, and GMs
            of the particles.

    Returns:
        accelerations: Accelerations on each particle due to Newtonian gravity

    """
    # Calculate pairwise differences
    N = inputs.positions.shape[0]
    dx = inputs.positions[:, None, :] - inputs.positions[None, :, :]  # (N,N,3)
    r2 = jnp.sum(dx * dx, axis=-1)  # (N,N)
    r = jnp.sqrt(r2)  # (N,N)
    r3 = r2 * r  # (N,N)

    # Mask for i!=j calculations
    mask = ~jnp.eye(N, dtype=bool)  # (N,N)

    prefac = 1.0 / r3
    prefac = jnp.where(mask, prefac, 0.0)
    a_newt = -jnp.sum(
        prefac[:, :, None] * dx * jnp.exp(inputs.log_gms[None, :, None]), axis=1
    )  # (N,3)
    return a_newt

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
    G = 1.0  # Gravitational constant
    separations = inputs.positions[:, None, :] - inputs.positions[None, :, :]
    separations_norm = jnp.linalg.norm(separations, axis=-1)
    separations_norm += jnp.eye(separations_norm.shape[0]) * 1e300

    acceleration_magnitudes_from_each = -G * inputs.gms / (separations_norm**3)
    accelerations = jnp.sum(
        separations * acceleration_magnitudes_from_each[:, :, None], axis=1
    )
    return accelerations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.utils.states import SystemState


@jax.jit
def newtonian_gravity(inputs: SystemState) -> jnp.ndarray:
    M = inputs.massive_positions.shape[0]  # number of massive particles
    T = inputs.tracer_positions.shape[0]  # number of tracer particles

    # 1. Compute accelerations on massive particles due to other massive particles
    dx_massive = (
        inputs.massive_positions[:, None, :] - inputs.massive_positions[None, :, :]
    )  # (M,M,3)
    r2_massive = jnp.sum(dx_massive * dx_massive, axis=-1)  # (M,M)
    r3_massive = r2_massive * jnp.sqrt(r2_massive)  # (M,M)

    # Mask for i!=j calculations among massive particles
    mask_massive = ~jnp.eye(M, dtype=bool)  # (M,M)
    prefac_massive = jnp.where(mask_massive, 1.0 / r3_massive, 0.0)

    # Accelerations on massive particles from massive particles
    a_massive = -jnp.sum(
        prefac_massive[:, :, None]
        * dx_massive
        * jnp.exp(inputs.log_gms[None, :, None]),
        axis=1,
    )  # (M,3)

    # 2. Compute accelerations on tracer particles due to massive particles
    # (This will work even when T=0 due to JAX's shape polymorphism)
    dx_tracers = (
        inputs.tracer_positions[:, None, :] - inputs.massive_positions[None, :, :]
    )  # (T,M,3)
    r2_tracers = jnp.sum(dx_tracers * dx_tracers, axis=-1)  # (T,M)
    r3_tracers = r2_tracers * jnp.sqrt(r2_tracers)  # (T,M)

    # Accelerations on tracer particles from massive particles
    a_tracers = -jnp.sum(
        (1.0 / r3_tracers)[:, :, None]
        * dx_tracers
        * jnp.exp(inputs.log_gms[None, :, None]),
        axis=1,
    )  # (T,3)

    # Combine accelerations for all particles
    # This works even when T=0 thanks to JAX's handling of zero-sized arrays
    return jnp.concatenate([a_massive, a_tracers], axis=0)

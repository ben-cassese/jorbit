import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.accelerations.newtonian import newtonian_gravity

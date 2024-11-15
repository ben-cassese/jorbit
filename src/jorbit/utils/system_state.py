import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.data.constants import JORBIT_EPOCH


@jax.tree_util.register_pytree_node_class
class SystemState:
    def __init__(
        self,
        positions,
        velocities,
        gms,
        time=JORBIT_EPOCH,
        acceleration_func_kwargs=None,
    ):
        self.gms = gms
        self.positions = positions
        self.velocities = velocities
        self.time = time
        self.acceleration_func_kwargs = acceleration_func_kwargs

    def tree_flatten(self):
        children = (
            self.positions,
            self.velocities,
            self.gms,
            self.time,
            self.acceleration_func_kwargs,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from typing import Callable
from functools import partial

from jorbit.utils.system_state import SystemState


@jax.tree_util.register_pytree_node_class
class IntegratorState:
    def __init__(
        self,
        dt,
        meta=None,
    ):
        self.dt = dt
        self.meta = meta

    def tree_flatten(self):
        children = (
            self.dt,
            self.meta,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


from jorbit.integrators.rk4 import rk4_step, rk4_evolve
from jorbit.integrators.yoshida import yoshida_step  # , yoshida_evolve

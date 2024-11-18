import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from typing import Callable
from functools import partial

from jorbit.utils.states import SystemState
from jorbit.data.constants import Y4_C, Y4_D, Y6_C, Y6_D, Y8_C, Y8_D


@jax.tree_util.register_pytree_node_class
class IntegratorState:
    def __init__(
        self,
        meta=None,
    ):
        self.meta = meta

    def tree_flatten(self):
        children = (self.meta,)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class RK4IntegratorState(IntegratorState):
    def __init__(self, dt, meta=None):
        super().__init__(meta=meta)
        self.dt = dt

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


@jax.tree_util.register_pytree_node_class
class LeapfrogIntegratorState(IntegratorState):
    def __init__(self, c_coeff, d_coeff, dt, meta=None):
        super().__init__(meta=meta)
        self.c_coeff = c_coeff
        self.d_coeff = d_coeff
        self.dt = dt

    def tree_flatten(self):
        children = (
            self.c_coeff,
            self.d_coeff,
            self.dt,
            self.meta,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


# @jax.tree_util.register_pytree_node_class
# class GaussJacksonIntegratorState(IntegratorState):
#     def __init__(self, dt):
#         super().__init__(dt)

# @jax.tree_util.register_pytree_node_class
# class IAS15IntegratorState(IntegratorState):
#     def __init__(self, dt):
#         super().__init__(dt)

from jorbit.integrators.rk4 import rk4_step, rk4_evolve
from jorbit.integrators.yoshida import yoshida_step, leapfrog_evolve

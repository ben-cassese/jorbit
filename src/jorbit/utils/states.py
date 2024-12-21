import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.astrometry.transformations import (
    elements_to_cartesian,
    cartesian_to_elements,
    icrs_to_horizons_ecliptic,
    horizons_ecliptic_to_icrs,
)


@jax.tree_util.register_pytree_node_class
class SystemState:
    def __init__(
        self,
        tracer_positions,
        tracer_velocities,
        massive_positions,
        massive_velocities,
        log_gms,
        time=0.0,
        acceleration_func_kwargs=None,
    ):
        self.tracer_positions = tracer_positions
        self.tracer_velocities = tracer_velocities
        self.massive_positions = massive_positions
        self.massive_velocities = massive_velocities
        self.log_gms = log_gms
        self.time = time
        self.acceleration_func_kwargs = acceleration_func_kwargs

    def tree_flatten(self):
        children = (
            self.tracer_positions,
            self.tracer_velocities,
            self.massive_positions,
            self.massive_velocities,
            self.log_gms,
            self.time,
            self.acceleration_func_kwargs,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class KeplerianState:
    def __init__(
        self,
        semi,
        ecc,
        inc,
        Omega,
        omega,
        nu,
        time,
    ):
        self.semi = semi
        self.ecc = ecc
        self.inc = inc
        self.Omega = Omega
        self.omega = omega
        self.nu = nu
        self.time = time

    def tree_flatten(self):
        children = (
            self.semi,
            self.ecc,
            self.inc,
            self.Omega,
            self.omega,
            self.nu,
            self.time,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @jax.jit
    def to_cartesian(self):
        x, v = elements_to_cartesian(
            jnp.array([self.semi]),
            jnp.array([self.ecc]),
            jnp.array([self.inc]),
            jnp.array([self.Omega]),
            jnp.array([self.omega]),
            jnp.array([self.nu]),
        )
        x = icrs_to_horizons_ecliptic(x)
        v = icrs_to_horizons_ecliptic(v)
        return CartesianState(x, v, self.time)

    @jax.jit
    def to_keplerian(self):
        return self


@jax.tree_util.register_pytree_node_class
class CartesianState:
    def __init__(
        self,
        x,
        v,
        time,
    ):
        self.x = x
        self.v = v
        self.time = time

    def tree_flatten(self):
        children = (
            self.x,
            self.v,
            self.time,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @jax.jit
    def to_keplerian(self):
        x = horizons_ecliptic_to_icrs(self.x)
        v = horizons_ecliptic_to_icrs(self.v)
        a, ecc, nu, inc, Omega, omega = cartesian_to_elements(x, v)
        return KeplerianState(a, ecc, inc, Omega, omega, nu, self.time)

    @jax.jit
    def to_cartesian(self):
        return self

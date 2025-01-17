import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


from jorbit.integrators.ias15 import ias15_step, ias15_evolve

__all__ = [
    "ias15_step",
    "ias15_evolve",
    "initialize_ias15_integrator_state",
    "IAS15IntegratorState",
    "IAS15Helper",
]


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


@jax.tree_util.register_pytree_node_class
class IAS15Helper:
    # the equivalent of the reb_dp7 struct in rebound, but obviously without pointers
    # kinda just a spicy dictionary, not sure if this is how I want to do it
    def __init__(self, p0, p1, p2, p3, p4, p5, p6):
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.p5 = p5
        self.p6 = p6

    def tree_flatten(self):
        children = (
            self.p0,
            self.p1,
            self.p2,
            self.p3,
            self.p4,
            self.p5,
            self.p6,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class IAS15IntegratorState(IntegratorState):
    def __init__(self, g, b, e, br, er, csx, csv, a0, dt, dt_last_done, meta=None):
        """

        Parameters:
            g: IAS15Helper
            b: IAS15Helper
            e: IAS15Helper
            br: IAS15Helper
            er: IAS15Helper
            csx: jnp.ndarray
            csv: jnp.ndarray
            a0: jnp.ndarray
            dt: float
            dt_last_done: float

        """
        super().__init__(meta=meta)
        self.g = g
        self.b = b
        self.e = e
        self.br = br
        self.er = er
        self.csx = csx
        self.csv = csv
        self.a0 = a0
        self.dt = dt
        self.dt_last_done = dt_last_done

        # at, x0, v0, a0, csx, csv, csa0,
        # self.at = jnp.zeros((n_particles, 3), dtype=jnp.float64)

    def tree_flatten(self):
        children = (
            self.g,
            self.b,
            self.e,
            self.br,
            self.er,
            self.csx,
            self.csv,
            self.a0,
            self.dt,
            self.dt_last_done,
            self.meta,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def initialize_ias15_helper(n_particles):
    return IAS15Helper(
        p0=jnp.zeros((n_particles, 3), dtype=jnp.float64),
        p1=jnp.zeros((n_particles, 3), dtype=jnp.float64),
        p2=jnp.zeros((n_particles, 3), dtype=jnp.float64),
        p3=jnp.zeros((n_particles, 3), dtype=jnp.float64),
        p4=jnp.zeros((n_particles, 3), dtype=jnp.float64),
        p5=jnp.zeros((n_particles, 3), dtype=jnp.float64),
        p6=jnp.zeros((n_particles, 3), dtype=jnp.float64),
    )


def initialize_ias15_integrator_state(a0):
    n_particles = a0.shape[0]
    return IAS15IntegratorState(
        g=initialize_ias15_helper(n_particles),
        b=initialize_ias15_helper(n_particles),
        e=initialize_ias15_helper(n_particles),
        br=initialize_ias15_helper(n_particles),
        er=initialize_ias15_helper(n_particles),
        csx=jnp.zeros((n_particles, 3), dtype=jnp.float64),
        csv=jnp.zeros((n_particles, 3), dtype=jnp.float64),
        a0=a0,
        dt=10.0,  # 10 days
        dt_last_done=0.0,
    )

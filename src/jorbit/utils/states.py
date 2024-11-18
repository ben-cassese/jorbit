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


@jax.tree_util.register_pytree_node_class
class LikelihoodState:
    """
    Data required to evaluate the likelihood of a set of observations for *one* object.

    This is a PyTree-compatible way to hold data required to evaluate a likelihood
    function for one set of observations. One set of observations implies that, while
    there may be multiple minor planets in a system, there will only be at most one
    tracer particle here (all massive particles always need to be included). If there
    are multiple observed particles in the same system, we'll either vmap or
    sequentially loop over multiple LikelihoodState objects.

    """

    def __init__(
        self,
        positions,
        velocities,
        gms,
        acceleration_func_kwargs,
        ind_of_observed_particle,
        gradient_masks,
        observation_times,
        observatory_locations,
        observed_coordinates,
    ):
        self.positions = positions
        self.velocities = velocities
        self.gms = gms
        self.acceleration_func_kwargs = acceleration_func_kwargs
        self.ind_of_observed_particle = ind_of_observed_particle
        self.gradient_masks = gradient_masks
        self.observation_times = observation_times
        self.observatory_locations = observatory_locations
        self.observed_coordinates = observed_coordinates

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

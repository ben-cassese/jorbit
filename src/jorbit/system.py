import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.particle import Particle
from jorbit.utils.states import SystemState


class System:
    def __init__(self, particles, acceleration_func, acceleration_func_kwargs=None):
        self.particles = particles
        self.acceleration_func = acceleration_func

        xs = jnp.array([p.x for p in particles])
        vs = jnp.array([p.v for p in particles])
        gms = jnp.array([p.gm for p in particles])

        self.checks()

        self.state = SystemState(
            positions=xs,
            velocities=vs,
            gms=gms,
            time=self.epoch,
            acceleration_func_kwargs=acceleration_func_kwargs,
        )

    def __repr__(self):
        return f"*************\njorbit System\n time: {self.state.time}\n particles: {self.particles}\n*************"

    def checks(self):
        times = jnp.array([p.time for p in self.particles])
        t0 = times[0]
        assert jnp.allclose(
            times, t0
        ), "All particles must have the same reference time"
        self.epoch = t0

    # break up the combinations of particles into "likelihood blocks", where each block
    # is the minimum number of particles needed to compute the likelihood of *1* set
    # of observations. This is useful for parallelizing the likelihood computation.

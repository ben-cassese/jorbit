import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from astropy.time import Time


from jorbit.ephemeris.ephemeris import Ephemeris
from jorbit.particle import Particle
from jorbit.utils.states import SystemState

from jorbit.accelerations import create_newtonian_ephemeris_acceleration_func
from jorbit.data.constants import (
    DEFAULT_PLANET_EPHEMERIS_URL,
    DEFAULT_ASTEROID_EPHEMERIS_URL,
)


class System:
    def __init__(
        self,
        particles,
        acceleration_func="newtonian planets",
        acceleration_func_kwargs=None,
        integrator=None,
        earliest_time=Time("1980-01-01"),
        latest_time=Time("2100-01-01"),
    ):

        self.particles = particles
        self.checks()

        self.setup_acceleration_func(acceleration_func)

        xs = jnp.array([p.x for p in particles])
        vs = jnp.array([p.v for p in particles])
        log_gms = jnp.array([p.log_gm for p in particles])
        self.state = SystemState(
            positions=xs,
            velocities=vs,
            log_gms=log_gms,
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

    def setup_acceleration_func(self, acceleration_func):
        if acceleration_func == "newtonian planets":
            eph = Ephemeris(
                earliest_time=Time("1980-01-01"),
                latest_time=Time("2100-01-01"),
                ssos="default planets",
            )
            acc_func = create_newtonian_ephemeris_acceleration_func(eph.processor)
            self.acceleration_func = acc_func
        elif acceleration_func == "newtonian solar system":
            eph = Ephemeris(
                earliest_time=Time("1980-01-01"),
                latest_time=Time("2100-01-01"),
                ssos="default solar system",
            )

            acc_func = create_newtonian_ephemeris_acceleration_func(eph.processor)

            self.acceleration_func = acc_func

    # break up the combinations of particles into "likelihood blocks", where each block
    # is the minimum number of particles needed to compute the likelihood of *1* set
    # of observations. This is useful for parallelizing the likelihood computation.

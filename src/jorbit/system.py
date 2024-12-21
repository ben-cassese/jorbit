import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from astropy.time import Time


from jorbit.ephemeris.ephemeris import Ephemeris
from jorbit.particle import Particle
from jorbit.utils.states import SystemState

from jorbit.accelerations import create_newtonian_ephemeris_acceleration_func
from jorbit.integrators import ias15_evolve

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
        integrator=ias15,
        earliest_time=Time("1980-01-01"),
        latest_time=Time("2100-01-01"),
    ):

        self.particles = particles
        self.checks()

        self._setup_acceleration_func(acceleration_func)
        self._setup_integrator(integrator)

        # the global state for the system, used
        # when evolving the entire thing simultaneously
        # to a certain time
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

        self._likelihood_units = self._setup_likelihood_units()

    def __repr__(self):
        return f"*************\njorbit System\n time: {self.state.time}\n particles: {self.particles}\n*************"

    def checks(self):
        times = jnp.array([p.time for p in self.particles])
        t0 = times[0]
        assert jnp.allclose(
            times, t0
        ), "All particles must have the same reference time"
        self.epoch = t0

    def _setup_acceleration_func(self, acceleration_func):
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

    def _setup_integrator(self, integrator):
        if integrator != "ias15":
            raise NotImplementedError(
                "Currently only the IAS15 integrator is supported"
            )

        a0 = self.acceleration_func(self.state)
        self.state.integrator_state = initialize_ias15_integrator_state(a0)
        self.integrator = ias15_evolve

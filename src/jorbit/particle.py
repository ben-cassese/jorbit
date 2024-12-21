import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from astropy.time import Time

from jorbit.accelerations import (
    create_newtonian_ephemeris_acceleration_func,
    create_gr_ephemeris_acceleration_func,
)
from jorbit.ephemeris.ephemeris import Ephemeris
from jorbit.utils.states import CartesianState, KeplerianState


class Particle:
    def __init__(
        self,
        x=None,
        v=None,
        elements=None,
        time=None,
        log_gm=-jnp.inf,
        observations=None,
        name="",
        gravity="newtonian planets",
        integrator="ias15",
        earliest_time=Time("1980-01-01"),
        latest_time=Time("2100-01-01"),
    ):
        self.x = x
        self.v = v
        self.elements = elements
        self.log_gm = log_gm
        self.time = time
        self.observations = observations
        self.name = name
        self.gravity = gravity
        self.integrator_func = integrator
        self.earliest_time = earliest_time
        self.latest_time = latest_time

        self._setup()

        self._setup_acceleration_func()

    def __repr__(self):
        return f"Particle: {self.name}"

    def _setup(self):

        assert self.time is not None, "Must provide an epoch for the particle"
        if isinstance(self.time, type(Time("2023-01-01"))):
            self.time = self.time.tdb.jd

        if self.elements is not None:
            assert self.x is None, "Cannot provide both x and elements"
            assert self.v is None, "Cannot provide both v and elements"

            if isinstance(self.elements, dict):
                self.keplerian_state = KeplerianState(**self.elements, time=self.time)

            self.cartesian_state = self.keplerian_state.to_cartesian()
        elif self.x is not None:
            assert self.v is not None, "Must provide both x and v"

            self.cartesian_state = CartesianState(
                x=jnp.array([self.x]), v=jnp.array([self.v]), time=self.time
            )
            self.keplerian_state = self.cartesian_state.to_keplerian()
        else:
            raise ValueError(
                "time must be either astropy.time.Time or float (interpreted as JD in"
                " TDB)"
            )

        if self.name == "":
            self.name = "unnamed"

    def _setup_acceleration_func(self):
        if self.gravity == "newtonian planets":
            eph = Ephemeris(
                earliest_time=self.earliest_time,
                latest_time=self.latest_time,
                ssos="default planets",
            )
            acc_func = create_newtonian_ephemeris_acceleration_func(eph.processor)
            self.gravity = acc_func
        elif self.gravity == "newtonian solar system":
            eph = Ephemeris(
                earliest_time=self.earliest_time,
                latest_time=self.latest_time,
                ssos="default solar system",
            )
            acc_func = create_newtonian_ephemeris_acceleration_func(eph.processor)
            self.gravity = acc_func
        elif self.gravity == "gr planets":
            eph = Ephemeris(
                earliest_time=self.earliest_time,
                latest_time=self.latest_time,
                ssos="default planets",
            )
            acc_func = create_gr_ephemeris_acceleration_func(eph.processor)
            self.gravity = acc_func
        elif self.gravity == "gr solar system":
            eph = Ephemeris(
                earliest_time=self.earliest_time,
                latest_time=self.latest_time,
                ssos="default solar system",
            )
            acc_func = create_gr_ephemeris_acceleration_func(eph.processor)
            self.gravity = acc_func

    def integrate(self, times, state=None):
        if state is None:
            state = self.cartesian_state
        return _integrate(times, state, self.gravity, self.integrator_func)

    def ephemeris(self, times=None, state=None, observer_positions=None):
        pass

    def errors(self):
        pass

    def likelihood(self):
        pass

    def max_likelihood(self):
        pass

    def fit(self):
        pass

    def fit_ephemeris(self):
        pass


@jax.jit
def _integrate(
    times,
    particle_state,
    acc_func,
    integrator_func,
):
    state = particle_state.to_cartesian()
    system_state = SystemState(
        positions=jnp.array([state.x]),
        velocities=jnp.array([state.v]),
        log_gms=jnp.array([state.log_gm]),
        time=state.time,
        acceleration_func_kwargs=state.acceleration_func_kwargs,
    )

    positions, velocities, final_system_state, final_integrator_state = integrator_func(
        system_state, acc_func, times, integrator_state
    )

    return positions[0], velocities[0]

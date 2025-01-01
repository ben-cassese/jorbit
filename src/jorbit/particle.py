import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from astropy.time import Time

from jorbit.accelerations import (
    create_newtonian_ephemeris_acceleration_func,
    create_gr_ephemeris_acceleration_func,
    create_default_ephemeris_acceleration_func,
)
from jorbit.integrators import ias15_evolve, initialize_ias15_integrator_state
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
        latest_time=Time("2050-01-01"),
    ):
        self._x = x
        self._v = v
        self._elements = elements
        self._log_gm = log_gm
        self._time = time
        self._observations = observations
        self._name = name
        self.gravity = gravity
        self._integrator = integrator
        self._earliest_time = earliest_time
        self._latest_time = latest_time

        self._setup()

        self._setup_acceleration_func()

        self._setup_integrator()

    def __repr__(self):
        return f"Particle: {self._name}"

    def _setup(self):

        assert self._time is not None, "Must provide an epoch for the particle"
        if isinstance(self._time, type(Time("2023-01-01"))):
            self._time = self._time.tdb.jd

        if self._elements is not None:
            assert self._x is None, "Cannot provide both x and elements"
            assert self._v is None, "Cannot provide both v and elements"

            if isinstance(self._elements, dict):
                self._keplerian_state = KeplerianState(
                    **self._elements, time=self._time
                )

            self._cartesian_state = self._keplerian_state.to_cartesian()
        elif self._x is not None:
            assert self._v is not None, "Must provide both x and v"

            self._cartesian_state = CartesianState(
                x=jnp.array([self._x]), v=jnp.array([self._v]), time=self._time
            )
            self._keplerian_state = self._cartesian_state.to_keplerian()
        else:
            raise ValueError(
                "time must be either astropy.time.Time or float (interpreted as JD in"
                " TDB)"
            )

        if self._name == "":
            self._name = "unnamed"

    def _setup_acceleration_func(self):
        if self.gravity == "newtonian planets":
            eph = Ephemeris(
                earliest_time=self._earliest_time,
                latest_time=self._latest_time,
                ssos="default planets",
            )
            acc_func = create_newtonian_ephemeris_acceleration_func(eph.processor)
            self.gravity = acc_func
        elif self.gravity == "newtonian solar system":
            eph = Ephemeris(
                earliest_time=self._earliest_time,
                latest_time=self._latest_time,
                ssos="default solar system",
            )
            acc_func = create_newtonian_ephemeris_acceleration_func(eph.processor)
            self.gravity = acc_func
        elif self.gravity == "gr planets":
            eph = Ephemeris(
                earliest_time=self._earliest_time,
                latest_time=self._latest_time,
                ssos="default planets",
            )
            acc_func = create_gr_ephemeris_acceleration_func(eph.processor)
            self.gravity = acc_func
        elif self.gravity == "gr solar system":
            eph = Ephemeris(
                earliest_time=self._earliest_time,
                latest_time=self._latest_time,
                ssos="default solar system",
            )
            acc_func = create_gr_ephemeris_acceleration_func(eph.processor)
            self.gravity = acc_func
        elif self.gravity == "default solar system":
            eph = Ephemeris(
                earliest_time=self._earliest_time,
                latest_time=self._latest_time,
                ssos="default solar system",
            )
            acc_func = create_default_ephemeris_acceleration_func(eph.processor)
            self.gravity = acc_func

    def _setup_integrator(self):
        if self._integrator != "ias15":
            raise NotImplementedError(
                "Currently only the IAS15 integrator is supported"
            )

        a0 = self.gravity(self._cartesian_state.to_system())
        self._integrator_state = initialize_ias15_integrator_state(a0)
        self._integrator = jax.tree_util.Partial(ias15_evolve)

    def integrate(self, times, state=None):
        if state is None:
            state = self._cartesian_state
        if type(times) == Time:
            times = jnp.array(times.tdb.jd)
        if times.shape == ():
            times = jnp.array([times])
        positions, velocities, final_system_state, final_integrator_state = _integrate(
            times, state, self.gravity, self._integrator, self._integrator_state
        )
        return positions[0], velocities[0]

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
    integrator_state,
):
    state = particle_state.to_system()
    positions, velocities, final_system_state, final_integrator_state = integrator_func(
        state, acc_func, times, integrator_state
    )

    return positions, velocities, final_system_state, final_integrator_state

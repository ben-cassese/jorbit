import jax

jax.config.update("jax_enable_x64", True)
import astropy.units as u
import jax.numpy as jnp
from astropy.coordinates import SkyCoord
from astropy.time import Time

from jorbit.accelerations import (
    create_default_ephemeris_acceleration_func,
    create_gr_ephemeris_acceleration_func,
    create_newtonian_ephemeris_acceleration_func,
)
from jorbit.astrometry.sky_projection import on_sky
from jorbit.ephemeris.ephemeris import Ephemeris
from jorbit.integrators import ias15_evolve, initialize_ias15_integrator_state
from jorbit.utils.horizons import get_observer_positions
from jorbit.utils.states import SystemState


class System:
    def __init__(
        self,
        particles=None,
        state=None,
        gravity="default solar system",
        integrator="ias15",
        earliest_time=Time("1980-01-01"),
        latest_time=Time("2050-01-01"),
    ):

        self._earliest_time = earliest_time
        self._latest_time = latest_time

        if state is None:
            assert particles is not None
            times = jnp.array([p._time for p in particles])
            t0 = times[0]
            assert jnp.allclose(
                times, t0
            ), "All particles must have the same reference time"

            self._state = SystemState(
                tracer_positions=jnp.array([p._x for p in particles]),
                tracer_velocities=jnp.array([p._v for p in particles]),
                massive_positions=jnp.empty((0, 3)),
                massive_velocities=jnp.empty((0, 3)),
                log_gms=jnp.empty((0,)),
                time=t0,
                acceleration_func_kwargs={},
            )
        else:
            self._state = state

        self.gravity = self._setup_acceleration_func(gravity)

        self._integrator_state, self._integrator = self._setup_integrator()

    def __repr__(self):
        return f"*************\njorbit System\n time: {Time(self._state.time, format='jd', scale='tdb').utc.iso}\n*************"

    def _setup_acceleration_func(self, gravity):

        if isinstance(gravity, jax.tree_util.Partial):
            return gravity

        if gravity == "newtonian planets":
            eph = Ephemeris(
                earliest_time=self._earliest_time,
                latest_time=self._latest_time,
                ssos="default planets",
            )
            acc_func = create_newtonian_ephemeris_acceleration_func(eph.processor)
        elif gravity == "newtonian solar system":
            eph = Ephemeris(
                earliest_time=self._earliest_time,
                latest_time=self._latest_time,
                ssos="default solar system",
            )
            acc_func = create_newtonian_ephemeris_acceleration_func(eph.processor)
        elif gravity == "gr planets":
            eph = Ephemeris(
                earliest_time=self._earliest_time,
                latest_time=self._latest_time,
                ssos="default planets",
            )
            acc_func = create_gr_ephemeris_acceleration_func(eph.processor)
        elif gravity == "gr solar system":
            eph = Ephemeris(
                earliest_time=self._earliest_time,
                latest_time=self._latest_time,
                ssos="default solar system",
            )
            acc_func = create_gr_ephemeris_acceleration_func(eph.processor)
        elif gravity == "default solar system":
            eph = Ephemeris(
                earliest_time=self._earliest_time,
                latest_time=self._latest_time,
                ssos="default solar system",
            )
            acc_func = create_default_ephemeris_acceleration_func(eph.processor)

        return acc_func

    def _setup_integrator(self):
        a0 = self.gravity(self._state)
        integrator_state = initialize_ias15_integrator_state(a0)
        integrator = jax.tree_util.Partial(ias15_evolve)

        return integrator_state, integrator

    ################
    # PUBLIC METHODS
    ################

    def integrate(self, times):
        if isinstance(times, Time):
            times = jnp.array(times.tdb.jd)
        if times.shape == ():
            times = jnp.array([times])

        positions, velocities, final_system_state, final_integrator_state = _integrate(
            times, self._state, self.gravity, self._integrator, self._integrator_state
        )
        return positions, velocities

    def ephemeris(self, times, observer):
        if isinstance(observer, str):
            observer_positions = get_observer_positions(times, observer)
        else:
            observer_positions = observer

        if isinstance(times, Time):
            times = jnp.array(times.tdb.jd)
        if times.shape == ():
            times = jnp.array([times])

        ras, decs = _ephem(
            times,
            self._state,
            self.gravity,
            self._integrator,
            self._integrator_state,
            observer_positions,
        )
        return SkyCoord(ra=ras, dec=decs, unit=u.rad, frame="icrs")


@jax.jit
def _integrate(
    times,
    state,
    acc_func,
    integrator_func,
    integrator_state,
):
    positions, velocities, final_system_state, final_integrator_state = integrator_func(
        state, acc_func, times, integrator_state
    )

    return positions, velocities, final_system_state, final_integrator_state


@jax.jit
def _ephem(
    times,
    particle_state,
    acc_func,
    integrator_func,
    integrator_state,
    observer_positions,
):
    positions, velocities, _, _ = _integrate(
        times, particle_state, acc_func, integrator_func, integrator_state
    )

    def interior(px, pv):
        def scan_func(carry, scan_over):
            position, velocity, time, observer_position = scan_over
            ra, dec = on_sky(position, velocity, time, observer_position, acc_func)
            return None, (ra, dec)

        _, (ras, decs) = jax.lax.scan(
            scan_func,
            None,
            (px, pv, times, observer_positions),
        )

        return ras, decs

    ras, decs = jax.vmap(interior, in_axes=(1, 1))(positions, velocities)
    return ras, decs

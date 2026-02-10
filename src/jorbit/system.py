"""The System class and its supporting functions."""
import jax

jax.config.update("jax_enable_x64", True)
from collections.abc import Callable

import astropy.units as u
import jax.numpy as jnp
from astropy.coordinates import SkyCoord
from astropy.time import Time

from jorbit.accelerations import (
    create_default_ephemeris_acceleration_func,
    create_gr_ephemeris_acceleration_func,
    create_newtonian_ephemeris_acceleration_func,
    newtonian_gravity,
    ppn_gravity,
)
from jorbit.astrometry.sky_projection import on_sky
from jorbit.data.constants import Y4_C, Y4_D, Y6_C, Y6_D, Y8_C, Y8_D
from jorbit.ephemeris.ephemeris import Ephemeris
from jorbit.integrators import (
    create_leapfrog_times,
    ias15_evolve,
    initialize_ias15_integrator_state,
    leapfrog_evolve,
)
from jorbit.utils.horizons import get_observer_positions
from jorbit.utils.states import (
    IAS15IntegratorState,
    LeapfrogIntegratorState,
    SystemState,
)


class System:
    """A system of particles in the solar system.

    Very similar in spirit to the `Particle` class, but now for multiple massless
    particles.
    """

    def __init__(
        self,
        particles: list | None = None,
        state: SystemState | None = None,
        gravity: str | Callable = "default solar system",
        de_ephemeris_version: str | None = "440",
        integrator: str = "ias15",
        earliest_time: Time = Time("1980-01-01"),
        latest_time: Time = Time("2050-01-01"),
        max_step_size: u.Quantity | None = None,
    ) -> None:
        """Initialize a System.

        Args:
            particles (list, optional):
                A list of Particle objects. None if state is provided. Defaults to None.
            state (SystemState, optional):
                A SystemState object. None if particles is provided. Defaults to None.
            gravity (str | Callable):
                The gravitational acceleration function to use when integrating the
                particle's orbit. Defaults to "default solar system", which corresponds
                to parameterized post-Newtonian interactions with the 10 bodies in the
                JPL DE440 ephemeris, plus Newtonian interactions with the 16 largest
                asteroids in the asteroids_de441/sb441-n16.bsp ephemeris. Can also be
                a jax.tree_util.Partial object that follows the same signature as the
                acceleration functions in jorbit.accelerations.
            de_ephemeris_version (str | None):
                Which version of the JPL DE ephemeris to use for perturber positions
                when using one of the built-in gravity models. Accepts either "440" or
                "430", default is "440".
            integrator (str):
                The integrator to use for the particle. Choices are "ias15", which is a
                15th order adaptive step-size integrator, or "Y4", "Y6", or "Y8", which
                are 4th, 6th, and 8th order Yoshida leapfrog integrators with fixed
                step sizes. Defaults to "ias15".
            earliest_time (Time):
                The earliest time we expect to integrate the particle to. Defaults to
                Time("1980-01-01"). Larger time windows will result in larger in-memory
                ephemeris objects.
            latest_time (Time):
                The latest time we expect to integrate the particle to. Defaults to
                Time("2050-01-01"). Larger time windows will result in larger in-memory
                ephemeris objects.
            max_step_size (u.Quantity, optional):
                The fixed step size to use for leapfrog integrators. Required if
                integrator is "Y4", "Y6", or "Y8". Ignored if integrator is "ias15".
                Note that this is the maximum step size; the actual step size may be
                smaller to ensure that the particle lands exactly on the requested
                output times, and that the step size may change if the spacing between
                output times is not constant. Defaults to None.
        """
        self._earliest_time = earliest_time
        self._latest_time = latest_time
        self._de_ephemeris_version = de_ephemeris_version

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
                fixed_perturber_positions=jnp.empty((0, 3)),
                fixed_perturber_velocities=jnp.empty((0, 3)),
                fixed_perturber_log_gms=jnp.empty((0,)),
                acceleration_func_kwargs={},
            )
        else:
            self._state = state

        self.gravity = self._setup_acceleration_func(gravity)

        self._integrator_state, self._integrator = self._setup_integrator(
            integrator, max_step_size
        )
        self._integrator_method = integrator
        self._max_step_size = max_step_size

    def __repr__(self) -> str:
        """Return a string representation of the System."""
        return f"*************\njorbit System\n time: {Time(self._state.time, format='jd', scale='tdb').utc.iso}\n*************"

    def _setup_acceleration_func(self, gravity: str | Callable) -> Callable:

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

        elif gravity == "generic newtonian":
            acc_func = jax.tree_util.Partial(newtonian_gravity)

        elif gravity == "generic gr":
            acc_func = jax.tree_util.Partial(ppn_gravity)

        return acc_func

    def _setup_integrator(
        self, integrator: str, max_step_size: u.Quantity | None
    ) -> tuple[IAS15IntegratorState | LeapfrogIntegratorState, Callable]:
        if integrator == "ias15":
            assert (
                max_step_size is None
            ), "max_step_size should not be provided for IAS15 integrator."
            a0 = self.gravity(self._state)
            integrator_state = initialize_ias15_integrator_state(a0)
            integrator = jax.tree_util.Partial(ias15_evolve)
        elif integrator in ["Y4", "Y6", "Y8"]:
            assert (
                max_step_size is not None
            ), "Must provide max_step_size for leapfrog integrators."
            dt = max_step_size.to(u.day).value
            if integrator == "Y4":
                c = Y4_C
                d = Y4_D
            elif integrator == "Y6":
                c = Y6_C
                d = Y6_D
            elif integrator == "Y8":
                c = Y8_C
                d = Y8_D

            integrator_state = LeapfrogIntegratorState(dt=dt, C=c, D=d)
            integrator = jax.tree_util.Partial(leapfrog_evolve)

        return integrator_state, integrator

    ################
    # PUBLIC METHODS
    ################

    def integrate(self, times: Time) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Integrate the System to a given time.

        Note: This method does not change the state of the system. It returns the
        positions and velocities at the given times, but the system itself is not
        changed.

        Args:
            times (Time | jnp.ndarray):
                The times to integrate to. Can be a single time or an array of times.
                If provided as a jnp.array, the entries are assumed to be in TDB JD.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]:
                The positions of the particle at the given times, in AU, and the
                The velocities of the particle at the given times, in AU/day.
        """
        if isinstance(times, Time):
            times = jnp.array(times.tdb.jd)
        if times.shape == ():
            times = jnp.array([times])

        if self._integrator_method in ["Y4", "Y6", "Y8"]:
            # For leapfrog, need to create intermediate times
            times, inds = create_leapfrog_times(
                t0=self._state.time,
                times=times,
                biggest_allowed_dt=self._integrator_state.dt,
            )
        else:
            inds = jnp.arange(times.shape[0])

        positions, velocities, _final_system_state, _final_integrator_state = (
            _integrate(
                times,
                self._state,
                self.gravity,
                self._integrator,
                self._integrator_state,
                inds,
            )
        )

        return positions, velocities

    def ephemeris(
        self, times: Time | jnp.ndarray, observer: str | jnp.ndarray
    ) -> SkyCoord:
        """Compute an ephemeris for the system.

        Args:
            times (Time | jnp.ndarray):
                The times to compute the ephemeris for. Can be a single time or an array
                of times. If provided as a jnp.array, the entries are assumed to be in
                TDB JD.
            observer (str | jnp.ndarray):
                The observer to compute the ephemeris for. Can be a string representing
                an observatory name, or a 3D position vector in AU. For more info on
                acceptable strings, see the get_observer_positions function.

        Returns:
            coords (SkyCoord):
                The ephemeris of each particle in the system at the given times, in ICRS
                coordinates and as seen from that specific observer. Each particle has
                its own light travel time correction applied individually.
        """
        if isinstance(observer, str):
            observer_positions = get_observer_positions(
                times, observer, self._de_ephemeris_version
            )
        else:
            observer_positions = observer

        if isinstance(times, Time):
            times = jnp.array(times.tdb.jd)
        if times.shape == ():
            times = jnp.array([times])

        if self._integrator_method in ["Y4", "Y6", "Y8"]:
            # For leapfrog, need to create intermediate times
            times, inds = create_leapfrog_times(
                t0=self._state.time,
                times=times,
                biggest_allowed_dt=self._integrator_state.dt,
            )
        else:
            inds = jnp.arange(times.shape[0])

        ras, decs = _ephem(
            times,
            self._state,
            self.gravity,
            self._integrator,
            self._integrator_state,
            observer_positions,
            inds,
        )

        return SkyCoord(ra=ras, dec=decs, unit=u.rad, frame="icrs")


@jax.jit
def _integrate(
    times: jnp.ndarray,
    state: SystemState,
    acc_func: Callable,
    integrator_func: Callable,
    integrator_state: IAS15IntegratorState,
    relevant_inds: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, SystemState, IAS15IntegratorState]:
    positions, velocities, final_system_state, final_integrator_state = integrator_func(
        state, acc_func, times, integrator_state
    )

    return (
        positions[relevant_inds],
        velocities[relevant_inds],
        final_system_state,
        final_integrator_state,
    )


@jax.jit
def _ephem(
    times: jnp.ndarray,
    state: SystemState,
    acc_func: Callable,
    integrator_func: Callable,
    integrator_state: IAS15IntegratorState,
    observer_positions: jnp.ndarray,
    relevant_inds: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    positions, velocities, _, _ = _integrate(
        times, state, acc_func, integrator_func, integrator_state, relevant_inds
    )

    def interior(px: jnp.ndarray, pv: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        def scan_func(
            carry: None, scan_over: tuple[jnp.ndarray, jnp.ndarray]
        ) -> tuple[None, tuple[jnp.ndarray, jnp.ndarray]]:
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

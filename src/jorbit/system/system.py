"""The System class.

The jitted/host helper functions that implement each integration branch live in sibling
modules of this subpackage:

- :mod:`jorbit.system.ephem` (generic integrate/ephem, leapfrog)
- :mod:`jorbit.system.ias15_dense` (IAS15 ``interpolate=True``)
- :mod:`jorbit.system.keplerian` (analytic two-body path)
"""

from __future__ import annotations

from collections.abc import Callable

import astropy.units as u
import jax
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
from jorbit.data.constants import Y4_C, Y4_D, Y6_C, Y6_D, Y8_C, Y8_D
from jorbit.ephemeris.ephemeris import Ephemeris
from jorbit.integrators import (
    create_leapfrog_times,
    ias15_evolve,
    initialize_ias15_integrator_state,
    leapfrog_evolve,
    next_proposed_dt_global,
    next_proposed_dt_PRS23,
    stitched_interpolate,
)
from jorbit.system.ephem import _ephem, _integrate
from jorbit.system.ias15_dense import _ephem_ias15_stitched
from jorbit.system.keplerian import _keplerian_system_ephem, _keplerian_system_integrate
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
        self._is_keplerian = gravity == "keplerian"

        # Mirrors the Particle rebase: the JAX-visible state always carries
        # relative_time=0.0, the absolute epoch lives on self._t_ref_astropy /
        # self._t_ref_jd (and on the state's time_reference), and any
        # user-supplied query times are converted to offsets via
        # _times_to_offsets before they hit the integrator.
        if state is None:
            assert particles is not None
            t_ref_jds = jnp.array([p._t_ref_jd for p in particles])
            assert jnp.allclose(
                t_ref_jds, t_ref_jds[0], atol=1e-6, rtol=0
            ), "All particles must have the same reference time (tolerance: 1e-6 days ~ 0.1 s)"
            self._t_ref_astropy = particles[0]._t_ref_astropy
            self._t_ref_jd = particles[0]._t_ref_jd

            self._state = SystemState(
                tracer_positions=jnp.array([p._x for p in particles]),
                tracer_velocities=jnp.array([p._v for p in particles]),
                massive_positions=jnp.empty((0, 3)),
                massive_velocities=jnp.empty((0, 3)),
                log_gms=jnp.empty((0,)),
                time_reference=self._t_ref_jd,
                relative_time=jnp.array(0.0),
                fixed_perturber_positions=jnp.empty((0, 3)),
                fixed_perturber_velocities=jnp.empty((0, 3)),
                fixed_perturber_log_gms=jnp.empty((0,)),
                acceleration_func_kwargs={},
            )
        else:
            # The state self-describes its absolute epoch as
            # relative_time + time_reference (the same convention as
            # CartesianState/KeplerianState and Particle.__init__). Use that sum
            # as the System's reference epoch, then rebase the stored state to
            # relative_time=0.0 against the new anchor.
            abs_jd = float(state.relative_time) + float(state.time_reference)
            self._t_ref_astropy = Time(abs_jd, format="jd", scale="tdb")
            self._t_ref_jd = jnp.array(abs_jd)
            self._state = state.replace(
                relative_time=jnp.array(0.0),
                time_reference=self._t_ref_jd,
            )

        if self._is_keplerian:
            self.gravity = "keplerian"
            self._integrator_state = None
            self._integrator = None
            self._integrator_method = "keplerian"
            self._max_step_size = None
        else:
            self.gravity = self._setup_acceleration_func(gravity)
            self._integrator_state, self._integrator = self._setup_integrator(
                integrator, max_step_size
            )
            self._integrator_method = integrator
            self._max_step_size = max_step_size

    def __repr__(self) -> str:
        """Return a string representation of the System."""
        return f"*************\njorbit System\n time: {Time(self._t_ref_jd, format='jd', scale='tdb').utc.iso}\n*************"

    @property
    def t_ref(self) -> Time:
        """Reference time (astropy Time, TDB) — the System's epoch.

        All JAX-visible times inside the System are offsets in days from this
        reference, which keeps the integrator's internal arithmetic
        well-conditioned at decadal timescales.
        """
        return self._t_ref_astropy

    @property
    def t_ref_jd(self) -> float:
        """Reference time as a float JD (TDB), matching ``t_ref``."""
        return self._t_ref_jd

    def _times_to_offsets(self, times: Time | jnp.ndarray) -> jnp.ndarray:
        """Convert user-provided query times to offsets from ``self._t_ref_astropy``.

        Astropy ``Time`` inputs preserve their internal jd1+jd2 high-precision pair
        through the subtraction, so the returned offsets retain sub-ns precision
        regardless of how far ``times`` is from ``t_ref``. Plain float/jnp inputs
        are assumed to be absolute JD (TDB) and are subtracted in float64 — this
        is Sterbenz-exact when the magnitudes match, but the offset precision is
        capped at ulp(JD) ≈ 40 μs (≈1 m for typical solar system velocities).
        """
        if isinstance(times, Time):
            return jnp.asarray((times.tdb - self._t_ref_astropy).to_value(u.day))
        return jnp.asarray(times) - self._t_ref_jd

    @staticmethod
    def _resolve_step_scheduler(name: str) -> Callable:
        """Translate a string name to a JIT-friendly Partial step scheduler."""
        if name.lower() == "prs23":
            return jax.tree_util.Partial(next_proposed_dt_PRS23)
        elif name.lower() == "global":
            return jax.tree_util.Partial(next_proposed_dt_global)
        raise ValueError(
            f"Unknown step_scheduler '{name}'. Choices are 'prs23' or 'global'."
        )

    def _setup_acceleration_func(self, gravity: str | Callable) -> Callable:

        if isinstance(gravity, jax.tree_util.Partial):
            # See Particle._setup_acceleration_func for the full rationale.
            # CONTRACT: the custom function must recover the absolute JD from the
            # state itself as inputs.relative_time + inputs.time_reference (what
            # jorbit's own factory functions now do). The integrator hands it a
            # state carrying time_reference == self._t_ref_jd, so no shim is needed.
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
            # newtonian_gravity / ppn_gravity don't read the state's time at all,
            # so they're used directly with no ephemeris lookup.
            acc_func = jax.tree_util.Partial(newtonian_gravity)

        elif gravity == "generic gr":
            acc_func = jax.tree_util.Partial(ppn_gravity)
        else:
            raise ValueError(
                f"Unrecognized gravity '{gravity}'. Valid options are: 'newtonian planets', "
                "'newtonian solar system', 'gr planets', 'gr solar system', "
                "'default solar system', 'generic newtonian', 'generic gr', 'keplerian'. "
                "For a custom acceleration function, pass a jax.tree_util.Partial."
            )

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

    def integrate(
        self,
        times: Time,
        step_scheduler: str = "prs23",
        return_steps: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Integrate the System to a given time.

        Note: This method does not change the state of the system. It returns the
        positions and velocities at the given times, but the system itself is not
        changed.

        Args:
            times (Time | jnp.ndarray):
                The times to integrate to. Can be a single time or an array of times.
                If provided as a jnp.array, the entries are assumed to be in TDB JD.
            step_scheduler (str):
                The scheduler to use for determining step sizes. Choices are "prs23",
                which uses the PRS23 controller from Pham+ 2023, or "global", which
                uses the controller from the original IAS15 paper. Default is "prs23".
                Ignored for leapfrog integrators and keplerian systems.
            return_steps (bool):
                If True, also return the total number of integration steps taken (summed
                across any stitched interpolation chunks), appended as the final element
                of the returned tuple. ``None`` for the analytic Keplerian and fixed-step
                leapfrog paths, which cannot truncate. Defaults to False.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]:
                The positions of the particle at the given times, in AU, and the
                The velocities of the particle at the given times, in AU/day. If
                return_steps is True, also the total integration step count.
        """
        times = self._times_to_offsets(times)
        if times.shape == ():
            times = jnp.array([times])

        if self._is_keplerian:
            positions, velocities = _keplerian_system_integrate(
                self._state.tracer_positions,
                self._state.tracer_velocities,
                self._state.relative_time,
                times,
            )
            steps = None
        elif self._integrator_method in ["Y4", "Y6", "Y8"]:
            # Leapfrog pre-expands its (uncapped) time array, so it never truncates.
            times, inds = create_leapfrog_times(
                t0=self._state.relative_time,
                times=times,
                biggest_allowed_dt=self._integrator_state.dt,
            )
            scheduler = self._resolve_step_scheduler(step_scheduler)
            positions, velocities, _fss, _fis = _integrate(
                times,
                self._state,
                self.gravity,
                self._integrator,
                self._integrator_state,
                inds,
                scheduler,
            )
            steps = None
        else:
            # IAS15: stitch dense-output chunks so the 15k buffer can't silently
            # truncate. See jorbit.integrators.budgeted.
            scheduler = self._resolve_step_scheduler(step_scheduler)
            positions, velocities, steps = stitched_interpolate(
                self._state,
                self.gravity,
                times,
                self._integrator_state,
                scheduler,
            )

        if return_steps:
            return positions, velocities, steps
        return positions, velocities

    def ephemeris(
        self,
        times: Time | jnp.ndarray,
        observer: str | jnp.ndarray,
        step_scheduler: str = "prs23",
        return_steps: bool = False,
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
            step_scheduler (str):
                The scheduler to use for determining step sizes. Choices are "prs23",
                which uses the PRS23 controller from Pham+ 2023, or "global", which
                uses the controller from the original IAS15 paper. Default is "prs23".
                Ignored for leapfrog integrators and keplerian systems.
            return_steps (bool):
                If True, return a tuple of the SkyCoord and the total number of
                integration steps taken (summed across any stitched interpolation
                chunks). ``None`` for the analytic Keplerian and fixed-step leapfrog
                paths, which cannot truncate. Defaults to False.

        Returns:
            coords (SkyCoord):
                The ephemeris of each particle in the system at the given times, in ICRS
                coordinates and as seen from that specific observer. Each particle has
                its own light travel time correction applied individually. If
                return_steps is True, returns ``(coords, steps)``.
        """
        if isinstance(observer, str):
            observer_positions = get_observer_positions(
                times, observer, self._de_ephemeris_version
            )
        else:
            observer_positions = observer

        times = self._times_to_offsets(times)
        if times.shape == ():
            times = jnp.array([times])

        if self._is_keplerian:
            ras, decs = _keplerian_system_ephem(
                self._state.tracer_positions,
                self._state.tracer_velocities,
                self._state.relative_time,
                times,
                observer_positions,
            )
            coords = SkyCoord(ra=ras, dec=decs, unit=u.rad, frame="icrs")
            if return_steps:
                return coords, None
            return coords

        if self._integrator_method in ["Y4", "Y6", "Y8"]:
            # For leapfrog, need to create intermediate times. Leapfrog has no
            # b-coefficients; fall back to the original constant-acceleration Taylor.
            times, inds = create_leapfrog_times(
                t0=self._state.relative_time,
                times=times,
                biggest_allowed_dt=self._integrator_state.dt,
            )
            scheduler = self._resolve_step_scheduler(step_scheduler)
            ras, decs = _ephem(
                times,
                self._state,
                self.gravity,
                self._integrator,
                self._integrator_state,
                observer_positions,
                inds,
                scheduler,
            )
            steps = None
        else:
            # IAS15: stitch dense-output chunks (truncation-proof) and use the
            # b-coefficients for each particle's light-travel-time correction.
            inds = jnp.arange(times.shape[0])
            scheduler = self._resolve_step_scheduler(step_scheduler)
            ras, decs, steps = _ephem_ias15_stitched(
                times,
                self._state,
                self.gravity,
                self._integrator_state,
                observer_positions,
                inds,
                scheduler,
            )

        coords = SkyCoord(ra=ras, dec=decs, unit=u.rad, frame="icrs")
        if return_steps:
            return coords, steps
        return coords

"""The Particle class.

The jitted/host helper functions that implement each integration branch live in sibling
modules of this subpackage:

- :mod:`jorbit.particle.ephem` (generic integrate/ephem, leapfrog)
- :mod:`jorbit.particle.ias15_dense` (IAS15 ``interpolate=True``)
- :mod:`jorbit.particle.ias15_forced` (IAS15 ``interpolate=False``)
- :mod:`jorbit.particle.keplerian` (analytic two-body path)
- :mod:`jorbit.particle.covariance` (shared forward-mode-AD covariance leaves)
- :mod:`jorbit.particle.likelihood` (residuals/log-likelihood for fitting)
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from copy import deepcopy

import astropy.units as u
import jax
import jax.numpy as jnp
from astropy.coordinates import SkyCoord
from astropy.time import Time
from scipy.optimize import minimize

from jorbit import Observations
from jorbit.accelerations import (
    create_default_ephemeris_acceleration_func,
    create_gr_ephemeris_acceleration_func,
    create_newtonian_ephemeris_acceleration_func,
)
from jorbit.astrometry.orbit_fit_seeds import gauss_method_orbit, simple_circular
from jorbit.data.constants import (
    SPEED_OF_LIGHT,
    Y4_C,
    Y4_D,
    Y6_C,
    Y6_D,
    Y8_C,
    Y8_D,
)
from jorbit.ephemeris.ephemeris import Ephemeris
from jorbit.integrators import (
    budgeted_forced_landing,
    create_leapfrog_times,
    ias15_evolve,
    ias15_evolve_forced_landing,
    ias15_span_probe,
    initialize_ias15_integrator_state,
    leapfrog_evolve,
    next_proposed_dt_global,
    next_proposed_dt_PRS23,
    stitched_interpolate,
)
from jorbit.integrators.budgeted import _loaded_ephemeris_bounds_jd
from jorbit.likelihoods.setup_static_likelihood import (
    create_default_static_residuals_func,
    precompute_likelihood_data,
)
from jorbit.particle.ephem import _ephem, _ephem_with_cov, _integrate
from jorbit.particle.ias15_dense import _ephem_ias15_stitched, _ephem_ias15_with_cov
from jorbit.particle.ias15_forced import _ephem_forced_budgeted
from jorbit.particle.keplerian import (
    _keplerian_ephem,
    _keplerian_ephem_with_cov,
    _keplerian_integrate,
    _keplerian_loglike,
    _keplerian_residuals,
)
from jorbit.particle.likelihood import _loglike, _residuals
from jorbit.utils.horizons import get_observer_positions, horizons_bulk_vector_query
from jorbit.utils.states import (
    CartesianState,
    IAS15IntegratorState,
    KeplerianState,
    LeapfrogIntegratorState,
)


class Particle:
    """An object representing a single particle in the solar system.

    This class is used to represent and manipulate a single particle moving within the
    solar system. It is mostly a collection of convenience wrappers around the more
    general integrators and accelerations, but it also provides some useful methods for
    projecting the particle's position onto the sky and fitting orbits to observations.

    By construction, `Particle` objects are massless.

    Note: none of the methods associated with this class will alter the underlying state
    of the particle. For example, "integrate" will give you the positions and velocities
    of the particle at future times, but after it returns, the particle will still be
    at its original state.

    Attributes:
        state: The state of the particle, either in Cartesian or Keplerian coordinates.
        time: The time of the particle's state.
        x: The position of the particle in Cartesian coordinates.
        v: The velocity of the particle in Cartesian coordinates.
        observations: A collection of observations of the particle.
        name: The name of the particle.
        gravity: The gravitational acceleration function to use for the particle.
        integrator: The integrator to use for the particle.
        earliest_time: The earliest time for which ephemeris data is available.
        latest_time: The latest time for which ephemeris data is available.
        fit_seed: A seed for fitting the orbit of the particle.
        static_residuals: Fast residuals via the pre-computed "static" pipeline, or
            None when unavailable (no observations, or gravity="keplerian"). **This
            path always uses the DE440 "default solar system" dynamical model,
            regardless of the `gravity` and `de_ephemeris_version` passed to
            __init__.** The static integrator currently has exactly one acceleration
            model implemented, so unlike `integrate`/`residuals`/`loglike` it cannot
            follow the requested one. Use `residuals` if you need the model you asked
            for.
    """

    def __init__(
        self,
        state: KeplerianState | CartesianState | None = None,
        time: Time | None = None,
        x: jnp.ndarray | None = None,
        v: jnp.ndarray | None = None,
        observations: Observations | None = None,
        name: str = "",
        gravity: str | Callable = "default solar system",
        de_ephemeris_version: str | None = "440",
        integrator: str = "ias15",
        earliest_time: Time = Time("1980-01-01"),
        latest_time: Time = Time("2050-01-01"),
        fit_seed: KeplerianState | CartesianState | None = None,
        max_step_size: u.Quantity | None = None,
        step_scheduler: str = "prs23",
        ias15_max_steps: int | None = None,
    ) -> None:
        """Initialize a Particle object.

        Args:
            state (KeplerianState | CartesianState | None):
                The state of the particle. None if x and v are provided.
            time (Time | None):
                The time of the particle's state. None if state is provided, since that
                will have its own time baked-in.
            x (jnp.ndarray | None):
                The 3D barycentric cartesian position of the particle in AU. None if
                state is provided.
            v (jnp.ndarray | None):
                The 3D barycentric cartesian velocity of the particle in AU/day. None if
                state is provided.
            observations (Observations | None):
                Optional Observations associated with the particle. Necessary if fitting
                or evaluating likelihoods.
            name (str):
                The name of the particle. Defaults to "".
            gravity (str | Callable):
                The gravitational acceleration function to use when integrating the
                particle's orbit. Defaults to "default solar system", which corresponds
                to parameterized post-Newtonian interactions with the 10 bodies in the
                JPL DE440 ephemeris, plus Newtonian interactions with the 16 largest
                asteroids in the asteroids_de441/sb441-n16.bsp ephemeris. Can also be
                a jax.tree_util.Partial object that follows the same signature as the
                acceleration functions in jorbit.accelerations. Other string options are
                "newtonian planets", "newtonian solar system", "gr planets", "gr
                solar system", and "keplerian" (pure 2-body Keplerian propagation with
                no ephemeris or integrator setup). Note this does *not* affect the
                `static_residuals` attribute, which always uses the "default solar
                system" model.
            de_ephemeris_version (str | None):
                Which version of the JPL DE ephemeris to use for perturber positions
                when using one of the built-in gravity models. Accepts either "440" or
                "430", default is "440". Ignored when gravity is "keplerian", and does
                *not* affect the `static_residuals` attribute, which always uses DE440.
            integrator (str):
                The integrator to use for the particle. Choices are "ias15", which is a
                15th order adaptive step-size integrator, or "Y4", "Y6", or "Y8", which
                are 4th, 6th, and 8th order Yoshida leapfrog integrators with fixed
                step sizes. Defaults to "ias15". Ignored when gravity is "keplerian".
            earliest_time (Time):
                The earliest time we expect to integrate the particle to. Defaults to
                Time("1980-01-01"). Larger time windows will result in larger in-memory
                ephemeris objects.
            latest_time (Time):
                The latest time we expect to integrate the particle to. Defaults to
                Time("2050-01-01"). Larger time windows will result in larger in-memory
                ephemeris objects.
            fit_seed (KeplerianState | CartesianState | None):
                A seed for fitting the orbit of the particle. If None, a seed will be
                generated from the observations if they exist. Otherwise, a circular
                orbit with semi-major axis 2.5 AU will be used.
            max_step_size (u.Quantity, optional):
                The fixed step size to use for leapfrog integrators. Required if
                integrator is "Y4", "Y6", or "Y8". Ignored if integrator is "ias15".
                Note that this is the maximum step size; the actual step size may be
                smaller to ensure that the particle lands exactly on the requested
                output times, and that the step size may change if the spacing between
                output times is not constant. Defaults to None.
            step_scheduler (str):
                The scheduler used by IAS15 for picking the next proposed step size.
                Choices are "prs23" (Pham+ 2023 controller, default) or "global"
                (the controller from the original IAS15 paper). Used consistently
                by ``integrate``, ``integrate_or_interpolate``, ``ephemeris``, and
                the residuals/loglike closures. Ignored when gravity is "keplerian"
                or for leapfrog integrators.
            ias15_max_steps (int | None):
                Depth of the IAS15 dense-output buffer used by ``integrate``,
                ``integrate_or_interpolate``, and ``ephemeris``: the maximum number
                of accepted adaptive steps a single integration chunk can capture.
                None (default) uses ``IAS15_MAX_DYNAMIC_STEPS`` (15000). Short arcs
                need only a few steps, so sizing this to the arc avoids allocating
                and zeroing the full-size buffers on every call. The nominal methods
                stitch extra chunks as needed, so a small value never truncates
                them; ``ephemeris(uncertainty=True)`` cannot stitch and instead
                raises if the span exceeds one buffer. Does not affect the
                residuals/loglike closures or ``max_likelihood``. Ignored when
                gravity is "keplerian" or for leapfrog integrators. Defaults to
                None.
        """
        self._observations = observations
        self._earliest_time = earliest_time
        self._latest_time = latest_time
        self._de_ephemeris_version = de_ephemeris_version
        self._is_keplerian = gravity == "keplerian"
        self._step_scheduler = self._resolve_step_scheduler(step_scheduler)
        self._ias15_max_steps = ias15_max_steps

        self.gravity = gravity
        # Preserve the original gravity spec (string or user-supplied Partial) so
        # max_likelihood can pass it to the result Particle rather than self.gravity.
        # self.gravity is overwritten by _setup_acceleration_func with a Partial that
        # already has t_ref_jd baked in; passing *that* Partial to a new Particle would
        # re-wrap it a second time and double-count the offset (see BUG_FIXES_MAY2026).
        self._gravity_str = gravity

        state = deepcopy(state) if state is not None else None

        # self._time is kept at jnp.array(0.0) internally; absolute time lives in
        # self._t_ref_astropy / self._t_ref_jd. All JAX-visible times are offsets
        # (in days) from _t_ref_jd, which keeps step-boundary and interpolation
        # arithmetic well-conditioned at decadal timescales.
        (
            self._x,
            self._v,
            self._time,
            self._t_ref_astropy,
            self._t_ref_jd,
            self._cartesian_state,
            self._keplerian_state,
            self._name,
            self._acc_func_kwargs,
        ) = self._setup_state(x, v, state, time, name)

        if self._is_keplerian:
            self.gravity = "keplerian"
            self._integrator_state = None
            self._integrator = None
            self._integrator_method = "keplerian"
            self._max_step_size = None
        else:
            self.gravity = self._setup_acceleration_func(gravity)
            self._integrator_state, self._integrator, self._forced_integrator = (
                self._setup_integrator(integrator, max_step_size)
            )
            self._integrator_method = integrator
            self._max_step_size = max_step_size

        self._fit_seed = self._setup_fit_seed(fit_seed)

        (
            self.residuals,
            self.loglike,
            self.scipy_objective,
            self.scipy_objective_grad,
        ) = self._setup_likelihood()

        self.static_residuals = self._setup_default_static_residuals()

    def __repr__(self) -> str:
        """Return a string representation of the Particle object."""
        return f"Particle: {self._name}"

    @property
    def cartesian_state(self) -> CartesianState:
        """Return the Cartesian state of the particle.

        The state is self-describing about its absolute epoch via the
        ``(relative_time, time_reference)`` pair: ``relative_time`` is ``0.0``
        (the offset in the particle's internal frame) and ``time_reference``
        is :attr:`t_ref_jd` (the absolute JD anchor).
        """
        return self._cartesian_state

    @property
    def keplerian_state(self) -> KeplerianState:
        """Return the Keplerian state of the particle.

        The state is self-describing about its absolute epoch via the
        ``(relative_time, time_reference)`` pair: ``relative_time`` is ``0.0``
        (the offset in the particle's internal frame) and ``time_reference``
        is :attr:`t_ref_jd` (the absolute JD anchor).
        """
        return self._keplerian_state

    @property
    def observations(self) -> Observations | None:
        """Return the observations associated with the particle."""
        return self._observations

    @property
    def t_ref(self) -> Time:
        """Reference time (astropy Time, TDB) — the particle's epoch.

        All JAX-visible times inside the Particle are offsets in days from this
        reference, which keeps the integrator's internal arithmetic
        well-conditioned at decadal timescales.
        """
        return self._t_ref_astropy

    @property
    def t_ref_jd(self) -> float:
        """Reference time as a float JD (TDB), matching ``t_ref``."""
        return self._t_ref_jd

    ###############
    # SETUP METHODS
    ###############

    def _times_to_offsets(self, times: Time | jnp.ndarray) -> jnp.ndarray:
        """Convert user-provided query times to offsets from ``self._t_ref_astropy``.

        Astropy ``Time`` inputs preserve their internal jd1+jd2 high-precision pair
        through the subtraction, so the returned offsets retain sub-ns precision
        regardless of how far ``times`` is from ``t_ref``. Plain float/jnp inputs
        are assumed to be absolute JD (TDB) and are subtracted in float64 — this
        is Sterbenz-exact when the magnitudes match, but the offset precision is
        capped at ulp(JD) ≈ 40 μs (≈1 m for typical solar system velocities).

        Also validates that the requested times — together with the reference
        epoch the integration marches from — stay inside the loaded ephemeris
        window. Outside it the Chebyshev interval lookup wraps to wrong-era
        coefficients and returns smooth garbage rather than failing, so every
        public method funnels its times through here to fail loudly instead.
        """
        if isinstance(times, Time):
            offsets = jnp.asarray((times.tdb - self._t_ref_astropy).to_value(u.day))
        else:
            offsets = jnp.asarray(times) - self._t_ref_jd
        bounds = _loaded_ephemeris_bounds_jd(self.gravity)
        if bounds is not None:
            t_ref = float(self._t_ref_jd)
            lo = min(t_ref + float(jnp.min(offsets)), t_ref)
            hi = max(t_ref + float(jnp.max(offsets)), t_ref)
            if lo < bounds[0] or hi > bounds[1]:
                raise ValueError(
                    f"Requested times span JD {lo:.2f} to JD {hi:.2f} (TDB, "
                    "including the reference epoch), which extends outside the "
                    f"loaded ephemeris window (JD {bounds[0]:.2f} to "
                    f"JD {bounds[1]:.2f}). Beyond that window the perturber "
                    "positions silently come from wrong-era Chebyshev intervals, "
                    "so integrations there return garbage. Rebuild the Particle "
                    "with `earliest_time`/`latest_time` covering the requested "
                    "times."
                )
        return offsets

    def _validate_state_epoch(self, state: CartesianState | KeplerianState) -> None:
        """Guard against a ``state=`` whose epoch disagrees with this particle's.

        Query times handed to ``integrate``/``ephemeris`` are offsets from
        ``self._t_ref_jd``, so a supplied ``state`` must share that anchor: the
        integrator marches ``relative_time`` and the acceleration function adds
        ``state.time_reference`` to recover the absolute JD. If the two disagree
        the ephemeris would be queried at the wrong epoch (potentially years off)
        while still "looking" plausible, so we fail loudly instead.
        """
        if abs(float(state.time_reference) - float(self._t_ref_jd)) > 1e-6:
            raise ValueError(
                "The provided state's time_reference "
                f"({float(state.time_reference)}) does not match this particle's "
                f"reference epoch ({float(self._t_ref_jd)}). States passed via "
                "`state=` must be expressed in the particle's frame "
                "(time_reference == particle.t_ref_jd, with relative_time the offset "
                "in days). Build the state at the particle's epoch, or create a new "
                "Particle at the desired epoch."
            )

    def _observations_times_as_offsets(self) -> jnp.ndarray:
        """Return the observation times as offsets from ``self._t_ref_astropy``.

        Prefers the full-precision astropy Time stored on the Observations object
        when available (sub-ns precision preserved). Falls back to subtracting
        :attr:`t_ref_jd` from the float-JD array, which is Sterbenz-exact but
        inherits the ulp(JD) ≈ 40 μs quantization of the stored float-JD.
        Delegates to :meth:`_times_to_offsets` (identical arithmetic per branch)
        so observation times get the same ephemeris-window validation.
        """
        obs = self._observations
        times_astropy = getattr(obs, "_times_astropy", None)
        if times_astropy is not None:
            return self._times_to_offsets(times_astropy)
        return self._times_to_offsets(jnp.asarray(obs.times))

    def _setup_state(
        self,
        x: jnp.ndarray | None,
        v: jnp.ndarray | None,
        state: CartesianState | KeplerianState | None,
        time: Time,
        name: str,
    ) -> tuple:
        if state is not None:
            assert time is None, "Cannot provide both state and time"
            # The state self-describes its absolute epoch as the sum of its
            # offset (relative_time) and its anchor (time_reference). We use
            # the sum as the Particle's absolute reference time, then rebase
            # the stored state to relative_time=0 against that new anchor.
            time = state.relative_time + state.time_reference

        assert time is not None, "Must provide an epoch for the particle"

        # Build the reference time pair (astropy Time at full precision + float JD)
        # and then set the Particle's internal epoch to 0.0 in the offset frame.
        if isinstance(time, Time):
            t_ref_astropy = time.tdb
        else:
            t_ref_astropy = Time(float(time), format="jd", scale="tdb")
        t_ref_jd = jnp.array(float(t_ref_astropy.tdb.jd))
        internal_time = jnp.array(0.0)

        if state is not None:
            assert x is None and v is None, "Cannot provide both state and x, v"

            # to_keplerian() and to_cartesian() are documented not to propagate `cov`
            # because the covariance is parameterisation-specific. Save and re-attach
            # it to the same-type output state so callers who set cov on the input
            # state can retrieve it on particle.cartesian_state / particle.keplerian_state.
            _input_is_keplerian = isinstance(state, KeplerianState)
            _input_cov = state.cov

            state = state.to_cartesian()
            if state.x.ndim != 2:
                state.x = state.x[None, :]
                state.v = state.v[None, :]
            state.relative_time = internal_time
            state.time_reference = t_ref_jd
            keplerian_state = state.to_keplerian()
            cartesian_state = state.to_cartesian()

            if _input_is_keplerian:
                keplerian_state.cov = _input_cov
            else:
                cartesian_state.cov = _input_cov

            x = state.x.flatten()
            v = state.v.flatten()

        elif x is not None:
            assert v is not None, "Must provide both x and v"

            x = x.flatten()
            v = v.flatten()
            cartesian_state = CartesianState(
                x=jnp.array([x]),
                v=jnp.array([v]),
                relative_time=internal_time,
                time_reference=t_ref_jd,
                acceleration_func_kwargs={"c2": SPEED_OF_LIGHT**2},
            )
            keplerian_state = cartesian_state.to_keplerian()

        else:
            raise ValueError(
                "time must be either astropy.time.Time or float (interpreted as JD in"
                " TDB)"
            )

        if name == "":
            name = "unnamed"

        acc_func_kwargs = cartesian_state.acceleration_func_kwargs
        return (
            x,
            v,
            internal_time,
            t_ref_astropy,
            t_ref_jd,
            cartesian_state,
            keplerian_state,
            name,
            acc_func_kwargs,
        )

    def _setup_acceleration_func(self, gravity: str) -> Callable:
        if isinstance(gravity, jax.tree_util.Partial):
            # CONTRACT: the custom function must recover the absolute JD from the
            # state itself, as ``inputs.relative_time + inputs.time_reference`` —
            # exactly what jorbit's own factory functions
            # (create_default_ephemeris_acceleration_func, etc.) now do. The state
            # the integrator hands to the function carries
            # ``time_reference == self._t_ref_jd`` and a small ``relative_time``
            # offset, so no wrapper/shim is needed: the state is self-describing.
            return gravity

        assert self._de_ephemeris_version in ["440", "430"], (
            "de_ephemeris_version must be either '440' or '430' if not using a custom "
            "gravity function"
        )

        if gravity == "newtonian planets":
            eph = Ephemeris(
                earliest_time=self._earliest_time,
                latest_time=self._latest_time,
                ssos="default planets",
                de_ephemeris_version=self._de_ephemeris_version,
            )
            acc_func = create_newtonian_ephemeris_acceleration_func(eph.processor)
        elif gravity == "newtonian solar system":
            eph = Ephemeris(
                earliest_time=self._earliest_time,
                latest_time=self._latest_time,
                ssos="default solar system",
                de_ephemeris_version=self._de_ephemeris_version,
            )
            acc_func = create_newtonian_ephemeris_acceleration_func(eph.processor)
        elif gravity == "gr planets":
            eph = Ephemeris(
                earliest_time=self._earliest_time,
                latest_time=self._latest_time,
                ssos="default planets",
                de_ephemeris_version=self._de_ephemeris_version,
            )
            acc_func = create_gr_ephemeris_acceleration_func(eph.processor)
        elif gravity == "gr solar system":
            eph = Ephemeris(
                earliest_time=self._earliest_time,
                latest_time=self._latest_time,
                ssos="default solar system",
                de_ephemeris_version=self._de_ephemeris_version,
            )
            acc_func = create_gr_ephemeris_acceleration_func(eph.processor)
        elif gravity == "default solar system":
            eph = Ephemeris(
                earliest_time=self._earliest_time,
                latest_time=self._latest_time,
                ssos="default solar system",
                de_ephemeris_version=self._de_ephemeris_version,
            )
            acc_func = create_default_ephemeris_acceleration_func(eph.processor)
        else:
            raise ValueError(
                f"Unrecognized gravity '{gravity}'. Valid options are: 'newtonian planets', "
                "'newtonian solar system', 'gr planets', 'gr solar system', "
                "'default solar system', 'keplerian'. For a custom acceleration function, "
                "pass a jax.tree_util.Partial."
            )

        return acc_func

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

    def _setup_integrator(
        self, integrator: str, max_step_size: u.Quantity | None
    ) -> tuple[IAS15IntegratorState | LeapfrogIntegratorState, Callable]:
        if integrator == "ias15":
            assert (
                max_step_size is None
            ), "max_step_size should not be provided for IAS15 integrator."

            a0 = self.gravity(self._cartesian_state.to_system())
            integrator_state = initialize_ias15_integrator_state(a0)
            integrator = jax.tree_util.Partial(ias15_evolve)
            forced_integrator = jax.tree_util.Partial(ias15_evolve_forced_landing)
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
            forced_integrator = integrator

        return integrator_state, integrator, forced_integrator

    def _setup_fit_seed(
        self, fit_seed: KeplerianState | CartesianState | None
    ) -> KeplerianState | CartesianState | None:
        if self._observations is None:
            return None

        if isinstance(fit_seed, (CartesianState, KeplerianState)):
            return fit_seed

        if len(self._observations) >= 3:
            mean_time = jnp.mean(self._observations.times)
            mid_idx = jnp.argmin(jnp.abs(self._observations.times - mean_time))
            fit_seed = gauss_method_orbit(
                self._observations[0]
                + self._observations[mid_idx]
                + self._observations[-1]
            )
            if fit_seed.to_keplerian().ecc > 1:
                warnings.warn(
                    "Warning: initial Gauss's method fit produced an unbound orbit",
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            # Pass the Particle's reference epoch (not the observation time)
            # so that the returned state's x, v represent the object at
            # self._t_ref_jd — the same epoch the optimizer will use as its
            # starting point. Using obs.times[0] instead would yield x, v at
            # a different epoch that the optimizer would then misinterpret.
            fit_seed = simple_circular(
                self._observations.ra[0],
                self._observations.dec[0],
                semi=2.5,
                time=self._t_ref_jd,
            )

        return fit_seed

    def _setup_likelihood(self) -> tuple[Callable, Callable, Callable, Callable]:
        if self._observations is None:
            return None, None, None, None

        obs_times = self._observations_times_as_offsets()

        if self._is_keplerian:
            times = obs_times

            residuals = jax.tree_util.Partial(
                _keplerian_residuals,
                times,
                self._observations.observer_positions,
                self._observations.ra,
                self._observations.dec,
            )

            ll = jax.tree_util.Partial(
                _keplerian_loglike,
                times,
                self._observations.observer_positions,
                self._observations.ra,
                self._observations.dec,
                self._observations.inv_cov_matrices,
                self._observations.cov_log_dets,
            )

            # Keplerian propagation is fully JIT-able, reverse mode works natively.
            # _keplerian_residuals/_keplerian_loglike are module-level jitted
            # functions, so the Partial-bound data above flows in as jit arguments;
            # re-wrapping in jax.jit here would bake it into the executables as
            # per-instance constants instead.
            loglike = ll

        else:
            if self._integrator_method == "ias15":
                times = obs_times
                inds = jnp.arange(len(obs_times))

            elif self._integrator_method in ["Y4", "Y6", "Y8"]:
                times, inds = create_leapfrog_times(
                    self._cartesian_state.relative_time,
                    obs_times,
                    self._max_step_size,
                )

            residuals = jax.tree_util.Partial(
                _residuals,
                times,
                self.gravity,
                self._integrator,
                self._integrator_state,
                self._observations.observer_positions,
                self._observations.ra,
                self._observations.dec,
                inds,
                step_scheduler=self._step_scheduler,
            )

            ll = jax.tree_util.Partial(
                _loglike,
                times,
                self.gravity,
                self._integrator,
                self._integrator_state,
                self._observations.observer_positions,
                self._observations.ra,
                self._observations.dec,
                self._observations.inv_cov_matrices,
                self._observations.cov_log_dets,
                inds,
                step_scheduler=self._step_scheduler,
            )

            # since we've gone with the while loop version of the ias15 integrator,
            # can no longer use reverse mode. But, actually specifying forward mode
            # everywhere is annoying, so we're going to re-define a custom vjp for
            # "reverse" mode that's actually just forward mode

            @jax.custom_vjp
            def loglike(params: CartesianState | KeplerianState) -> float:
                return ll(params)

            def loglike_fwd(params: CartesianState | KeplerianState) -> tuple:
                output = ll(params)
                jac = jax.jacfwd(ll)(params)
                return output, (jac,)

            def loglike_bwd(res: tuple, g: float) -> float:
                jac = res
                val = jax.tree.map(lambda x: x * g, jac)
                return val

            loglike.defvjp(loglike_fwd, loglike_bwd)

            # Deliberately NOT wrapped in jax.jit: _residuals/_loglike are already
            # module-level jitted functions, so the Partial-bound data (ephemeris,
            # observations, integrator state) flows into shared compilations as
            # traced arguments. Re-wrapping per instance would re-embed all of it
            # in fresh executables as compile-time constants (~150 MB per compile,
            # never freed while the Particle lives) — the cause of a CI OOM.

        def scipy_objective(x: jnp.ndarray) -> float:
            c = CartesianState(
                x=jnp.array([x[:3]]),
                v=jnp.array([x[3:]]),
                relative_time=self._time,
                time_reference=self._t_ref_jd,
                acceleration_func_kwargs=self._acc_func_kwargs,
            )
            return -loglike(c)

        def scipy_grad(x: jnp.ndarray) -> jnp.ndarray:
            c = CartesianState(
                x=jnp.array([x[:3]]),
                v=jnp.array([x[3:]]),
                relative_time=self._time,
                time_reference=self._t_ref_jd,
                acceleration_func_kwargs=self._acc_func_kwargs,
            )
            c_grad = jax.grad(loglike)(c)
            g = jnp.concatenate([c_grad.x.flatten(), c_grad.v.flatten()])
            return -g

        return residuals, loglike, scipy_objective, scipy_grad

    def _setup_default_static_residuals(self) -> Callable:
        if self._observations is None or self._is_keplerian:
            return None
        precomputed_data = precompute_likelihood_data(self, self._step_scheduler)
        static_residuals_func = create_default_static_residuals_func(precomputed_data)
        return static_residuals_func

    ################
    # PUBLIC METHODS
    ################

    @classmethod
    def from_horizons(
        cls,
        name: str,
        time: Time,
        observations: Observations | None = None,
        gravity: str | Callable = "default solar system",
        integrator: str = "ias15",
        earliest_time: Time = Time("1980-01-01"),
        latest_time: Time = Time("2050-01-01"),
        fit_seed: KeplerianState | CartesianState | None = None,
        max_step_size: u.Quantity | None = None,
        de_ephemeris_version: str | None = "440",
    ) -> Particle:
        """Query JPL Horizons for an SSOs state at a given time and create a Particle object.

        Args:
            name (str):
                The name of the SSO to query. Can be a string or an integer.
            time (Time):
                The time to query the SSO at.
            observations (Observations | None):
                Optional Observations associated with the particle. Necessary if fitting
                or evaluating likelihoods.
            gravity (str | Callable):
                The gravitational acceleration function to use when integrating the
                particle's orbit. Defaults to "default solar system", which corresponds
                to parameterized post-Newtonian interactions with the 10 bodies in the
                JPL DE440 ephemeris, plus Newtonian interactions with the 16 largest
                asteroids in the asteroids_de441/sb441-n16.bsp ephemeris. Can also be
                a jax.tree_util.Partial object that follows the same signature as the
                acceleration functions in jorbit.accelerations. Other string options are
                "newtonian planets", "newtonian solar system", "gr planets", and "gr
                solar system".
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
            fit_seed (KeplerianState | CartesianState | None):
                A seed for fitting the orbit of the particle. If None, a seed will be
                generated from the observations if they exist. Otherwise, a circular
                orbit with semi-major axis 2.5 AU will be used.
            max_step_size (u.Quantity, optional):
                The fixed step size to use for leapfrog integrators. Required if
                integrator is "Y4", "Y6", or "Y8". Ignored if integrator is "ias15".
                Note that this is the maximum step size; the actual step size may be
                smaller to ensure that the particle lands exactly on the requested
                output times, and that the step size may change if the spacing between
                output times is not constant. Defaults to None.
            de_ephemeris_version (str | None):
                Which version of the JPL DE ephemeris to use for perturber positions.
                When using `from_horizons` to pull an initial state, only DE440 is
                supported. Defaults to "440", will error on anything else as a
                safeguard.

        Returns:
            Particle:
                A Particle object representing the SSO at the given time.
        """
        if de_ephemeris_version != "440":
            raise ValueError(
                "Only DE440 ephemeris version is supported when pulling "
                "an initial state from JPL Horizons."
            )
        data = horizons_bulk_vector_query(target=name, center="500@0", times=time)
        x0 = jnp.array([data["x"][0], data["y"][0], data["z"][0]])
        v0 = jnp.array([data["vx"][0], data["vy"][0], data["vz"][0]])

        return cls(
            x=x0,
            v=v0,
            time=time,
            observations=observations,
            name=name,
            gravity=gravity,
            integrator=integrator,
            earliest_time=earliest_time,
            latest_time=latest_time,
            fit_seed=fit_seed,
            max_step_size=max_step_size,
        )

    def _integrate_base(
        self,
        times: Time,
        state: CartesianState | KeplerianState | None = None,
        forced_landing: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if state is not None:
            self._validate_state_epoch(state)

        if self._is_keplerian:
            if state is not None:
                state = state.to_cartesian()
                # relative_time is already in this Particle's offset frame
                # (validated above: state.time_reference == self._t_ref_jd).
                x, v, t0 = (
                    state.x.flatten(),
                    state.v.flatten(),
                    state.relative_time,
                )
            else:
                x, v, t0 = self._x, self._v, self._time

            times = self._times_to_offsets(times)
            if times.shape == ():
                times = jnp.array([times])

            return *_keplerian_integrate(x, v, t0, times), None

        if state is None:
            state = self._cartesian_state
            integrator_state = self._integrator_state
        else:
            a0 = self.gravity(state.to_system())
            integrator_state = initialize_ias15_integrator_state(a0)

        times = self._times_to_offsets(times)
        if times.shape == ():
            times = jnp.array([times])

        if self._integrator_method in ["Y4", "Y6", "Y8"]:
            # Leapfrog pre-expands its (uncapped) time array, so it never truncates.
            times, inds = create_leapfrog_times(
                state.relative_time, times, self._max_step_size
            )
            integrator = self._forced_integrator if forced_landing else self._integrator
            positions, velocities, _fss, _fis, steps = _integrate(
                times,
                state,
                self.gravity,
                integrator,
                integrator_state,
                inds,
                self._step_scheduler,
            )
            return positions[:, 0, :], velocities[:, 0, :], steps

        # IAS15: orchestrate on the host so neither the dense-output buffer
        # (interpolation) nor the per-interval cap (forced landing) can silently
        # truncate. See jorbit.integrators.budgeted.
        sys_state = state.to_system()
        if forced_landing:
            positions, velocities, steps = budgeted_forced_landing(
                sys_state, self.gravity, times, integrator_state, self._step_scheduler
            )
        else:
            positions, velocities, steps = stitched_interpolate(
                sys_state,
                self.gravity,
                times,
                integrator_state,
                self._step_scheduler,
                self._ias15_max_steps,
            )
        return positions[:, 0, :], velocities[:, 0, :], steps

    def integrate(
        self,
        times: Time,
        state: CartesianState | KeplerianState | None = None,
        return_steps: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Integrate the particle's orbit to specified times, landing exactly on each one.

        Note that this method does not change the state of the particle. It returns the
        positions and velocities of the particle at the given times, but the particle
        itself is not changed.

        Args:
            times (Time | jnp.ndarray):
                The times to integrate to. Can be a single time or an array of times.
                If provided as a jnp.array, the entries are assumed to be in TDB JD.
            state (CartesianState | None):
                The state to integrate from. If None, the particle's current state will
                be used. Usually not necessary to provide this.
            return_steps (bool):
                Whether to return the number of steps taken to reach each output time.
                If True, the method returns a tuple of (positions, velocities, steps).
                If False, only returns (positions, velocities). Defaults to False.


        Returns:
            tuple[jnp.ndarray, jnp.ndarray]:
                The positions of the particle at the given times, in AU, and the
                The velocities of the particle at the given times, in AU/day. If
                return_steps is True, also returns an array of the number of steps taken
                to reach each output time.
        """
        x, v, steps = self._integrate_base(times, state, forced_landing=True)
        if return_steps:
            return x, v, steps
        return x, v

    def integrate_or_interpolate(
        self,
        times: Time,
        state: CartesianState | KeplerianState | None = None,
        return_steps: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Integrate the particle's orbit to specified times, overshooting and 'interpolating' if necessary.

        Note that this method does not change the state of the particle. It returns the
        positions and velocities of the particle at the given times, but the particle
        itself is not changed.

        Args:
            times (Time | jnp.ndarray):
                The times to integrate to. Can be a single time or an array of times.
                If provided as a jnp.array, the entries are assumed to be in TDB JD.
            state (CartesianState | None):
                The state to integrate from. If None, the particle's current state will
                be used. Usually not necessary to provide this.
            return_steps (bool):
                Whether to return the number of steps taken to reach each output time.
                If True, the method returns a tuple of (positions, velocities, steps).
                If False, only returns (positions, velocities). Defaults to False.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]:
                The positions of the particle at the given times, in AU, and the
                The velocities of the particle at the given times, in AU/day. If
                return_steps is True, also returns an array of the number of steps taken
                to reach each output time.
        """
        x, v, steps = self._integrate_base(times, state, forced_landing=False)
        if return_steps:
            return x, v, steps
        return x, v

    def ephemeris(
        self,
        times: Time,
        observer: str | jnp.ndarray,
        state: CartesianState | KeplerianState | None = None,
        interpolate: bool = True,
        uncertainty: bool = False,
        return_steps: bool = False,
    ) -> SkyCoord | tuple:
        """Compute an ephemeris for the particle.

        Args:
            times (Time | jnp.ndarray):
                The times to compute the ephemeris for. Can be a single time or an array
                of times. If provided as a jnp.array, the entries are assumed to be in
                TDB JD.
            observer (str | jnp.ndarray):
                The observer to compute the ephemeris for. Can be a string representing
                an observatory name, or a 3D position vector in AU. For more info on
                acceptable strings, see the get_observer_positions function.
            state (CartesianState | None):
                The state to compute the ephemeris from. If None, the particle's current
                state will be used. Usually not necessary to provide this.
            interpolate (bool):
                Whether to use `integrate` or `integrate_or_interpolate` for the
                underlying integrations.
            uncertainty (bool):
                If True, also propagate the 6x6 covariance matrix stored on the state's
                ``cov`` field onto the sky plane via forward-mode autodiff (linear error
                propagation). The covariance is propagated in whichever parameterization
                the input state uses (Cartesian or Keplerian); no automatic conversion
                between parameterizations is performed. Expect roughly a ``6x`` cost
                relative to the nominal call because ``jax.jacfwd`` performs one JVP
                evaluation per input dimension. Defaults to False.
            return_steps (bool):
                If True, also return the total number of integration steps taken (summed
                across any stitched interpolation chunks / inserted forced-landing
                landings). Appended as the final element of the returned tuple. For the
                IAS15 integrator this is the figure to watch: the nominal ephemeris is
                truncation-proof, but the count reveals an unusually heavy integration.
                ``None`` for the analytic Keplerian and fixed-step leapfrog paths, which
                cannot truncate. Defaults to False.

        Returns:
            coords (SkyCoord | tuple):
                If ``uncertainty=False`` (default), returns a SkyCoord with the
                ephemeris of the particle at the given times, in ICRS coordinates, as
                seen from that specific observer and correcting for light travel time.
                If ``uncertainty=True``, returns a tuple with the same SkyCoord and the
                propagated ``(N, 2, 2)`` sky-plane covariance in ``arcsec**2``
                (axis order (RA, Dec)). If ``return_steps=True``, the step count is
                appended as the final tuple element.
        """
        if isinstance(observer, str):
            observer_positions = get_observer_positions(
                times, observer, self._de_ephemeris_version
            )
        else:
            observer_positions = observer

        if state is not None:
            self._validate_state_epoch(state)

        if uncertainty:
            cov_state = (
                state
                if state is not None
                else (
                    self._keplerian_state
                    if self._is_keplerian
                    else self._cartesian_state
                )
            )
            cov = cov_state.cov
            if cov.shape != (6, 6):
                raise ValueError(
                    "uncertainty=True requires a (6, 6) covariance matrix on the "
                    f"state's `cov` field, but got shape {tuple(cov.shape)}. "
                    "Build a CartesianState or KeplerianState with `cov=...` and pass "
                    "it via the `state=` keyword (or set it on the particle's "
                    "internal state)."
                )

        if self._is_keplerian:
            offsets = self._times_to_offsets(times)
            if offsets.shape == ():
                offsets = jnp.array([offsets])

            if uncertainty:
                kep_state = state if state is not None else self._keplerian_state
                ras, decs, cov_radec = _keplerian_ephem_with_cov(
                    kep_state, offsets, observer_positions, cov
                )
                coords = SkyCoord(ra=ras, dec=decs, unit=u.rad, frame="icrs")
                # Keplerian propagation is analytic: no integration steps to report.
                if return_steps:
                    return (coords, cov_radec, None)
                return (coords, cov_radec)

            if state is not None:
                state = state.to_cartesian()
                x, v, t0 = (
                    state.x.flatten(),
                    state.v.flatten(),
                    state.relative_time,
                )
            else:
                x, v, t0 = self._x, self._v, self._time

            ras, decs = _keplerian_ephem(x, v, t0, offsets, observer_positions)
            coords = SkyCoord(ra=ras, dec=decs, unit=u.rad, frame="icrs")
            if return_steps:
                return (coords, None)
            return coords

        if state is None:
            state = self._cartesian_state
            integrator_state = self._integrator_state
        else:
            a0 = self.gravity(state.to_system())
            integrator_state = initialize_ias15_integrator_state(a0)

        times = self._times_to_offsets(times)
        if times.shape == ():
            times = jnp.array([times])

        if self._integrator_method in ["Y4", "Y6", "Y8"]:
            times, inds = create_leapfrog_times(
                state.relative_time, times, self._max_step_size
            )
        else:
            inds = jnp.arange(times.shape[0])

        integrator = self._integrator if interpolate else self._forced_integrator
        is_leapfrog = self._integrator_method in ("Y4", "Y6", "Y8")
        # IAS15 with interpolation has dense-output b-coefficients available, so
        # we can use them to evaluate the polynomial at light-travel-delayed times in on_sky's
        # LTT loop instead of a constant-acceleration Taylor expansion. All other
        # paths (leapfrog, IAS15 forced-landing) fall back to the original Taylor.
        use_dense_ltt = interpolate and not is_leapfrog

        if uncertainty:
            # The covariance path runs the integration inside jax.jacfwd, which cannot
            # be threaded through the host-side stitching loop. For IAS15, detect a span
            # that would overflow the dense-output buffer and raise rather than silently
            # truncate (a nominal ephemeris() call auto-handles the same span).
            steps = None
            if not is_leapfrog:
                would_truncate, steps = ias15_span_probe(
                    state.to_system(),
                    self.gravity,
                    times,
                    integrator_state,
                    self._step_scheduler,
                    self._ias15_max_steps,
                )
                if would_truncate:
                    raise RuntimeError(
                        "ephemeris(uncertainty=True) over this span would exceed the "
                        "IAS15 dense-output buffer and silently truncate. The covariance "
                        "path uses forward-mode autodiff and cannot be transparently "
                        "stitched; request a shorter span or more closely spaced times. "
                        "(A nominal ephemeris(uncertainty=False) call auto-handles this.)"
                    )
            if use_dense_ltt:
                ras, decs, cov_radec = _ephem_ias15_with_cov(
                    times,
                    state,
                    self.gravity,
                    observer_positions,
                    inds,
                    self._step_scheduler,
                    cov,
                    self._ias15_max_steps,
                )
            else:
                ras, decs, cov_radec = _ephem_with_cov(
                    times,
                    state,
                    self.gravity,
                    integrator,
                    integrator_state,
                    observer_positions,
                    inds,
                    self._step_scheduler,
                    cov,
                )
            coords = SkyCoord(ra=ras, dec=decs, unit=u.rad, frame="icrs")
            if return_steps:
                return (coords, cov_radec, steps)
            return (coords, cov_radec)

        if use_dense_ltt:
            ras, decs, steps = _ephem_ias15_stitched(
                times,
                state,
                self.gravity,
                integrator_state,
                observer_positions,
                inds,
                self._step_scheduler,
                self._ias15_max_steps,
            )
        elif not is_leapfrog:
            # IAS15 forced-landing (interpolate=False): Taylor-LTT on budgeted landings.
            ras, decs, steps = _ephem_forced_budgeted(
                times,
                state,
                self.gravity,
                integrator_state,
                observer_positions,
                inds,
                self._step_scheduler,
            )
        else:
            ras, decs = _ephem(
                times,
                state,
                self.gravity,
                integrator,
                integrator_state,
                observer_positions,
                inds,
                self._step_scheduler,
            )
            steps = None
        coords = SkyCoord(ra=ras, dec=decs, unit=u.rad, frame="icrs")
        if return_steps:
            return (coords, steps)
        return coords

    def max_likelihood(
        self,
        fit_seed: CartesianState | KeplerianState | None = None,
        verbose: bool = False,
    ) -> Particle:
        """Find the maximum likelihood orbit for the particle.

        Args:
            fit_seed (CartesianState | KeplerianState | None):
                A seed for fitting the orbit of the particle. If None, a seed will be
                generated from the observations if they exist. Otherwise, a circular
                orbit with semi-major axis 2.5 AU will be used.
            verbose (bool):
                Whether to print the optimization progress. Defaults to False.

        Returns:
            Particle:
                A new Particle object whose state matches the maximum likelihood orbit.
        """
        if self.loglike is None:
            raise ValueError("No observations provided, cannot fit an orbit")

        if fit_seed is None:
            fit_seed = self._fit_seed

        x0 = jnp.concatenate(
            [
                fit_seed.to_cartesian().x.flatten(),
                fit_seed.to_cartesian().v.flatten(),
            ]
        )

        if (not self._is_keplerian) and self._integrator_method == "ias15":
            # Best-effort detect-and-raise guard. The likelihood integrates the
            # observation span with the interpolation path (the 15k dense-output
            # buffer). The orbit changes during optimization, but if even the seed
            # already overflows the buffer the fit would silently truncate. Orbit
            # fitting runs inside forward-mode autodiff and so cannot be transparently
            # stitched the way the nominal integrate/ephemeris methods are.
            seed_state = CartesianState(
                x=jnp.array([x0[:3]]),
                v=jnp.array([x0[3:]]),
                relative_time=self._time,
                time_reference=self._t_ref_jd,
                acceleration_func_kwargs=self._acc_func_kwargs,
            )
            obs_times = self._observations_times_as_offsets()
            a0 = self.gravity(seed_state.to_system())
            would_truncate, _steps = ias15_span_probe(
                seed_state.to_system(),
                self.gravity,
                obs_times,
                initialize_ias15_integrator_state(a0),
                self._step_scheduler,
            )
            if would_truncate:
                raise RuntimeError(
                    "The observation time span would exceed the IAS15 dense-output "
                    "buffer and silently truncate the likelihood integration. Orbit "
                    "fitting runs inside forward-mode autodiff and cannot be "
                    "transparently stitched; fit over a shorter arc or split the data."
                )

        if self._is_keplerian:
            # Keplerian propagation produces NaN for hyperbolic orbits, so we
            # guard the objective and gradient. We also scale the parameters so
            # the initial identity-Hessian step in L-BFGS-B is well-sized.
            scale = jnp.abs(x0) + 1e-10

            def _scaled_objective(x_scaled: jnp.ndarray) -> float:
                val = self.scipy_objective(x_scaled * scale)
                return float(jnp.where(jnp.isnan(val), 1e30, val))

            def _scaled_grad(x_scaled: jnp.ndarray) -> jnp.ndarray:
                g = self.scipy_objective_grad(x_scaled * scale) * scale
                return jnp.where(jnp.isnan(g), 0.0, g)

            result = minimize(
                fun=_scaled_objective,
                x0=x0 / scale,
                jac=_scaled_grad,
                method="L-BFGS-B",
                options={
                    "disp": verbose,
                    "maxls": 100,
                    "maxcor": 100,
                    "maxfun": 5000,
                    "maxiter": 1000,
                    "ftol": 1e-14,
                },
            )
            result.x = result.x * scale
        else:
            result = minimize(
                fun=lambda x: self.scipy_objective(x),
                x0=x0,
                jac=lambda x: self.scipy_objective_grad(x),
                method="L-BFGS-B",
                options={
                    "disp": verbose,
                    "maxls": 100,
                    "maxcor": 100,
                    "maxfun": 5000,
                    "maxiter": 1000,
                    "ftol": 1e-14,
                },
            )

        if result.success:
            c = CartesianState(
                x=jnp.array([result.x[:3]]),
                v=jnp.array([result.x[3:]]),
                relative_time=self._time,
                time_reference=self._t_ref_jd,
                acceleration_func_kwargs=self._acc_func_kwargs,
            )
            if c.to_keplerian().ecc > 1:
                warnings.warn(
                    "Warning: max_likelihood fit produced an unbound orbit",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return Particle(
                x=result.x[:3],
                v=result.x[3:],
                time=self._t_ref_astropy,
                state=None,
                observations=self._observations,
                name=self._name,
                gravity=self._gravity_str,
                de_ephemeris_version=self._de_ephemeris_version,
                integrator=self._integrator_method,
                earliest_time=self._earliest_time,
                latest_time=self._latest_time,
                fit_seed=self._fit_seed,
                max_step_size=self._max_step_size,
            )
        else:
            raise ValueError("Failed to converge")

    def is_observable(
        self,
        times: Time,
        observer: str | jnp.ndarray,
        sun_limit: float = 20.0,
        ephem: SkyCoord | None = None,
        return_angle: bool = False,
    ) -> jnp.ndarray:
        """Check whether a particle is observable or too close to the Sun.

        Args:
            times (Time):
                The times to check the observability.
            observer (str | jnp.ndarray):
                The observer/observatory making the observations. Can be a string for
                the name/code of an observatory, or a jnp.array of 3D barycentric ICRS
                positions in AU.
            sun_limit (float):
                The minimum allowed angular separation from the Sun, in degrees.
                Defaults to 20 degrees.
            ephem (SkyCoord | None, optional):
                Optionally, the ephemeris of the particle at the given times. If not
                provided, will be computed using the ephemeris method. Helpful if you've
                already computed the ephemeris and want to avoid doing it twice.
            return_angle (bool, optional):
                If True, will return the angles to the Sun in degrees, not the mask.
                Default is False.

        Returns:
            jnp.ndarray:
                A boolean array indicating whether the particle is observable at each
                time (True) or too close to the Sun (False).
        """
        if isinstance(observer, str):
            observer_pos = get_observer_positions(
                times, observer, self._de_ephemeris_version
            )
        else:
            observer_pos = observer

        if ephem is None:
            ephem = self.ephemeris(times, observer)

        if isinstance(times, Time):
            times = jnp.array(times.tdb.jd)
        if times.shape == ():
            times = jnp.array([times])

        eph = Ephemeris(
            earliest_time=self._earliest_time,
            latest_time=self._latest_time,
            de_ephemeris_version=self._de_ephemeris_version,
        )
        sun_pos = jax.vmap(eph.processor.state)(times)[0][:, 0, :]

        eph_unit_vec = jnp.array(ephem.cartesian.xyz).T

        # want the angle between the eph_unit_vec, with its tail on the observer, and the vector from the observer to the sun
        obs_to_sun_vec = sun_pos - observer_pos
        obs_to_sun_unit_vec = (
            obs_to_sun_vec / jnp.linalg.norm(obs_to_sun_vec, axis=1)[:, None]
        )
        angle = jnp.arccos(jnp.sum(obs_to_sun_unit_vec * eph_unit_vec, axis=1))
        if return_angle:
            return jnp.rad2deg(angle)
        else:
            return angle > jnp.deg2rad(sun_limit)

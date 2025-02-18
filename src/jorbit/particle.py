"""The Particle class and its supporting functions."""

import jax

jax.config.update("jax_enable_x64", True)
import warnings
from collections.abc import Callable

import astropy.units as u
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
from jorbit.astrometry.sky_projection import on_sky, tangent_plane_projection
from jorbit.ephemeris.ephemeris import Ephemeris
from jorbit.integrators import ias15_evolve, initialize_ias15_integrator_state
from jorbit.utils.horizons import get_observer_positions
from jorbit.utils.states import (
    CartesianState,
    IAS15IntegratorState,
    KeplerianState,
    SystemState,
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
        integrator: str = "ias15",
        earliest_time: Time = Time("1980-01-01"),
        latest_time: Time = Time("2050-01-01"),
        fit_seed: KeplerianState | CartesianState | None = None,
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
                acceleration functions in jorbit.accelerations.
            integrator (str):
                The integrator to use for the particle. Defaults to "ias15", which is a
                15th order adaptive step-size integrator. Currently IAS15 is the only
                option- this is a vestige of previous experiments with Gauss-Jackson
                integrators that we might return to someday.
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
        """
        self._observations = observations
        self._earliest_time = earliest_time
        self._latest_time = latest_time

        self.gravity = gravity
        self._integrator = integrator

        (
            self._x,
            self._v,
            self._time,
            self._cartesian_state,
            self._keplerian_state,
            self._name,
        ) = self._setup_state(x, v, state, time, name)

        self.gravity = self._setup_acceleration_func(gravity)

        self._integrator_state, self._integrator = self._setup_integrator()

        self._fit_seed = self._setup_fit_seed(fit_seed)

        (
            self.residuals,
            self.loglike,
            self.scipy_objective,
            self.scipy_objective_grad,
        ) = self._setup_likelihood()

    def __repr__(self) -> str:
        """Return a string representation of the Particle object."""
        return f"Particle: {self._name}"

    @property
    def cartesian_state(self) -> CartesianState:
        """Return the Cartesian state of the particle."""
        return self._cartesian_state

    @property
    def keplerian_state(self) -> KeplerianState:
        """Return the Keplerian state of the particle."""
        return self._keplerian_state

    ###############
    # SETUP METHODS
    ###############

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
            time = state.time

        assert time is not None, "Must provide an epoch for the particle"
        if isinstance(time, type(Time("2023-01-01"))):
            time = time.tdb.jd

        if state is not None:
            assert x is None and v is None, "Cannot provide both state and x, v"

            state = state.to_cartesian()
            if state.x.ndim != 2:
                state.x = state.x[None, :]
                state.v = state.v[None, :]
            state.time = time
            keplerian_state = state.to_keplerian()
            cartesian_state = state.to_cartesian()

            x = state.x.flatten()
            v = state.v.flatten()

        elif x is not None:
            assert v is not None, "Must provide both x and v"

            x = x.flatten()
            v = v.flatten()
            cartesian_state = CartesianState(
                x=jnp.array([x]), v=jnp.array([v]), time=time
            )
            keplerian_state = cartesian_state.to_keplerian()
        else:
            raise ValueError(
                "time must be either astropy.time.Time or float (interpreted as JD in"
                " TDB)"
            )

        if name == "":
            name = "unnamed"

        return x, v, time, cartesian_state, keplerian_state, name

    def _setup_acceleration_func(self, gravity: str) -> Callable:

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

    def _setup_integrator(self) -> tuple:
        a0 = self.gravity(self._cartesian_state.to_system())
        integrator_state = initialize_ias15_integrator_state(a0)
        integrator = jax.tree_util.Partial(ias15_evolve)

        return integrator_state, integrator

    def _setup_fit_seed(
        self, fit_seed: KeplerianState | CartesianState | None
    ) -> KeplerianState | CartesianState | None:

        if self._observations is None:
            return None

        if isinstance(fit_seed, (CartesianState | KeplerianState)):
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
            fit_seed = simple_circular(
                self._observations.ra[0],
                self._observations.dec[0],
                semi=2.5,
                time=self._time,
            )

        return fit_seed

    def _setup_likelihood(self) -> tuple[Callable, Callable, Callable, Callable]:
        if self._observations is None:
            return None, None, None, None

        residuals = jax.tree_util.Partial(
            _residuals,
            self._observations.times,
            self.gravity,
            self._integrator,
            self._integrator_state,
            self._observations.observer_positions,
            self._observations.ra,
            self._observations.dec,
        )

        ll = jax.tree_util.Partial(
            _loglike,
            self._observations.times,
            self.gravity,
            self._integrator,
            self._integrator_state,
            self._observations.observer_positions,
            self._observations.ra,
            self._observations.dec,
            self._observations.inv_cov_matrices,
            self._observations.cov_log_dets,
        )

        # since we've gone with the while loop version of the integrator, can no
        # longer use reverse mode. But, actually specifying forward mode everywhere is
        # annoying, so we're going to re-define a custom vjp for "reverse" mode that's
        # actually just forward mode

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

        residuals = jax.jit(residuals)
        loglike = jax.jit(loglike)

        def scipy_objective(x: jnp.ndarray) -> float:
            c = CartesianState(
                x=jnp.array([x[:3]]), v=jnp.array([x[3:]]), time=self._time
            )
            return -loglike(c)

        def scipy_grad(x: jnp.ndarray) -> jnp.ndarray:
            c = CartesianState(
                x=jnp.array([x[:3]]), v=jnp.array([x[3:]]), time=self._time
            )
            c_grad = jax.grad(loglike)(c)
            g = jnp.concatenate([c_grad.x.flatten(), c_grad.v.flatten()])
            return -g

        return residuals, loglike, scipy_objective, scipy_grad

    ################
    # PUBLIC METHODS
    ################

    def integrate(
        self, times: Time, state: CartesianState | KeplerianState | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Integrate the particle's orbit to a given time.

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

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]:
                The positions of the particle at the given times, in AU, and the
                The velocities of the particle at the given times, in AU/day.
        """
        if state is None:
            state = self._cartesian_state

        if isinstance(times, Time):
            times = jnp.array(times.tdb.jd)
        if times.shape == ():
            times = jnp.array([times])

        positions, velocities, final_system_state, final_integrator_state = _integrate(
            times, state, self.gravity, self._integrator, self._integrator_state
        )
        return positions[:, 0, :], velocities[:, 0, :]

    def ephemeris(
        self,
        times: Time,
        observer: str | jnp.ndarray,
        state: CartesianState | KeplerianState | None = None,
    ) -> SkyCoord:
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

        Returns:
            coords (SkyCoord):
                The ephemeris of the particle at the given times, in ICRS coordinates,
                as seen from that specific observer and correcting for light travel
                time.
        """
        if isinstance(observer, str):
            observer_positions = get_observer_positions(times, observer)
        else:
            observer_positions = observer

        if state is None:
            state = self._cartesian_state

        if isinstance(times, Time):
            times = jnp.array(times.tdb.jd)
        if times.shape == ():
            times = jnp.array([times])

        ras, decs = _ephem(
            times,
            state,
            self.gravity,
            self._integrator,
            self._integrator_state,
            observer_positions,
        )
        return SkyCoord(ra=ras, dec=decs, unit=u.rad, frame="icrs")

    def max_likelihood(
        self,
        fit_seed: CartesianState | KeplerianState | None = None,
        verbose: bool = False,
    ) -> "Particle":
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

        result = minimize(
            fun=lambda x: self.scipy_objective(x),
            x0=jnp.concatenate(
                [
                    fit_seed.to_cartesian().x.flatten(),
                    fit_seed.to_cartesian().v.flatten(),
                ]
            ),
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
                time=self._time,
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
                time=self._time,
                state=None,
                observations=self._observations,
                name=self._name,
                gravity=self.gravity,
                integrator=self._integrator,
                earliest_time=self._earliest_time,
                latest_time=self._latest_time,
                fit_seed=self._fit_seed,
            )
        else:
            raise ValueError("Failed to converge")

    # def fit_mcmc(
    #     self,
    #     max_likelihood_position,
    #     enforce_bounded_orbit=True,
    #     n_walkers=32,
    #     n_steps=1000,
    #     n_burn_steps="auto",
    #     thin="auto",
    #     flat=True,
    #     verbose=True,
    # ):
    #     # get the initial flattened vector
    #     tmp = max_likelihood_position.to_keplerian()
    #     x0_vec = jnp.concatenate(
    #         [
    #             tmp.semi.flatten(),
    #             tmp.ecc.flatten(),
    #             tmp.inc.flatten(),
    #             tmp.Omega.flatten(),
    #             tmp.omega.flatten(),
    #             tmp.nu.flatten(),
    #         ]
    #     )
    #     # # save the scale of each parameter- we'll rescale the samples back to this
    #     # # at the end, but work in a space near all 1s
    #     # parameter_scales = x0_vec

    #     # # set up the initial walker positions, small normal ball around max likelihood
    #     # x0 = jnp.ones((n_walkers, len(x0_vec)))
    #     x0 = jnp.tile(x0_vec, (n_walkers, 1))
    #     # x0 = x0 + jax.random.normal(jax.random.PRNGKey(0), x0.shape) * 1e-5
    #     grads = -self.scipy_objective_grad(x0_vec)
    #     return x0, grads

    #     def bounded_orbit_prior(x):
    #         # c = CartesianState(
    #         #     x=jnp.array([x[:3]]), v=jnp.array([x[3:]]), time=self._time
    #         # )
    #         # if c.to_keplerian().ecc >= 1-1e-3:
    #         if x[1] >= 1 - 1e-3:
    #             return -jnp.inf
    #         return 0

    #     prior = bounded_orbit_prior if enforce_bounded_orbit else lambda x: 0

    #     def emcee_loglike(x):
    #         # x = x * parameter_scales
    #         p = prior(x)
    #         if not jnp.isfinite(p):
    #             return -jnp.inf
    #         # c = CartesianState(
    #         #     x=jnp.array([x[:3]]), v=jnp.array([x[3:]]), time=self._time
    #         # )
    #         c = KeplerianState(
    #             semi=jnp.array([x[0]]),
    #             ecc=jnp.array([x[1]]),
    #             inc=jnp.array([x[2]]),
    #             Omega=jnp.array([x[3]]),
    #             omega=jnp.array([x[4]]),
    #             nu=jnp.array([x[5]]),
    #             time=self._time,
    #         )
    #         return self.loglike(c) + p

    #     # set up the sampler
    #     sampler = EnsembleSampler(
    #         n_walkers,
    #         len(x0_vec),
    #         lambda x: emcee_loglike(x),
    #         # moves=[StretchMove],
    #     )

    #     # run the sampler
    #     sampler.run_mcmc(x0, n_steps, progress=verbose)

    #     # return sampler, parameter_scales, prior, x0
    #     return sampler, prior, x0

    #     # post-process the chains
    #     x = sampler.get_chain()

    #     if n_burn_steps == "auto":
    #         # find the point where each chain first crosses its median value
    #         medians = jnp.median(x, axis=0)
    #         above_median = x > medians[None, ...]
    #         crossings = above_median[:-1] != above_median[1:]
    #         first_med_crossings = jnp.argmax(crossings, axis=0)

    #         # nan out the everything before the first crossings:
    #         mask = (
    #             jnp.arange(x.shape[0])[:, None, None] > first_med_crossings[None, ...]
    #         )
    #         x = jnp.where(mask, x, jnp.nan)
    #     else:
    #         x = x[n_burn_steps:]

    #     if thin == "auto":
    #         try:
    #             tau = sampler.get_autocorr_time()
    #             thin = int(jnp.ceil(2 * tau.max()))
    #         except AutocorrError:
    #             warnings.warn(
    #                 "emcee could not compute the autocorrelation time, chain is likely too short. Proceeding without any thinning.",
    #                 stacklevel=2,
    #             )
    #             thin = 1

    #     # thin the chain
    #     x = x[::thin]

    #     # rescale the samples back to the original scale
    #     x = x * parameter_scales

    #     # flatten the chain
    #     if flat:
    #         x = x.reshape(-1, x.shape[-1])

    #     # get rid of the nans
    #     x = x[~jnp.isnan(x).any(axis=-1)]

    #     return x

    # def fit(self):
    #     pass


###########################
# EXTERNAL JITTED FUNCTIONS
###########################


@jax.jit
def _integrate(
    times: jnp.ndarray,
    particle_state: CartesianState | KeplerianState,
    acc_func: Callable,
    integrator_func: Callable,
    integrator_state: IAS15IntegratorState,
) -> tuple[jnp.ndarray, jnp.ndarray, SystemState, IAS15IntegratorState]:
    state = particle_state.to_system()
    positions, velocities, final_system_state, final_integrator_state = integrator_func(
        state, acc_func, times, integrator_state
    )

    return positions, velocities, final_system_state, final_integrator_state


@jax.jit
def _ephem(
    times: jnp.ndarray,
    particle_state: CartesianState | KeplerianState,
    acc_func: Callable,
    integrator_func: Callable,
    integrator_state: IAS15IntegratorState,
    observer_positions: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    positions, velocities, _, _ = _integrate(
        times, particle_state, acc_func, integrator_func, integrator_state
    )

    # # only one particle, so take the 0th particle. shape is (time, particles, 3)
    # ras, decs = jax.vmap(on_sky, in_axes=(0, 0, 0, 0, None))(
    #         positions[:,0,:], velocities[:,0,:], times, observer_positions, acc_func
    #     )

    def scan_func(carry: None, scan_over: tuple) -> tuple[None, tuple]:
        position, velocity, time, observer_position = scan_over
        ra, dec = on_sky(position, velocity, time, observer_position, acc_func)
        return None, (ra, dec)

    _, (ras, decs) = jax.lax.scan(
        scan_func,
        None,
        (positions[:, 0, :], velocities[:, 0, :], times, observer_positions),
    )

    return ras, decs


@jax.jit
def _residuals(
    times: jnp.ndarray,
    gravity: Callable,
    integrator: Callable,
    integrator_state: IAS15IntegratorState,
    observer_positions: jnp.ndarray,
    ra: jnp.ndarray,
    dec: jnp.ndarray,
    particle_state: CartesianState | KeplerianState,
) -> jnp.ndarray:
    ras, decs = _ephem(
        times,
        particle_state,
        gravity,
        integrator,
        integrator_state,
        observer_positions,
    )

    xis_etas = jax.vmap(tangent_plane_projection)(ra, dec, ras, decs)

    return xis_etas


# note: this external jitted function does not have fwd mode autodiff enforced, will
# break on reverse mode
@jax.jit
def _loglike(
    times: jnp.ndarray,
    gravity: Callable,
    integrator: Callable,
    integrator_state: IAS15IntegratorState,
    observer_positions: jnp.ndarray,
    ra: jnp.ndarray,
    dec: jnp.ndarray,
    inv_cov_matrices: jnp.ndarray,
    cov_log_dets: jnp.ndarray,
    particle_state: CartesianState | KeplerianState,
) -> float:
    xis_etas = _residuals(
        times,
        gravity,
        integrator,
        integrator_state,
        observer_positions,
        ra,
        dec,
        particle_state,
    )

    quad = jnp.einsum("bi,bij,bj->b", xis_etas, inv_cov_matrices, xis_etas)

    ll = jnp.sum(-0.5 * (2 * jnp.log(2 * jnp.pi) + cov_log_dets + quad))

    return ll

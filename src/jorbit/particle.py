import jax

jax.config.update("jax_enable_x64", True)
import warnings

import astropy.units as u
import jax.numpy as jnp
from astropy.coordinates import SkyCoord
from astropy.time import Time
from scipy.optimize import minimize

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
from jorbit.utils.states import CartesianState, KeplerianState


class Particle:
    def __init__(
        self,
        state=None,
        time=None,
        x=None,
        v=None,
        observations=None,
        name="",
        gravity="default solar system",
        integrator="ias15",
        earliest_time=Time("1980-01-01"),
        latest_time=Time("2050-01-01"),
        fit_seed=None,
    ):

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

    def __repr__(self):
        return f"Particle: {self._name}"

    @property
    def cartesian_state(self):
        return self._cartesian_state

    @property
    def keplerian_state(self):
        return self._keplerian_state

    ###############
    # SETUP METHODS
    ###############

    def _setup_state(self, x, v, state, time, name):

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
        elif self.gravity == "default solar system":
            eph = Ephemeris(
                earliest_time=self._earliest_time,
                latest_time=self._latest_time,
                ssos="default solar system",
            )
            acc_func = create_default_ephemeris_acceleration_func(eph.processor)

        return acc_func

    def _setup_integrator(self):
        a0 = self.gravity(self._cartesian_state.to_system())
        integrator_state = initialize_ias15_integrator_state(a0)
        integrator = jax.tree_util.Partial(ias15_evolve)

        return integrator_state, integrator

    def _setup_fit_seed(self, fit_seed):

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

    def _setup_likelihood(self):
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
        def loglike(params):
            return ll(params)

        def loglike_fwd(params):
            output = ll(params)
            jac = jax.jacfwd(ll)(params)
            return output, (jac,)

        def loglike_bwd(res, g):
            jac = res
            val = jax.tree.map(lambda x: x * g, jac)
            return val

        loglike.defvjp(loglike_fwd, loglike_bwd)

        residuals = jax.jit(residuals)
        loglike = jax.jit(loglike)

        def scipy_loglike(x):
            c = CartesianState(
                x=jnp.array([x[:3]]), v=jnp.array([x[3:]]), time=self._time
            )
            return -loglike(c)

        def scipy_grad(x):
            c = CartesianState(
                x=jnp.array([x[:3]]), v=jnp.array([x[3:]]), time=self._time
            )
            c_grad = jax.grad(loglike)(c)
            g = jnp.concatenate([c_grad.x.flatten(), c_grad.v.flatten()])
            return -g

        return residuals, loglike, scipy_loglike, scipy_grad

    ################
    # PUBLIC METHODS
    ################

    def integrate(self, times, state=None):
        if state is None:
            state = self._cartesian_state

        if isinstance(times, Time):
            times = jnp.array(times.tdb.jd)
        if times.shape == ():
            times = jnp.array([times])

        positions, velocities, final_system_state, final_integrator_state = _integrate(
            times, state, self.gravity, self._integrator, self._integrator_state
        )
        return positions[0], velocities[0]

    def ephemeris(self, times, observer, state=None):
        observer_positions = get_observer_positions(times, observer)

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

    def max_likelihood(self, fit_seed=None, verbose=False):
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

    def fit_mcmc(self):
        pass

    def fit(self):
        pass


###########################
# EXTERNAL JITTED FUNCTIONS
###########################


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

    # # only one particle, so take the 0th particle. shape is (time, particles, 3)
    # ras, decs = jax.vmap(on_sky, in_axes=(0, 0, 0, 0, None))(
    #         positions[:,0,:], velocities[:,0,:], times, observer_positions, acc_func
    #     )

    def scan_func(carry, scan_over):
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
    times,
    gravity,
    integrator,
    integrator_state,
    observer_positions,
    ra,
    dec,
    particle_state,
):
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
    times,
    gravity,
    integrator,
    integrator_state,
    observer_positions,
    ra,
    dec,
    inv_cov_matrices,
    cov_log_dets,
    particle_state,
):
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

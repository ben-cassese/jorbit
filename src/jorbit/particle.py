import jax

jax.config.update("jax_enable_x64", True)
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
from jorbit.astrometry.sky_projection import on_sky, tangent_plane_projection
from jorbit.astrometry.transformations import icrs_to_horizons_ecliptic
from jorbit.ephemeris.ephemeris import Ephemeris
from jorbit.integrators import ias15_evolve, initialize_ias15_integrator_state
from jorbit.utils.horizons import get_observer_positions
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
        gravity="default solar system",
        integrator="ias15",
        earliest_time=Time("1980-01-01"),
        latest_time=Time("2050-01-01"),
        fit_seed=None,
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
        self._fit_seed = fit_seed

        self._setup_state()

        self._setup_acceleration_func()

        self._setup_integrator()

        self.residuals, self.loglike = self._setup_likelihood()

    def __repr__(self):
        return f"Particle: {self._name}"

    @property
    def cartesian_state(self):
        return self._cartesian_state

    @property
    def keplerian_state(self):
        return self._keplerian_state

    def _setup_state(self):

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
        a0 = self.gravity(self._cartesian_state.to_system())
        self._integrator_state = initialize_ias15_integrator_state(a0)
        self._integrator = jax.tree_util.Partial(ias15_evolve)

    def _setup_likelihood(self):
        if self._observations is None:
            return None

        def _residuals(x):
            state = CartesianState(x=x[:3][None, :], v=x[3:][None, :], time=self._time)
            ras, decs = _ephem(
                self._observations.times,
                state,
                self.gravity,
                self._integrator,
                self._integrator_state,
                self._observations.observer_positions,
            )

            xis_etas = jax.vmap(tangent_plane_projection)(
                self._observations.ra, self._observations.dec, ras, decs
            )

            return xis_etas

        def _loglike(x):
            xis_etas = _residuals(x)

            quad = jnp.einsum(
                "bi,bij,bj->b", xis_etas, self._observations.inv_cov_matrices, xis_etas
            )

            ll = jnp.sum(
                -0.5
                * (2 * jnp.log(2 * jnp.pi) + self._observations.cov_log_dets + quad)
            )

            return ll

        _loglike = jax.tree_util.Partial(_loglike)
        _residuals = jax.tree_util.Partial(_residuals)

        # since we've gone with the while loop version of the integrator, can no
        # longer use reverse mode. But, actually specifying forward mode everywhere is
        # annoying, so we're going to re-define a custom vjp for "reverse" mode that's
        # actually just forward mode

        @jax.custom_vjp
        def loglike(params):
            return _loglike(params)

        def loglike_fwd(params):
            output = _loglike(params)
            jac = jax.jacfwd(_loglike)(params)
            return output, (jac,)

        def loglike_bwd(res, g):
            jac = res
            val = jax.tree.map(lambda x: x * g, jac)
            return val

        loglike.defvjp(loglike_fwd, loglike_bwd)

        residuals = jax.jit(_residuals)
        loglike = jax.jit(loglike)

        return residuals, loglike

    def _generate_initial_guess(self, ra, dec):
        phi = ra
        theta = jnp.pi / 2 - dec

        x = jnp.sin(theta) * jnp.cos(phi)
        y = jnp.sin(theta) * jnp.sin(phi)
        z = jnp.cos(theta)

        x_icrs = jnp.hstack([x, y, z])
        x = icrs_to_horizons_ecliptic(x_icrs)

        # assume we're observing the thing at its highest excursion from the ecliptic:
        inc = jnp.array([jnp.abs(jnp.arcsin(x[2])) / jnp.linalg.norm(x) * 180 / jnp.pi])

        # its longitude of ascending node is the angle between the x-axis and the projection of the vector onto the xy-plane:
        varphi = (jnp.arctan2(x[1], x[0]) * 180 / jnp.pi) % 360
        Omega = jnp.array([varphi]) - 90 if x[2] > 0 else jnp.array([varphi]) + 90

        nu = jnp.array([90.0]) if x[2] > 0 else jnp.array([270.0])
        a = jnp.array([2.5])
        ecc = jnp.array([0.0])
        omega = jnp.array([0.0])

        k = KeplerianState(
            semi=a,
            ecc=ecc,
            nu=nu,
            inc=inc,
            Omega=Omega,
            omega=omega,
            time=self._observations.times[0],
        )
        c = k.to_cartesian()

        return jnp.concatenate([c.x.flatten(), c.v.flatten()])

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

    def max_likelihood(self):
        if self.loglike is None:
            raise ValueError("No observations provided, cannot fit an orbit")

        if self._fit_seed is None:
            x0 = self._generate_initial_guess(
                self._observations.ra[0], self._observations.dec[0]
            )
        else:
            x0 = jnp.concatenate(
                [self.fit_seed.to_cartesian.x, self.fit_seed.to_cartesian.v]
            )

        result = minimize(
            fun=lambda x: -self.loglike(x),
            x0=x0,
            jac=lambda x: -jax.grad(self.loglike)(x),
            method="L-BFGS-B",
            options={
                "disp": True,
                "maxls": 100,
                "maxcor": 50,
                "maxfun": 5000,
                "maxiter": 1000,
                "ftol": 1e-10,
            },
        )

        if result.success:
            return Particle(
                x=result.x[:3],
                v=result.x[3:],
                time=self._time,
                elements=None,
                log_gm=self._log_gm,
                observations=self._observations,
                name=self._name,
                gravity=self.gravity,
                integrator=self._integrator,
                earliest_time=self._earliest_time,
                latest_time=self._latest_time,
                fit_seed=None,
            )
        else:
            raise ValueError("Failed to converge")

    def fit(self):
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

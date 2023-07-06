import jax
from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore', module='erfa')
from astropy.time import Time


from .data.constants import all_planets, large_asteroids
from .construct_perturbers import construct_perturbers, STANDARD_SUN_PARAMS
from .engine import (
    negative_loglike_single,
    negative_loglike_single_grad,
    integrate_multiple,
    on_sky,
    sky_error,
    cart_to_elements,
)


class Particle:
    def __init__(
        self,
        x=None,
        v=None,
        gm=0,
        time=2451545.0,
        observations=None,
        planet_perturbers=all_planets,
        asteroid_perturbers=large_asteroids,
        earliest_time=Time("1980-01-01"),
        latest_time=Time("2100-01-01"),
        name="",
        fit_state=True,
        fit_gm=False
    ):
        self.x = x
        self.v = v
        self.gm = gm
        if isinstance(time, type(Time("2023-01-01"))):
            self.time = time.tdb.jd
        elif isinstance(time, float):
            self.time = time
        else:
            raise ValueError(
                "time must be either astropy.time.Time or float (interpreted as JD in TDB)"
            )
        self.observations = observations
        if type(observations) != type(None):
            self.time = observations.times[0]

        (
            self.planet_params,
            self.asteroid_params,
            self.planet_gms,
            self.asteroid_gms,
        ) = construct_perturbers(
            planets=planet_perturbers, asteroids=asteroid_perturbers,
            earliest_time=earliest_time,
            latest_time=latest_time,
        )
        self.earliest_time = earliest_time
        self.latest_time = latest_time
        self.name = name
        self.fit_state = fit_state
        self.fit_gm = fit_gm

        # if type(observations) == type(None) and not fit_gm:
        #     warnings.warn("No observations provided, and fit_gm=False; therefore, this particle cannot affect the likelihood of any fits")
        # if not fit_state and not fit_gm:
        #     warnings.warn("Both fit_state and fit_gm are False; therefore, this particle cannot affect the likelihood of any fits")

    def _negative_loglike(self, X):
        return negative_loglike_single(
            X=X,
            times=self.observations.times,
            planet_params=self.planet_params,
            asteroid_params=self.asteroid_params,
            planet_gms=self.planet_gms,
            asteroid_gms=self.asteroid_gms,
            observer_positions=self.observations.observer_positions,
            ra=self.observations.ra,
            dec=self.observations.dec,
            position_uncertainties=self.observations.astrometry_uncertainties,
        )

    def _negative_loglike_grad(self, X):
        return negative_loglike_single_grad(
            X,
            times=self.observations.times,
            planet_params=self.planet_params,
            asteroid_params=self.asteroid_params,
            planet_gms=self.planet_gms,
            asteroid_gms=self.asteroid_gms,
            observer_positions=self.observations.observer_positions,
            ra=self.observations.ra,
            dec=self.observations.dec,
            position_uncertainties=self.observations.astrometry_uncertainties,
        )

    def compute_best_fit(self):
        assert self.observations is not None, "No observations provided to fit"

        def inner():
            x0 = jnp.array(
                list(np.random.uniform(-10, 10, size=3))
                + list(np.random.uniform(-1, 1, size=3))
            )
            res = minimize(self._negative_loglike, x0, jac=self._negative_loglike_grad)
            x = res.x[:3][None, :]
            v = res.x[3:][None, :]

            xs, vs, final_times, success = integrate_multiple(
                xs=x,
                vs=v,
                gms=jnp.array([0]),
                initial_time=self.observations.times[0],
                final_times=self.observations.times[1:],
                planet_params=self.planet_params,
                asteroid_params=self.asteroid_params,
                planet_gms=self.planet_gms,
                asteroid_gms=self.asteroid_gms,
            )

            xs = jnp.concatenate((x[:, None, :], xs), axis=1)
            vs = jnp.concatenate((v[:, None, :], vs), axis=1)

            calc_RAs, calc_Decs = on_sky(
                xs=xs[0],
                vs=vs[0],
                gms=jnp.array([0]),
                times=self.observations.times,
                observer_positions=self.observations.observer_positions,
                planet_params=self.planet_params,
                asteroid_params=self.asteroid_params,
                planet_gms=self.planet_gms,
                asteroid_gms=self.asteroid_gms,
            )

            err = sky_error(
                calc_ra=calc_RAs,
                calc_dec=calc_Decs,
                true_ra=self.observations.ra,
                true_dec=self.observations.dec,
            )
            return x[0], v[0], err, res

        good = False
        for i in range(5):
            x, v, err, res = inner()
            if jnp.max(err) < 60:
                good = True
                break
        if good:
            self.best_fit_computed = True
            self.time = self.observations.times[0]
            self.xs = res.x[:3]
            self.vs = res.x[3:]
            self.time = self.observations.times[0]
            return {
                "Status": "Success",
                "Residuals (arcsec)": err,
                "Best Fit": {
                    "x (au)": self.xs,
                    "v (au / day)": self.vs,
                    "time": self.time,
                },
            }
        else:
            raise ValueError("Failed: Best fit had residuals > 1 arcmin")

    def propagate(self, times, use_GR=False, obey_large_step_limits=True):
        if isinstance(times, type(Time("2023-01-01"))):
            times = jnp.array(times.tdb.jd)
        elif isinstance(times, list):
            times = jnp.array([t.tdb.jd for t in times])
        if times.shape == ():
            times = jnp.array([times])

        assert jnp.max(times) < self.latest_time.tdb.jd, "Requested propagation includes times beyond the latest time in considered in the ephemeris for this particle. Consider initially setting a broader time range for the ephemeris."

        jumps = jnp.abs(jnp.diff(times))
        if jumps.shape != (0,): largest_jump = jnp.max(jumps)
        else: largest_jump = 0
        first_jump = jnp.abs(self.time - times[0])
        largest_jump = jnp.where(first_jump > largest_jump, first_jump, largest_jump)
        if obey_large_step_limits:
            assert largest_jump < 7305, "Requested propagation includes at least one step that is too large- max default is 20 years. May have to increase max_steps manually to proceed."
        if largest_jump < 1000: max_steps = jnp.arange(100)
        else: max_steps = jnp.arange(1000)

        if not obey_large_step_limits and largest_jump > 1000: max_steps = jnp.arange((largest_jump*1.25 / 12).astype(int))

        xs, vs, final_time, success = integrate_multiple(
            xs=self.x[None, :],
            vs=self.v[None, :],
            gms=jnp.array([self.gm]),
            initial_time=self.time,
            final_times=times,
            planet_params=self.planet_params,
            asteroid_params=self.asteroid_params,
            planet_gms=self.planet_gms,
            asteroid_gms=self.asteroid_gms,
            max_steps=max_steps,
            use_GR=use_GR,
        )

        self.x = xs[0, -1]
        self.v = vs[0, -1]
        self.time = final_time[0]
        # if jnp.sum(success) != len(success):
        #     warnings.warn("Integration may not have converged for all times")
        # print(success)
        # print(final_time)
        return xs[0], vs[0]

    def elements(self):
        return cart_to_elements(
            X=self.x, V=self.v, time=self.time, sun_params=STANDARD_SUN_PARAMS
        )

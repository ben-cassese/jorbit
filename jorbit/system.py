import jax
from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
import numpy as np
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore", module="erfa")
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u

from jorbit import Observations
from .data.constants import all_planets, large_asteroids
from .construct_perturbers import (
    construct_perturbers,
    STANDARD_PLANET_PARAMS,
    STANDARD_ASTEROID_PARAMS,
    STANDARD_PLANET_GMS,
    STANDARD_ASTEROID_GMS,
    STANDARD_SUN_PARAMS,
)
from .engine import (
    integrate_multiple,
    on_sky,
    prepare_loglike_input,
    loglike
)


class System:
    def __init__(
        self,
        particles,
        planets=all_planets,
        asteroids=large_asteroids,
        fit_planet_gms=False,
        fit_asteroid_gms=False,
        max_steps=jnp.arange(100),
    ):
        # initial stuff we can save/check right away
        self._time = particles[0].time
        earlys = []
        lates = []
        for p in particles:
            earlys.append(p.earliest_time.tdb.jd)
            lates.append(p.latest_time.tdb.jd)
            assert (
                p.time == self._time
            ), "All particles must be initalized to the same time"

        self._earliest_time = jnp.min(jnp.array(earlys))
        self._latest_time = jnp.max(jnp.array(lates))
        self._max_steps = max_steps

        ########################################################################
        # now we're going to loop through all of the particles and collect/sort
        # their states, gms, and what parameters can be varied in a fit

        # these lists (soon to be arrays) will contain info about every particle
        xs = []  # will be (n_particles, 3)
        vs = []  # will be (n_particles, 3)
        gms = []  # will be (n_particles,)

        # This is a mask which, when applied to xs, vs, or gms, tells you which
        # particles are tracers, and which could have non-zero gms
        TRACERS = []  # will be (n_particles,)

        # these lists (soon to be arrays) will *not* be as long as the # of particles.
        # instead, each will be as long as either TRACERS.sum() or (~TRACERS).sum()
        # they exist to show which tracers/massive particles have free/fixed states/gms
        # free_xx_mask will say "of the (tracers)/(massive) particles, which
        # have (states)/)gms) that can vary in a fit?"
        # F = [free state,(gm=0 & fit_gm=False)]
        # S = [free state,gm>0|fit_gm=True]
        free_tracer_state_mask = []  # will be (n_particles with F)
        fixed_tracer_xs = []  # will be ((n_particles - n_particles with F), 3)
        free_tracer_xs = []  # will be (n_particles with F, 3)
        fixed_tracer_vs = []  # will be ((n_particles - n_particles with F), 3)
        free_tracer_vs = []  # will be (n_particles with F, 3)
        free_massive_state_mask = []  # will be (n_particles with S)
        fixed_massive_xs = []  # will be ((n_particles - n_particles with S), 3)
        free_massive_xs = []  # will be (n_particles with S, 3)
        fixed_massive_vs = []  # will be ((n_particles - n_particles with S), 3)
        free_massive_vs = []  # will be (n_particles with S, 3)
        free_massive_gm_mask = []  # will be (n_particles with S)
        fixed_massive_gms = []  # will be ((n_particles - n_particles with S),)
        free_massive_gms = []  # will be (n_particles with S,)

        for p in particles:
            # if the particle has no initial state, (just observations),
            # randomly initialize it
            if type(p.x) == type(None):
                xs.append(np.random.uniform(-10, 10, size=3))
                vs.append(np.random.uniform(-0.1, 0.1, size=3))
            else:
                xs.append(p.x)
                vs.append(p.v)
            gms.append(p.gm)

            # set up the masks for which particles have free positions/velocities
            if p.free_orbit:  # if you want to fit the particle's state...
                if (
                    p.gm > 0.0
                ) | p.free_gm:  # if it's definitely a massive particle, or could be...
                    free_massive_state_mask.append(True)
                    free_massive_xs.append(p.x)
                    free_massive_vs.append(p.v)
                    TRACERS.append(False)
                else:  # if it's definitely a tracer particle
                    free_tracer_state_mask.append(True)
                    free_tracer_xs.append(p.x)
                    free_tracer_vs.append(p.v)
                    TRACERS.append(True)
            else:
                if (
                    p.gm > 0.0
                ) | p.free_gm:  # if it's definitely a massive particle, or could be...
                    free_massive_state_mask.append(False)
                    fixed_massive_xs.append(p.x)
                    fixed_massive_vs.append(p.v)
                    TRACERS.append(False)
                else:  # if it's definitely a tracer particle
                    free_tracer_state_mask.append(False)
                    fixed_tracer_xs.append(p.x)
                    fixed_tracer_vs.append(p.v)
                    TRACERS.append(True)

            # set up the masks for which particles have free gms
            if p.free_gm:
                free_massive_gm_mask.append(True)
                free_massive_gms.append(p.gm)
            else:
                if p.gm > 0:
                    free_massive_gm_mask.append(False)
                    fixed_massive_gms.append(p.gm)

        # lock in the current states of all particles
        self._xs = jnp.array(xs)
        self._vs = jnp.array(vs)
        self._gms = jnp.array(gms)

        # turn all of the definitely unchanging parameters into arrays/class attributes
        # the ones that can vary are left for now, will be collected into a dict later
        self._free_tracer_state_mask = jnp.array(free_tracer_state_mask).astype(bool)
        self._fixed_tracer_xs = jnp.array(fixed_tracer_xs)
        self._fixed_tracer_vs = jnp.array(fixed_tracer_vs)
        self._free_massive_state_mask = jnp.array(free_massive_state_mask).astype(bool)
        self._fixed_massive_xs = jnp.array(fixed_massive_xs)
        self._fixed_massive_vs = jnp.array(fixed_massive_vs)
        self._free_massive_gm_mask = jnp.array(free_massive_gm_mask).astype(bool)
        self._fixed_massive_gms = jnp.array(fixed_massive_gms)
        self._TRACERS = jnp.array(TRACERS)

        ########################################################################
        # Set up the planets/asteroids. First get the gms/Chebyshev coefficients
        # from the "construct_perturbers" module, if needed
        self._planets = planets
        self._asteroids = asteroids
        if (
            (planets != all_planets)
            | (asteroids != large_asteroids)
            | (self._earliest_time != Time("1980-01-01").tdb.jd)
            | (self._latest_time != Time("2100-01-01").tdb.jd)
        ):
            (
                self._planet_params,
                self._asteroid_params,
                self._planet_gms,
                self._asteroid_gms,
            ) = construct_perturbers(
                planets=planets,
                asteroids=asteroids,
                earliest_time=Time(self._earliest_time, format="jd")-10*u.day,
                latest_time=Time(self._latest_time, format="jd")+10*u.day,
            )
        else:
            self._planet_params = STANDARD_PLANET_PARAMS
            self._asteroid_params = STANDARD_ASTEROID_PARAMS
            self._planet_gms = STANDARD_PLANET_GMS
            self._asteroid_gms = STANDARD_ASTEROID_GMS

        assert len(self._planet_params[0]) == len(self._planets) + 1, 'Ephemeris could not be generated for at least one requested perturbing planet'
        assert len(self._asteroid_params[0]) == len(self._asteroids), 'Ephemeris could not be generated for at least one requested perturbing asteroid'

        # in case you want to let the masses vary, you can set up free/fixed
        # gms for the planets/asteroids just like we did for the particles
        if type(fit_planet_gms) != bool:
            assert len(fit_planet_gms) == len(
                planets
            ), "If fitting any planet gms, must provide a list of booleans indicating which to fit of length len(planets)"
            self._free_planet_gm_mask = fit_planet_gms
        else:
            self._free_planet_gm_mask = jnp.array([False] * len(self._planet_gms)).astype(
                bool
            )
        free_planet_gms = self._planet_gms[self._free_planet_gm_mask]
        self._fixed_planet_gms = self._planet_gms[~self._free_planet_gm_mask]

        if type(fit_asteroid_gms) != bool:
            assert len(fit_asteroid_gms) == len(
                asteroids
            ), "If fitting any asteroid gms, must provide a list of booleans indicating which to fit of length len(asteroids)"
            self._free_asteroid_gm_mask = fit_asteroid_gms
        else:
            self._free_asteroid_gm_mask = jnp.array([False] * len(asteroids)).astype(
                bool
            )

        if len(asteroids) == 0:
            self._free_asteroid_gm_mask = jnp.array([False])
            free_asteroid_gms = jnp.array([])
            self._fixed_asteroid_gms = self._asteroid_gms
        else:
            free_asteroid_gms = self._asteroid_gms[self._free_asteroid_gm_mask]
            self._fixed_asteroid_gms = self._asteroid_gms[~self._free_asteroid_gm_mask]

        ########################################################################
        # this is somewhat inelegant, but- for the jax.scans to work, all of the
        # particles need the same number of observations. so, we're going to
        # pad the observation_times, ras, decs, observer_positions, and uncertainties
        # so that the array isn't irregular
        # I don't love this design choice, since loglike will then carry out a bunch
        # of useless calculations. luckily you don't need to integrate the particle
        # for all of the dummy indecies since the "times" of the fake observations
        # are all the same, but you still go through the sky projection stuff
        # and only get saved by the 1/inf=0.0 in the uncertainty calculation

        # determine max # of observations
        largest_coverage = 0
        for p in particles:
            if type(p.observations) == type(None):
                continue
            if len(p.observations.times) > largest_coverage:
                largest_coverage = len(p.observations.times)

        # create empty padded arrays
        # Close to Jan 2020- something hopefully close to other observations so the integration isn't too long
        observation_times = jnp.ones((len(particles), largest_coverage)) * 2458849
        observation_ras = jnp.zeros((len(particles), largest_coverage))
        observation_decs = jnp.zeros((len(particles), largest_coverage))
        observer_positions = jnp.ones((len(particles), largest_coverage, 3)) * 999
        astrometry_uncertainties = (
            jnp.ones((len(particles), largest_coverage)) * jnp.inf
        )

        # fill in the padded arrays with the actual observations
        for i, p in enumerate(particles):
            if type(p.observations) == type(None):
                continue
            observation_times = observation_times.at[
                i, : len(p.observations.times)
            ].set(p.observations.times)
            observation_ras = observation_ras.at[i, : len(p.observations.ra)].set(
                p.observations.ra
            )
            observation_decs = observation_decs.at[i, : len(p.observations.dec)].set(
                p.observations.dec
            )
            observer_positions = observer_positions.at[
                i, : len(p.observations.observer_positions)
            ].set(p.observations.observer_positions)
            astrometry_uncertainties = astrometry_uncertainties.at[
                i, : len(p.observations.astrometry_uncertainties)
            ].set(p.observations.astrometry_uncertainties)

        # break up the padded arrays into the tracers and massive particles
        # these will not vary during fits (for now- might be nice to marginalize
        # over uncertainty in times for example), so we save them as class attributes
        self._tracer_particle_times = observation_times[self._TRACERS]
        self._tracer_particle_ras = observation_ras[self._TRACERS]
        self._tracer_particle_decs = observation_decs[self._TRACERS]
        self._tracer_particle_observer_positions = observer_positions[self._TRACERS]
        self._tracer_particle_astrometry_uncertainties = astrometry_uncertainties[
            self._TRACERS
        ]

        self._massive_particle_times = observation_times[~self._TRACERS]
        self._massive_particle_ras = observation_ras[~self._TRACERS]
        self._massive_particle_decs = observation_decs[~self._TRACERS]
        self._massive_particle_observer_positions = observer_positions[~self._TRACERS]
        self._massive_particle_astrometry_uncertainties = astrometry_uncertainties[
            ~self._TRACERS
        ]

        ########################################################################
        # here we'll group all of the parameters that can/can't vary in a fit
        # into different dicts. these will be combined via "prepare_loglike_input"
        # during actual fitting, which will produce a dict appropriate for "loglike"

        # these are the parameters that will definitely never change
        self._fixed_params = {
            "free_tracer_state_mask": self._free_tracer_state_mask,
            "fixed_tracer_xs": self._fixed_tracer_xs,
            "fixed_tracer_vs": self._fixed_tracer_vs,
            "free_massive_state_mask": self._free_massive_state_mask,
            "fixed_massive_xs": self._fixed_massive_xs,
            "fixed_massive_vs": self._fixed_massive_vs,
            "free_massive_gm_mask": self._free_massive_gm_mask,
            "fixed_massive_gms": self._fixed_massive_gms,
            "free_planet_gm_mask": self._free_planet_gm_mask,
            "fixed_planet_gms": self._fixed_planet_gms,
            "free_asteroid_gm_mask": self._free_asteroid_gm_mask,
            "fixed_asteroid_gms": self._fixed_asteroid_gms,
            "tracer_particle_times": self._tracer_particle_times,
            "tracer_particle_ras": self._tracer_particle_ras,
            "tracer_particle_decs": self._tracer_particle_decs,
            "tracer_particle_observer_positions": self._tracer_particle_observer_positions,
            "tracer_particle_astrometry_uncertainties": self._tracer_particle_astrometry_uncertainties,
            "massive_particle_times": self._massive_particle_times,
            "massive_particle_ras": self._massive_particle_ras,
            "massive_particle_decs": self._massive_particle_decs,
            "massive_particle_observer_positions": self._massive_particle_observer_positions,
            "massive_particle_astrometry_uncertainties": self._massive_particle_astrometry_uncertainties,
            "planet_params": self._planet_params,
            "asteroid_params": self._asteroid_params,
        }

        # here are the parameters that in principle could change
        # but, some of them are just empty arrays (i.e., if there are no particles
        # we want to fit a mass for). it doesn't make sense to differentiate
        # wrt those, so we'll add them to the fixed_params dict if they're empty
        self._free_params = {}
        if len(free_tracer_xs) == 0:
            self._fixed_params = {
                **self._fixed_params,
                **{"free_tracer_xs": jnp.empty((0, 3))},
            }
            self._fixed_params = {
                **self._fixed_params,
                **{"free_tracer_vs": jnp.empty((0, 3))},
            }
        else:
            self._free_params = {
                **self._free_params,
                **{"free_tracer_xs": jnp.array(free_tracer_xs)},
            }
            self._free_params = {
                **self._free_params,
                **{"free_tracer_vs": jnp.array(free_tracer_vs)},
            }

        if len(free_massive_xs) == 0:
            self._fixed_params = {
                **self._fixed_params,
                **{"free_massive_xs": jnp.empty((0, 3))},
            }
            self._fixed_params = {
                **self._fixed_params,
                **{"free_massive_vs": jnp.empty((0, 3))},
            }
        else:
            self._free_params = {
                **self._free_params,
                **{"free_massive_xs": jnp.array(free_massive_xs)},
            }
            self._free_params = {
                **self._free_params,
                **{"free_massive_vs": jnp.array(free_massive_vs)},
            }

        if len(free_massive_gms) == 0:
            self._fixed_params = {
                **self._fixed_params,
                **{"free_massive_gms": jnp.empty((0))},
            }
        else:
            self._free_params = {
                **self._free_params,
                **{"free_massive_gms": jnp.array(free_massive_gms)},
            }

        if len(free_planet_gms) == 0:
            self._fixed_params = {
                **self._fixed_params,
                **{"free_planet_gms": jnp.empty((0))},
            }
        else:
            self._free_params = {
                **self._free_params,
                **{"free_planet_gms": jnp.array(free_planet_gms)},
            }

        if len(free_asteroid_gms) == 0:
            self._fixed_params = {
                **self._fixed_params,
                **{"free_asteroid_gms": jnp.empty((0))},
            }
        else:
            self._free_params = {
                **self._free_params,
                **{"free_asteroid_gms": jnp.array(free_asteroid_gms)},
            }

        ########################################################################
        # check that everything just worked
        assert len(self._xs) == len(self._vs) == len(self._gms)
        assert len(self._fixed_params) + len(self._free_params) == 31
        assert len(self._TRACERS) == len(self._xs)

        assert jnp.sum(self._TRACERS) == len(self._free_tracer_state_mask)
        assert jnp.sum(self._free_tracer_state_mask) == len(free_tracer_xs)
        assert jnp.sum(self._free_tracer_state_mask) == len(free_tracer_vs)
        assert jnp.sum(~self._free_tracer_state_mask) == len(self._fixed_tracer_xs)
        assert jnp.sum(~self._free_tracer_state_mask) == len(self._fixed_tracer_vs)

        assert jnp.sum(~self._TRACERS) == len(self._free_massive_state_mask)
        assert jnp.sum(self._free_massive_state_mask) == len(free_massive_xs)
        assert jnp.sum(self._free_massive_state_mask) == len(free_massive_vs)
        assert jnp.sum(~self._free_massive_state_mask) == len(self._fixed_massive_xs)
        assert jnp.sum(~self._free_massive_state_mask) == len(self._fixed_massive_vs)

        assert jnp.sum(~self._TRACERS) == len(self._free_massive_gm_mask)
        assert jnp.sum(self._free_massive_gm_mask) == len(free_massive_gms)
        assert jnp.sum(~self._free_massive_gm_mask) == len(self._fixed_massive_gms)

        assert len(self._planet_gms) == len(self._free_planet_gm_mask)
        assert jnp.sum(self._free_planet_gm_mask) == len(free_planet_gms)
        assert jnp.sum(~self._free_planet_gm_mask) == len(self._fixed_planet_gms)

        assert len(self._asteroid_gms) == len(self._free_asteroid_gm_mask)
        assert jnp.sum(self._free_asteroid_gm_mask) == len(free_asteroid_gms)
        assert jnp.sum(~self._free_asteroid_gm_mask) == len(self._fixed_asteroid_gms)

        assert (
            len(self._tracer_particle_times)
            == len(self._tracer_particle_ras)
            == len(self._tracer_particle_decs)
            == len(self._tracer_particle_observer_positions)
            == len(self._tracer_particle_astrometry_uncertainties)
        )

        assert (
            len(self._massive_particle_times)
            == len(self._massive_particle_ras)
            == len(self._massive_particle_decs)
            == len(self._massive_particle_observer_positions)
            == len(self._massive_particle_astrometry_uncertainties)
        )

    def __repr__(self):
        return f"System with {len(self._xs)} particles"

    def __len__(self):
        return len(self._xs)

    ################################################################################
    # Methods
    ################################################################################
    def add_particle(self):
        pass


    def propagate(self, times, use_GR=False, obey_large_step_limits=True, sky_positions=False,
                  observatory_locations=[]):
            
            # get "times" into an array of (n_times,)
            if isinstance(times, type(Time("2023-01-01"))):
                times = jnp.array(times.tdb.jd)
            elif isinstance(times, list):
                times = jnp.array([t.tdb.jd for t in times])
            if times.shape == ():
                times = jnp.array([times])

            assert jnp.max(times) < self._latest_time, "Requested propagation includes times beyond the latest time in considered in the ephemeris for this particle. Consider initially setting a broader time range for the ephemeris."
            assert jnp.min(times) > self._earliest_time, "Requested propagation includes times before the earliest time in considered in the ephemeris for this particle. Consider initially setting a broader time range for the ephemeris."
            jumps = jnp.abs(jnp.diff(times))
            if jumps.shape != (0,): largest_jump = jnp.max(jumps)
            else: largest_jump = 0
            first_jump = jnp.abs(self._time - times[0])
            largest_jump = jnp.where(first_jump > largest_jump, first_jump, largest_jump)
            if obey_large_step_limits:
                assert largest_jump <= 7305, "Requested propagation includes at least one step that is too large- max default is 20 years. Shrink the jumps, or set obey_large_step_limits to False."
            if largest_jump < 1000: max_steps = jnp.arange(100)
            else: max_steps = jnp.arange(1000)

            if not obey_large_step_limits and largest_jump > 1000: max_steps = jnp.arange((largest_jump*1.25 / 12).astype(int))

            xs, vs, final_times, success = integrate_multiple(
                xs=self._xs,
                vs=self._vs,
                gms=self._gms,
                initial_time=self._time,
                final_times=times,
                planet_params=self._planet_params,
                asteroid_params=self._asteroid_params,
                planet_gms=self._planet_gms,
                asteroid_gms=self._asteroid_gms,
                max_steps=max_steps,
                use_GR=use_GR,
            )

            self._xs = xs
            self._vs = vs
            self._time = final_times[-1]



            if sky_positions:
                if observatory_locations == []:
                    raise ValueError("Must provide observatory locations if on_sky=True. See Observations docstring for more info.")
                pos = [SkyCoord(0*u.deg, 0*u.deg)]*len(times)
                obs = Observations(positions=pos, times=times,
                                   observatory_locations=observatory_locations,
                                   astrometry_uncertainties=[10*u.mas]*len(times))
                
                ras, decs = on_sky(
                    xs=xs.reshape(-1, xs.shape[-1]),
                    vs=vs.reshape(-1, vs.shape[-1]),
                    gms=jnp.tile(self._gms, len(times)),
                    times=jnp.tile(times, len(xs)),
                    observer_positions=jnp.array(list(obs.observer_positions)*len(xs)),
                    planet_params=self._planet_params,
                    asteroid_params=self._asteroid_params,
                    planet_gms=self._planet_gms,
                    asteroid_gms=self._asteroid_gms,
                )

                s = SkyCoord(ra=ras*u.rad, dec=decs*u.rad)
                if xs.shape[0] == 1:
                    return xs[0], vs[0], s
                return xs, vs, s.reshape((xs.shape[0], xs.shape[1]))

            if xs.shape[0] == 1:
                return xs[0], vs[0]
            return xs, vs

    ################################################################################
    # Properties
    ################################################################################

    @property
    def xs(self):
        if self._xs.shape[0] == 1:
            return self._xs[0]
        return self._xs

    @xs.setter
    def xs(self, value):
        raise AttributeError("cannot change xs directly- use propagate(), which will update the entire state of the system") from None

    @property
    def vs(self):
        if self._vs.shape[0] == 1:
            return self._vs[0]
        return self._vs

    @vs.setter
    def vs(self, value):
        raise AttributeError("cannot change vs directly- use propagate(), which will update the entire state of the system") from None

    @property
    def gms(self):
        if self._gms.shape[0] == 1:
            return self._gms[0]
        return self._gms

    @gms.setter
    def gms(self, value):
        raise AttributeError("cannot change gms ") from None
    
    @property
    def time(self):
        return self._time
    
    @property
    def loglike(self, use_GR=True, obey_large_step_limits=True):
        largest = 0
        for p in jnp.concatenate((self._tracer_particle_times, self._massive_particle_times)):
            times = p[p != 2458849]
            if times.shape == (0,): continue
            jumps = jnp.abs(jnp.diff(times))
            if jumps.shape != (0,): largest_jump = jnp.max(jumps)
            else: largest_jump = 0
            first_jump = jnp.abs(self._time - times[0])
            largest_jump = jnp.where(first_jump > largest_jump, first_jump, largest_jump)
            largest = jnp.where(largest_jump > largest, largest_jump, largest)
        if obey_large_step_limits:
            assert largest_jump <= 7305, "Requested propagation includes at least one step that is too large- max default is 20 years. Shrink the jumps, or set obey_large_step_limits to False."
        if largest_jump < 1000: max_steps = jnp.arange(100)
        else: max_steps = jnp.arange(1000)

        if not obey_large_step_limits and largest_jump > 1000: max_steps = jnp.arange((largest_jump*1.25 / 12).astype(int))

        d = prepare_loglike_input(free_params=self._free_params, fixed_params=self._fixed_params,
                          use_GR=use_GR, max_steps=max_steps)
        return loglike(d)
    
    @property
    def residuals(self):
        pass

    @property
    def particles(self):
        # collapse the current state back into a list of particles
        pass

    @property
    def elements(self):
        pass





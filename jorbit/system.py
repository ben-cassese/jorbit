import jax
from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore", module="erfa")
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm

from jorbit import Observations, Particle
from jorbit.engine.utils import construct_perturbers
from jorbit.engine.utils import (
    prep_gj_integrator_single,
    prep_gj_integrator_multiple,
    prep_uneven_GJ_integrator,
)
from jorbit.engine.ephemeris import planet_state
from jorbit.engine.sky_projection import on_sky
from jorbit.engine.likelihood import tracer_likelihoods, massive_likelihoods
from jorbit.engine.gauss_jackson_integrator import gj_integrate, gj_integrate_multiple

from jorbit.data import (
    STANDARD_PLANET_PARAMS,
    STANDARD_ASTEROID_PARAMS,
    STANDARD_PLANET_GMS,
    STANDARD_ASTEROID_GMS,
)
from jorbit.data.constants import (
    ALL_PLANETS,
    LARGE_ASTEROIDS,
    GJ6_A,
    GJ6_B,
    GJ8_A,
    GJ8_B,
    GJ10_A,
    GJ10_B,
    GJ12_A,
    GJ12_B,
    GJ14_A,
    GJ14_B,
    GJ16_A,
    GJ16_B,
    Y4_C,
    Y4_D,
    Y6_C,
    Y6_D,
    Y8_C,
    Y8_D,
)

coeffs_dict = {
    "a_jk": {
        "6": GJ6_A,
        "8": GJ8_A,
        "10": GJ10_A,
        "12": GJ12_A,
        "14": GJ14_A,
        "16": GJ16_A,
    },
    "b_jk": {
        "6": GJ6_B,
        "8": GJ8_B,
        "10": GJ10_B,
        "12": GJ12_B,
        "14": GJ14_B,
        "16": GJ16_B,
    },
    "yc": {
        "4": Y4_C,
        "6": Y6_C,
        "8": Y8_C,
    },
    "yd": {
        "4": Y4_D,
        "6": Y6_D,
        "8": Y8_D,
    },
}


class System:
    def __init__(
        self,
        particles,
        planets=ALL_PLANETS,
        asteroids=LARGE_ASTEROIDS,
        integrator_order=8,
        warmup_order=4,
        likelihood_timestep=4.0,
        force_common_epoch=True,
        parallelize=False,
        fit_planet_gms=False,
        fit_asteroid_gms=False,
    ):
        self._particles = particles  # need to re-order though, saved to compare times
        self._planets = planets
        self._asteroids = asteroids
        self._integrator_order = integrator_order
        self._warmup_order = warmup_order
        self._likelihood_timestep = likelihood_timestep
        self._force_common_epoch = force_common_epoch
        self._parallelize = parallelize
        self._fit_planet_gms = fit_planet_gms
        self._fit_asteroid_gms = fit_asteroid_gms

        self._input_checks()

        (
            self._earliest_time,
            self._latest_time,
            self._planet_params,
            self._asteroid_params,
            self._planet_gms,
            self._asteroid_gms,
        ) = self._initialize_planets()

        self._particles, self._time = self._initialize_particles(self._particles)

        (
            self._free_params,
            self._fixed_params,
            self._ordered_tracer_obs,
            self._ordered_massive_obs,
            self._particles,
            self._particle_indecies,
        ) = self._create_free_fixed_params()

        self._padded_supporting_data = self._prep_system_GJ_integrator()

        if self._parallelize:
            self._residual_func, self._likelihood_func = (
                self._generate_parallelized_likelihood_funcs()
            )
        else:
            self._residual_func, self._likelihood_func = (
                self._generate_single_device_likelihood_funcs()
            )

        self._xs = jnp.array([p.x for p in self._particles])
        self._vs = jnp.array([p.v for p in self._particles])
        self._gms = jnp.array([p.gm for p in self._particles])

    def __repr__(self):
        return f"System with {len(self._xs)} particles"

    def __len__(self):
        return len(self._xs)

    ################################################################################
    # Misc general heler methods
    ################################################################################
    def _input_checks(self):
        num_devices = jax.local_device_count()
        if self._parallelize:
            assert num_devices > 1, (
                "Must have more than one device to parallelize. If using CPUs, try"
                " setting the enivronment variable"
                " XLA_FLAGS='--xla_force_host_platform_device_count=n' to emulate n"
                " devices. As of writing, "
                "XLA backend for multiple CPUs is under active development"
                " (https://github.com/google/jax/issues/1408)"
            )

        assert self._integrator_order in [
            6,
            8,
            10,
            12,
            14,
            16,
        ], "Integrator order must be 6, 8, 10, 12, 14, or 16"

        assert self._warmup_order in [4, 6, 8], "Warmup order must be 4, 6, or 8"

        if not self._force_common_epoch:
            t = self._particles[0].time
            for p in self.particles:
                assert p.time == t, (
                    "If not forcing common epoch, all particles must"
                    " be initialized at the same time"
                )

        names = []
        for p in self._particles:
            names.append(p.name)
        num_unique_names = len(set(names))
        assert num_unique_names == len(self._particles), (
            "2 or more identical names detected. If using custom names, make sure"
            " they are each unique and not 'Particle (int < # of particles in the"
            " system)'"
        )

    def _initialize_planets(self):
        earlys = []
        lates = []

        for p in self._particles:
            earlys.append(p.earliest_time.tdb.jd)
            lates.append(p.latest_time.tdb.jd)

        earliest_time = jnp.min(jnp.array(earlys))
        latest_time = jnp.max(jnp.array(lates))

        if (
            (self._planets != ALL_PLANETS)
            | (self._asteroids != LARGE_ASTEROIDS)
            | (earliest_time != Time("1980-01-01").tdb.jd)
            | (latest_time != Time("2100-01-01").tdb.jd)
        ):
            (
                planet_params,
                asteroid_params,
                planet_gms,
                asteroid_gms,
            ) = construct_perturbers(
                planets=self._planets,
                asteroids=self._asteroids,
                earliest_time=Time(earliest_time, format="jd") - 10 * u.day,
                latest_time=Time(latest_time, format="jd") + 10 * u.day,
            )
        else:
            planet_params = STANDARD_PLANET_PARAMS
            asteroid_params = STANDARD_ASTEROID_PARAMS
            planet_gms = STANDARD_PLANET_GMS
            asteroid_gms = STANDARD_ASTEROID_GMS

        assert len(planet_params[0]) == len(self._planets) + 1, (
            "Ephemeris could not be generated for at least one requested perturbing"
            " planet"
        )
        assert len(asteroid_params[0]) == len(self._asteroids), (
            "Ephemeris could not be generated for at least one requested perturbing"
            " asteroid"
        )

        return (
            earliest_time,
            latest_time,
            planet_params,
            asteroid_params,
            planet_gms,
            asteroid_gms,
        )

    def _initialize_particles(self, particles):
        if self._force_common_epoch:
            # find the mean epoch of all observations, weighted by the precision of those
            # observations. There might be a better way to do this.
            # Also, right now I'm forcing the whole system to have a common epoch. In theory
            # the tracer particles could all have their own optimial epochs
            times = []
            weights = []
            for p in particles:
                if p.observations != None:
                    times.append(p.observations.times)
                    weights.append(1 / p.observations.astrometric_uncertainties**2)
            times = jnp.concatenate(times)
            weights = jnp.concatenate(weights)
            epoch = jnp.sum(times * weights) / jnp.sum(weights)

            # TODO this is only until I figure out how to split particles into leading and trailing epochs
            epoch = jnp.min(times) - 1e-8

        # _input_checks already verified all particles have same epoch
        else:
            epoch = particles[0].time

        # move each particles to the mean epoch. To do this, we neglect the influence of
        # massive particles and evolve each particle only under the influence of fixed
        # solar system perturbers
        blank_obs = Observations(
            observed_coordinates=SkyCoord(0 * u.deg, 0 * u.deg),
            times=jnp.array([epoch]),
            observatory_locations=jnp.array([[999.0, 999, 999]]),
            astrometric_uncertainties=jnp.inf * u.arcsec,
            verbose_downloading=False,
            mpc_file=None,
        )
        modified_particles = []
        for p in particles:
            if p.time != epoch:
                (
                    planet_xs,
                    planet_vs,
                    planet_as,
                    asteroid_xs,
                    planet_xs_warmup,
                    asteroid_xs_warmup,
                    dts_warmup,
                ) = prep_gj_integrator_single(
                    t0=p.time,
                    tf=epoch,
                    jumps=max(
                        (
                            jnp.abs(
                                jnp.ceil(
                                    (p.time - epoch) / self._likelihood_timestep
                                ).astype(int)
                            ),
                            self._integrator_order + 2,
                        )
                    ),
                    a_jk=coeffs_dict["a_jk"][str(self._integrator_order)],
                    planet_params=self._planet_params,
                    asteroid_params=self._asteroid_params,
                )
                x, v = gj_integrate(
                    x0=jnp.array([p.x]),
                    v0=jnp.array([p.v]),
                    gms=jnp.array([p.gm]),
                    b_jk=coeffs_dict["b_jk"][str(self._integrator_order)],
                    a_jk=coeffs_dict["a_jk"][str(self._integrator_order)],
                    t0=p.time,
                    tf=epoch,
                    valid_steps=planet_xs.shape[1],
                    planet_xs=planet_xs,
                    planet_vs=planet_vs,
                    planet_as=planet_as,
                    asteroid_xs=asteroid_xs,
                    planet_xs_warmup=planet_xs_warmup,
                    asteroid_xs_warmup=asteroid_xs_warmup,
                    dts_warmup=dts_warmup,
                    warmup_C=coeffs_dict["yc"][str(self._warmup_order)],
                    warmup_D=coeffs_dict["yd"][str(self._warmup_order)],
                    planet_gms=self._planet_gms,
                    asteroid_gms=self._asteroid_gms,
                    use_GR=True,
                )
                x = x[0]
                v = v[0]
            else:
                x = p.x
                v = p.v

            if p.observations != None:
                obs = p.observations + blank_obs
            else:
                obs = None

            new_particle = Particle(
                x=x,
                v=v,
                elements=None,
                gm=p.gm,
                time=epoch,
                observations=obs,
                earliest_time=p.earliest_time,
                latest_time=p.latest_time,
                name=p.name,
                free_orbit=p.free_orbit,
                free_gm=p.free_gm,
            )
            modified_particles.append(new_particle)
        return modified_particles, epoch

    def _create_free_fixed_params(self):
        # f for Free, r for Rigid
        tracer_fx_rm = {"x": [], "v": [], "gm": [], "obs": [], "particles": [], "k": []}
        tracer_rx_rm = {"x": [], "v": [], "gm": [], "obs": [], "particles": [], "k": []}
        massive_fx_rm = {
            "x": [],
            "v": [],
            "gm": [],
            "obs": [],
            "particles": [],
            "k": [],
        }
        massive_rx_fm = {
            "x": [],
            "v": [],
            "gm": [],
            "obs": [],
            "particles": [],
            "k": [],
        }
        massive_fx_fm = {
            "x": [],
            "v": [],
            "gm": [],
            "obs": [],
            "particles": [],
            "k": [],
        }
        massive_rx_rm = {
            "x": [],
            "v": [],
            "gm": [],
            "obs": [],
            "particles": [],
            "k": [],
        }

        for i, p in enumerate(self._particles):
            if p.free_orbit and p.free_gm:
                massive_fx_fm["x"].append(p.x)
                massive_fx_fm["v"].append(p.v)
                massive_fx_fm["gm"].append(p.gm)
                massive_fx_fm["obs"].append(p.observations)
                massive_fx_fm["particles"].append(p)
                massive_fx_fm["k"].append(i)
            elif p.free_orbit and not p.free_gm:
                if p.gm == 0:
                    tracer_fx_rm["x"].append(p.x)
                    tracer_fx_rm["v"].append(p.v)
                    tracer_fx_rm["gm"].append(p.gm)
                    tracer_fx_rm["obs"].append(p.observations)
                    tracer_fx_rm["particles"].append(p)
                    tracer_fx_rm["k"].append(i)
                else:
                    massive_fx_rm["x"].append(p.x)
                    massive_fx_rm["v"].append(p.v)
                    massive_fx_rm["gm"].append(p.gm)
                    massive_fx_rm["obs"].append(p.observations)
                    massive_fx_rm["particles"].append(p)
                    massive_fx_rm["k"].append(i)
            elif not p.free_orbit and p.free_gm:
                massive_rx_fm["x"].append(p.x)
                massive_rx_fm["v"].append(p.v)
                massive_rx_fm["gm"].append(p.gm)
                massive_rx_fm["obs"].append(p.observations)
                massive_rx_fm["particles"].append(p)
                massive_rx_fm["k"].append(i)
            elif not p.free_orbit and not p.free_gm:
                if p.gm == 0:
                    tracer_rx_rm["x"].append(p.x)
                    tracer_rx_rm["v"].append(p.v)
                    tracer_rx_rm["gm"].append(p.gm)
                    tracer_rx_rm["obs"].append(p.observations)
                    tracer_rx_rm["particles"].append(p)
                    tracer_rx_rm["k"].append(i)
                else:
                    massive_rx_rm["x"].append(p.x)
                    massive_rx_rm["v"].append(p.v)
                    massive_rx_rm["gm"].append(p.gm)
                    massive_rx_rm["obs"].append(p.observations)
                    massive_rx_rm["particles"].append(p)
                    massive_rx_rm["k"].append(i)

        fixed_params = {}
        # if len(tracer_fx_rm["x"]) > 0:
        #     fixed_params["tracer_fx_rm__gm"] = jnp.array(tracer_fx_rm["gm"])
        # else:
        #     fixed_params["tracer_fx_rm__gm"] = jnp.empty((0,))
        if len(tracer_rx_rm["x"]) > 0:
            fixed_params["tracer_rx_rm__x"] = jnp.array(tracer_rx_rm["x"])
            fixed_params["tracer_rx_rm__v"] = jnp.array(tracer_rx_rm["v"])
            # fixed_params["tracer_rx_rm__gm"] = jnp.array(tracer_rx_rm["gm"])
        else:
            fixed_params["tracer_rx_rm__x"] = jnp.empty((0, 3))
            fixed_params["tracer_rx_rm__v"] = jnp.empty((0, 3))
            # fixed_params["tracer_rx_rm__gm"] = jnp.empty((0,))
        if len(massive_fx_rm["x"]) > 0:
            fixed_params["massive_fx_rm__gm"] = jnp.array(massive_fx_rm["gm"])
        else:
            fixed_params["massive_fx_rm__gm"] = jnp.empty((0,))
        if len(massive_rx_fm["x"]) > 0:
            fixed_params["massive_rx_fm__x"] = jnp.array(massive_rx_fm["x"])
            fixed_params["massive_rx_fm__v"] = jnp.array(massive_rx_fm["v"])
        else:
            fixed_params["massive_rx_fm__x"] = jnp.empty((0, 3))
            fixed_params["massive_rx_fm__v"] = jnp.empty((0, 3))
        if len(massive_rx_rm["x"]) > 0:
            fixed_params["massive_rx_rm__x"] = jnp.array(massive_rx_rm["x"])
            fixed_params["massive_rx_rm__v"] = jnp.array(massive_rx_rm["v"])
            fixed_params["massive_rx_rm__gm"] = jnp.array(massive_rx_rm["gm"])
        else:
            fixed_params["massive_rx_rm__x"] = jnp.empty((0, 3))
            fixed_params["massive_rx_rm__v"] = jnp.empty((0, 3))
            fixed_params["massive_rx_rm__gm"] = jnp.empty((0,))

        free_params = {}
        if len(tracer_fx_rm["x"]) > 0:
            free_params["tracer_fx_rm__x"] = jnp.array(tracer_fx_rm["x"])
            free_params["tracer_fx_rm__v"] = jnp.array(tracer_fx_rm["v"])
        else:
            fixed_params["tracer_fx_rm__x"] = jnp.empty((0, 3))
            fixed_params["tracer_fx_rm__v"] = jnp.empty((0, 3))

        if len(massive_fx_rm["x"]) > 0:
            free_params["massive_fx_rm__x"] = jnp.array(massive_fx_rm["x"])
            free_params["massive_fx_rm__v"] = jnp.array(massive_fx_rm["v"])
        else:
            fixed_params["massive_fx_rm__x"] = jnp.empty((0, 3))
            fixed_params["massive_fx_rm__v"] = jnp.empty((0, 3))

        if len(massive_rx_fm["x"]) > 0:
            free_params["massive_rx_fm__gm"] = jnp.array(massive_rx_fm["gm"])
        else:
            fixed_params["massive_rx_fm__gm"] = jnp.empty((0,))

        if len(massive_fx_fm["x"]) > 0:
            free_params["massive_fx_fm__x"] = jnp.array(massive_fx_fm["x"])
            free_params["massive_fx_fm__v"] = jnp.array(massive_fx_fm["v"])
            free_params["massive_fx_fm__gm"] = jnp.array(massive_fx_fm["gm"])
        else:
            fixed_params["massive_fx_fm__x"] = jnp.empty((0, 3))
            fixed_params["massive_fx_fm__v"] = jnp.empty((0, 3))
            fixed_params["massive_fx_fm__gm"] = jnp.empty((0,))

        if self._fit_planet_gms:
            free_params["planet_gms"] = self._planet_gms
        else:
            fixed_params["planet_gms"] = self._planet_gms

        if self._fit_asteroid_gms:
            free_params["asteroid_gms"] = self._asteroid_gms
        else:
            fixed_params["asteroid_gms"] = self._asteroid_gms

        ordered_tracer_obs = tracer_fx_rm["obs"] + tracer_rx_rm["obs"]
        ordered_massive_obs = (
            massive_fx_rm["obs"]
            + massive_rx_fm["obs"]
            + massive_fx_fm["obs"]
            + massive_rx_rm["obs"]
        )

        reordered_particles = (
            tracer_fx_rm["particles"]
            + tracer_rx_rm["particles"]
            + massive_fx_rm["particles"]
            + massive_rx_fm["particles"]
            + massive_fx_fm["particles"]
            + massive_rx_rm["particles"]
        )

        unscramble_key = (
            tracer_fx_rm["k"]
            + tracer_rx_rm["k"]
            + massive_fx_rm["k"]
            + massive_rx_fm["k"]
            + massive_fx_fm["k"]
            + massive_rx_rm["k"]
        )

        return (
            free_params,
            fixed_params,
            ordered_tracer_obs,
            ordered_massive_obs,
            reordered_particles,
            unscramble_key,
        )

    def _prep_system_GJ_integrator(self):
        def _inner_prep_system_GJ_integrator(ordered_obs, message):
            # The the data needed to integrate each individual particle to the times it
            # was observed. Each set of arrays for an individual particle is padded with 999s
            # so that there is the same number of jumps between epochs, but then the number of
            # epochs and the number of jumps per epoch will vary particle-to-particle
            individual_integrator_prep = []
            # times=jnp.concatenate((jnp.array([system_time]), o.times)),
            for o in tqdm(ordered_obs, desc=message, position=0):
                z = prep_uneven_GJ_integrator(
                    times=o.times,
                    low_order=self._integrator_order,
                    high_order=self._integrator_order,
                    low_order_cutoff=1e-6,
                    targeted_low_order_timestep=self._likelihood_timestep,
                    targeted_high_order_timestep=self._likelihood_timestep,
                    coeffs_dict=coeffs_dict,
                    planet_params=self._planet_params,
                    asteroid_params=self._asteroid_params,
                    ragged=False,
                )
                individual_integrator_prep.append(z)

            # Now pad those precomputed (and already padded) arrays so that every particle
            # has the same number of epochs and the same number of jumps per epoch
            most_jumps = 0
            most_steps_per_epoch = 0
            for p in individual_integrator_prep:
                p = p[1]  # Planet_Xs. p[0] is Valid_Steps
                if p.shape[0] > most_jumps:
                    most_jumps = p.shape[0]
                if p.shape[2] > most_steps_per_epoch:
                    most_steps_per_epoch = p.shape[2]
            Valid_Steps = []
            Planet_Xs = []
            Planet_Vs = []
            Planet_As = []
            Asteroid_Xs = []
            Planet_Xs_Warmup = []
            Asteroid_Xs_Warmup = []
            Dts_Warmup = []
            for p in individual_integrator_prep:
                Valid_Steps.append(
                    jnp.pad(
                        p[0],
                        (0, most_jumps - p[0].shape[0]),
                        mode="constant",
                        constant_values=999,
                    )
                )
                Planet_Xs.append(
                    jnp.pad(
                        p[1],
                        (
                            (0, most_jumps - p[1].shape[0]),
                            (0, 0),
                            (0, most_steps_per_epoch - p[1].shape[2]),
                            (0, 0),
                        ),
                        mode="constant",
                        constant_values=999,
                    )
                )
                Planet_Vs.append(
                    jnp.pad(
                        p[2],
                        (
                            (0, most_jumps - p[2].shape[0]),
                            (0, 0),
                            (0, most_steps_per_epoch - p[2].shape[2]),
                            (0, 0),
                        ),
                        mode="constant",
                        constant_values=999,
                    )
                )
                Planet_As.append(
                    jnp.pad(
                        p[3],
                        (
                            (0, most_jumps - p[3].shape[0]),
                            (0, 0),
                            (0, most_steps_per_epoch - p[3].shape[2]),
                            (0, 0),
                        ),
                        mode="constant",
                        constant_values=999,
                    )
                )
                Asteroid_Xs.append(
                    jnp.pad(
                        p[4],
                        (
                            (0, most_jumps - p[4].shape[0]),
                            (0, 0),
                            (0, most_steps_per_epoch - p[4].shape[2]),
                            (0, 0),
                        ),
                        mode="constant",
                        constant_values=999,
                    )
                )
                Planet_Xs_Warmup.append(
                    jnp.pad(
                        p[5],
                        (
                            (0, most_jumps - p[5].shape[0]),
                            (0, 0),
                            (0, 0),
                            (0, 0),
                            (0, 0),
                            (0, 0),
                        ),
                        mode="constant",
                        constant_values=999,
                    )
                )
                Asteroid_Xs_Warmup.append(
                    jnp.pad(
                        p[6],
                        (
                            (0, most_jumps - p[6].shape[0]),
                            (0, 0),
                            (0, 0),
                            (0, 0),
                            (0, 0),
                            (0, 0),
                        ),
                        mode="constant",
                        constant_values=999,
                    )
                )
                Dts_Warmup.append(
                    jnp.pad(
                        p[7],
                        ((0, most_jumps - p[7].shape[0]), (0, 0), (0, 0)),
                    )
                )
            if len(Valid_Steps) > 0:
                Valid_Steps = jnp.stack(Valid_Steps)
                Planet_Xs = jnp.stack(Planet_Xs)
                Planet_Vs = jnp.stack(Planet_Vs)
                Planet_As = jnp.stack(Planet_As)
                Asteroid_Xs = jnp.stack(Asteroid_Xs)
                Planet_Xs_Warmup = jnp.stack(Planet_Xs_Warmup)
                Asteroid_Xs_Warmup = jnp.stack(Asteroid_Xs_Warmup)
                Dts_Warmup = jnp.stack(Dts_Warmup)
            else:
                Valid_Steps = jnp.empty((0, 0))
                Planet_Xs = jnp.empty((0, 0, 0, 0, 0))
                Planet_Vs = jnp.empty((0, 0, 0, 0, 0))
                Planet_As = jnp.empty((0, 0, 0, 0, 0))
                Asteroid_Xs = jnp.empty((0, 0, 0, 0, 0))
                Planet_Xs_Warmup = jnp.empty((0, 0, 0, 0, 0, 0))
                Asteroid_Xs_Warmup = jnp.empty((0, 0, 0, 0, 0, 0))
                Dts_Warmup = jnp.empty((0, 0, 0))

            # unrelated, we need the initial times, jump times, RAs, Decs, astrometric
            # uncertainties, and location from which each observation was taken for each
            # particle. Every but the initial times need to be padded to match the number of
            # epochs, which is now the same for all particles
            init_times = []
            jump_times = []
            ras = []
            decs = []
            astrometric_uncertainties = []
            observer_positions = []
            observed_planet_xs = []
            observed_asteroid_xs = []
            for o in ordered_obs:
                init_times.append(o.times[0])
                jump_times.append(
                    jnp.pad(
                        o.times[1:],
                        (0, most_jumps - len(o.times) + 1),
                        mode="constant",
                        constant_values=999,
                    )
                )
                ras.append(
                    jnp.pad(
                        o.ra,
                        (0, most_jumps - len(o.times) + 1),
                        mode="constant",
                        constant_values=999,
                    ),
                )
                decs.append(
                    jnp.pad(
                        o.dec,
                        (0, most_jumps - len(o.times) + 1),
                        mode="constant",
                        constant_values=999,
                    ),
                )
                astrometric_uncertainties.append(
                    jnp.pad(
                        o.astrometric_uncertainties,
                        (0, most_jumps - len(o.times) + 1),
                        mode="constant",
                        constant_values=jnp.inf,
                    ),
                )
                observer_positions.append(
                    jnp.pad(
                        o.observer_positions,
                        ((0, most_jumps - len(o.times) + 1), (0, 0)),
                        mode="constant",
                        constant_values=999,
                    ),
                )
                planetxs, _, _ = planet_state(
                    planet_params=self._planet_params,
                    times=o.times,
                    velocity=False,
                    acceleration=False,
                )
                asteroidxs, _, _ = planet_state(
                    planet_params=self._asteroid_params,
                    times=o.times,
                    velocity=False,
                    acceleration=False,
                )
                observed_planet_xs.append(
                    jnp.pad(
                        planetxs,
                        ((0, 0), (0, most_jumps - len(o.times) + 1), (0, 0)),
                        mode="constant",
                        constant_values=999,
                    ),
                )
                observed_asteroid_xs.append(
                    jnp.pad(
                        asteroidxs,
                        ((0, 0), (0, most_jumps - len(o.times) + 1), (0, 0)),
                        mode="constant",
                        constant_values=999,
                    ),
                )

            if len(init_times) > 0:
                init_times = jnp.array(init_times)
                jump_times = jnp.stack(jump_times)
                ras = jnp.stack(ras)
                decs = jnp.stack(decs)
                astrometric_uncertainties = jnp.stack(astrometric_uncertainties)
                observer_positions = jnp.stack(observer_positions)
                observed_planet_xs = jnp.stack(observed_planet_xs)
                observed_asteroid_xs = jnp.stack(observed_asteroid_xs)
            else:
                init_times = jnp.empty((0,))
                jump_times = jnp.empty((0, 0))
                ras = jnp.empty((0, 0))
                decs = jnp.empty((0, 0))
                astrometric_uncertainties = jnp.empty((0, 0))
                observer_positions = jnp.empty((0, 0, 0))
                observed_planet_xs = jnp.empty((0, 0, 0, 0))
                observed_asteroid_xs = jnp.empty((0, 0, 0, 0))

            return (
                Valid_Steps,
                Planet_Xs,
                Planet_Vs,
                Planet_As,
                Asteroid_Xs,
                Planet_Xs_Warmup,
                Asteroid_Xs_Warmup,
                Dts_Warmup,
                init_times,
                jump_times,
                ras,
                decs,
                astrometric_uncertainties,
                observer_positions,
                observed_planet_xs,
                observed_asteroid_xs,
            )

        (
            tracer_Valid_Steps,
            tracer_Planet_Xs,
            tracer_Planet_Vs,
            tracer_Planet_As,
            tracer_Asteroid_Xs,
            tracer_Planet_Xs_Warmup,
            tracer_Asteroid_Xs_Warmup,
            tracer_Dts_Warmup,
            tracer_Init_Times,
            tracer_Jump_Times,
            tracer_RAs,
            tracer_Decs,
            tracer_Astrometric_Uncertainties,
            tracer_Observer_Positions,
            tracer_Observed_Planet_Xs,
            tracer_Observed_Asteroid_Xs,
        ) = _inner_prep_system_GJ_integrator(
            self._ordered_tracer_obs,
            "Pre-computing perturber positions for each tracer particle",
        )

        (
            massive_Valid_Steps,
            massive_Planet_Xs,
            massive_Planet_Vs,
            massive_Planet_As,
            massive_Asteroid_Xs,
            massive_Planet_Xs_Warmup,
            massive_Asteroid_Xs_Warmup,
            massive_Dts_Warmup,
            massive_Init_Times,
            massive_Jump_Times,
            massive_RAs,
            massive_Decs,
            massive_Astrometric_Uncertainties,
            massive_Observer_Positions,
            massive_Observed_Planet_Xs,
            massive_Observed_Asteroid_Xs,
        ) = _inner_prep_system_GJ_integrator(
            self._ordered_massive_obs,
            "Pre-computing perturber positions for each massive particle",
        )

        if self._parallelize:
            num_devices = jax.local_device_count()
            # can split the pre-computed data into pmappable chunks
            # do this for everything but astrometric uncertainties- for now we're not
            # going to parallelize that last conversion from residuals to likelihood
            num_tracer_particles = tracer_Valid_Steps.shape[0]
            # if there are more tracer particles than available devices, split them up
            # the first tuple entry should be (shape of excess particles, 1, ...)
            # the second should be (num_devices, ...)
            # that way all excess particles can be pmapped at once,
            # then all the others a chunked efficiently
            if num_tracer_particles > num_devices:
                ex = num_tracer_particles % num_devices

                def r(arr):
                    s = arr.shape
                    if len(s) > 1:
                        return (
                            arr[:ex][:, None, :],
                            jnp.array(jnp.split(arr[ex:], num_devices)),
                        )
                    else:
                        return (
                            arr[:ex][:, None],
                            jnp.array(jnp.split(arr[ex:], num_devices)),
                        )

            else:

                def r(arr):
                    s = arr.shape
                    if len(s) > 1:
                        return (
                            arr.reshape([list(s)[0]] + [1] + list(s)[1:]),
                            jnp.array([]),
                        )
                    else:
                        return (arr.reshape([list(s)[0]] + [1]), jnp.array([]))

            tracer_Valid_Steps = r(tracer_Valid_Steps)
            tracer_Planet_Xs = r(tracer_Planet_Xs)
            tracer_Planet_Vs = r(tracer_Planet_Vs)
            tracer_Planet_As = r(tracer_Planet_As)
            tracer_Asteroid_Xs = r(tracer_Asteroid_Xs)
            tracer_Planet_Xs_Warmup = r(tracer_Planet_Xs_Warmup)
            tracer_Asteroid_Xs_Warmup = r(tracer_Asteroid_Xs_Warmup)
            tracer_Dts_Warmup = r(tracer_Dts_Warmup)
            tracer_Init_Times = r(tracer_Init_Times)
            tracer_Jump_Times = r(tracer_Jump_Times)
            tracer_RAs = r(tracer_RAs)
            tracer_Decs = r(tracer_Decs)
            tracer_Observer_Positions = r(tracer_Observer_Positions)
            tracer_Observed_Planet_Xs = r(tracer_Observed_Planet_Xs)
            tracer_Observed_Asteroid_Xs = r(tracer_Observed_Asteroid_Xs)
            tracer_Astrometric_Uncertainties = r(tracer_Astrometric_Uncertainties)

            num_massive_particles = massive_Valid_Steps.shape[0]
            if num_massive_particles > num_devices:
                ex = num_massive_particles % num_devices
                inds = jnp.arange(num_massive_particles)
                massive_pmap_inds = (
                    inds[:ex][:, None],
                    jnp.array(jnp.split(inds[ex:], num_devices)),
                )
            else:
                massive_pmap_inds = (
                    jnp.arange(num_massive_particles)[:, None],
                    jnp.array([]),
                )
        else:
            massive_pmap_inds = jnp.array([])

        padded_supporting_data = {
            "tracer_Valid_Steps": tracer_Valid_Steps,
            "tracer_Planet_Xs": tracer_Planet_Xs,
            "tracer_Planet_Vs": tracer_Planet_Vs,
            "tracer_Planet_As": tracer_Planet_As,
            "tracer_Asteroid_Xs": tracer_Asteroid_Xs,
            "tracer_Planet_Xs_Warmup": tracer_Planet_Xs_Warmup,
            "tracer_Asteroid_Xs_Warmup": tracer_Asteroid_Xs_Warmup,
            "tracer_Dts_Warmup": tracer_Dts_Warmup,
            "tracer_Init_Times": tracer_Init_Times,
            "tracer_Jump_Times": tracer_Jump_Times,
            "tracer_RAs": tracer_RAs,
            "tracer_Decs": tracer_Decs,
            "tracer_Astrometric_Uncertainties": tracer_Astrometric_Uncertainties,
            "tracer_Observer_Positions": tracer_Observer_Positions,
            "tracer_Observed_Planet_Xs": tracer_Observed_Planet_Xs,
            "tracer_Observed_Asteroid_Xs": tracer_Observed_Asteroid_Xs,
            "massive_Valid_Steps": massive_Valid_Steps,
            "massive_Planet_Xs": massive_Planet_Xs,
            "massive_Planet_Vs": massive_Planet_Vs,
            "massive_Planet_As": massive_Planet_As,
            "massive_Asteroid_Xs": massive_Asteroid_Xs,
            "massive_Planet_Xs_Warmup": massive_Planet_Xs_Warmup,
            "massive_Asteroid_Xs_Warmup": massive_Asteroid_Xs_Warmup,
            "massive_Dts_Warmup": massive_Dts_Warmup,
            "massive_Init_Times": massive_Init_Times,
            "massive_Jump_Times": massive_Jump_Times,
            "massive_RAs": massive_RAs,
            "massive_Decs": massive_Decs,
            "massive_Astrometric_Uncertainties": massive_Astrometric_Uncertainties,
            "massive_Observer_Positions": massive_Observer_Positions,
            "massive_Observed_Planet_Xs": massive_Observed_Planet_Xs,
            "massive_Observed_Asteroid_Xs": massive_Observed_Asteroid_Xs,
            "jax_local_device_count": jnp.arange(
                jax.local_device_count()
            ),  # make it arange to encode in shape, not value
            "massive_pmap_inds": massive_pmap_inds,
            "ajk": coeffs_dict["a_jk"][str(self._integrator_order)],
            "bjk": coeffs_dict["b_jk"][str(self._integrator_order)],
            "yc": coeffs_dict["yc"][str(self._warmup_order)],
            "yd": coeffs_dict["yd"][str(self._warmup_order)],
        }

        return padded_supporting_data

    def _generate_single_device_likelihood_funcs(self):
        def _resids(
            tracer_fx_rm__x,
            tracer_fx_rm__v,
            tracer_rx_rm__x,
            tracer_rx_rm__v,
            massive_fx_rm__x,
            massive_fx_rm__v,
            massive_fx_rm__gm,
            massive_rx_fm__x,
            massive_rx_fm__v,
            massive_rx_fm__gm,
            massive_fx_fm__x,
            massive_fx_fm__v,
            massive_fx_fm__gm,
            massive_rx_rm__x,
            massive_rx_rm__v,
            massive_rx_rm__gm,
            tracer_Valid_Steps,
            tracer_Planet_Xs,
            tracer_Planet_Vs,
            tracer_Planet_As,
            tracer_Asteroid_Xs,
            tracer_Planet_Xs_Warmup,
            tracer_Asteroid_Xs_Warmup,
            tracer_Dts_Warmup,
            tracer_Init_Times,
            tracer_Jump_Times,
            tracer_RAs,
            tracer_Decs,
            tracer_Astrometric_Uncertainties,
            tracer_Observer_Positions,
            tracer_Observed_Planet_Xs,
            tracer_Observed_Asteroid_Xs,
            massive_Valid_Steps,
            massive_Planet_Xs,
            massive_Planet_Vs,
            massive_Planet_As,
            massive_Asteroid_Xs,
            massive_Planet_Xs_Warmup,
            massive_Asteroid_Xs_Warmup,
            massive_Dts_Warmup,
            massive_Init_Times,
            massive_Jump_Times,
            massive_RAs,
            massive_Decs,
            massive_Astrometric_Uncertainties,
            massive_Observer_Positions,
            massive_Observed_Planet_Xs,
            massive_Observed_Asteroid_Xs,
            planet_gms,
            asteroid_gms,
            jax_local_device_count,
            massive_pmap_inds,
            ajk,
            bjk,
            yc,
            yd,
        ):
            # setup, merge everything together
            # called out in a separate function so we can jit it- can no longer jit
            # all of _resids because of the pmaps, so we'll just pull some bits out and
            # jit them separately
            tracer_x0s = jnp.concatenate((tracer_fx_rm__x, tracer_rx_rm__x))
            tracer_v0s = jnp.concatenate((tracer_fx_rm__v, tracer_rx_rm__v))
            massive_x0s = jnp.concatenate(
                (massive_fx_rm__x, massive_rx_fm__x, massive_fx_fm__x, massive_rx_rm__x)
            )
            massive_v0s = jnp.concatenate(
                (massive_fx_rm__v, massive_rx_fm__v, massive_fx_fm__v, massive_rx_rm__v)
            )
            massive_gms = jnp.concatenate(
                (
                    massive_fx_rm__gm,
                    massive_rx_fm__gm,
                    massive_fx_fm__gm,
                    massive_rx_rm__gm,
                )
            )

            if tracer_x0s.shape[0] > 0:
                tracer_ras, tracer_decs, tracer_resids, tracer_loglike = (
                    tracer_likelihoods(
                        tracer_x0s,
                        tracer_v0s,
                        tracer_Init_Times,
                        tracer_Jump_Times,
                        tracer_Valid_Steps,
                        tracer_Planet_Xs,
                        tracer_Planet_Vs,
                        tracer_Planet_As,
                        tracer_Asteroid_Xs,
                        tracer_Planet_Xs_Warmup,
                        tracer_Asteroid_Xs_Warmup,
                        tracer_Dts_Warmup,
                        tracer_Observer_Positions,
                        tracer_RAs,
                        tracer_Decs,
                        tracer_Observed_Planet_Xs,
                        tracer_Observed_Asteroid_Xs,
                        tracer_Astrometric_Uncertainties,
                        massive_x0s,
                        massive_v0s,
                        massive_gms,
                        planet_gms,
                        asteroid_gms,
                        bjk,
                        ajk,
                        yc,
                        yd,
                    )
                )
            else:
                tracer_ras = jnp.empty((0, 0))
                tracer_decs = jnp.empty((0, 0))
                tracer_resids = jnp.empty((0, 0))
                tracer_loglike = jnp.array(0.0)

            if massive_x0s.shape[0] > 0:
                scan_inds = jnp.arange(massive_x0s.shape[0])
                massive_ras, massive_decs, massive_resids, massive_loglike = (
                    massive_likelihoods(
                        scan_inds,
                        massive_x0s,
                        massive_v0s,
                        massive_gms,
                        massive_Valid_Steps,
                        massive_Init_Times,
                        massive_Jump_Times,
                        massive_Planet_Xs,
                        massive_Planet_Vs,
                        massive_Planet_As,
                        massive_Asteroid_Xs,
                        massive_Planet_Xs_Warmup,
                        massive_Asteroid_Xs_Warmup,
                        massive_Dts_Warmup,
                        massive_Observer_Positions,
                        massive_Observed_Planet_Xs,
                        massive_Observed_Asteroid_Xs,
                        massive_RAs,
                        massive_Decs,
                        massive_Astrometric_Uncertainties,
                        bjk,
                        ajk,
                        yc,
                        yd,
                        planet_gms,
                        asteroid_gms,
                    )
                )
            else:
                massive_ras = jnp.empty((0, 0))
                massive_decs = jnp.empty((0, 0))
                massive_resids = jnp.empty((0, 0))
                massive_loglike = jnp.array(0.0)

            return (
                (tracer_ras, massive_ras),
                (tracer_decs, massive_decs),
                (tracer_resids, massive_resids),
                tracer_loglike + massive_loglike,
            )

        # Collect all of that into Partial functions with all of the constants baked in
        def resids_function(free_params, fixed_params, padded_supporting_data):
            tmp = _resids(**free_params, **fixed_params, **padded_supporting_data)
            return tmp[0], tmp[1], tmp[2]

        resids_function = jax.jit(
            jax.tree_util.Partial(
                resids_function,
                fixed_params=self._fixed_params,
                padded_supporting_data=self._padded_supporting_data,
            )
        )

        def loglike_function(free_params, fixed_params, padded_supporting_data):
            tmp = _resids(**free_params, **fixed_params, **padded_supporting_data)
            return tmp[3]

        loglike_function = jax.jit(
            jax.tree_util.Partial(
                loglike_function,
                fixed_params=self._fixed_params,
                padded_supporting_data=self._padded_supporting_data,
            )
        )

        return resids_function, loglike_function

    def _generate_parallelized_likelihood_funcs(self):
        def _resids(
            tracer_fx_rm__x,
            tracer_fx_rm__v,
            tracer_rx_rm__x,
            tracer_rx_rm__v,
            massive_fx_rm__x,
            massive_fx_rm__v,
            massive_fx_rm__gm,
            massive_rx_fm__x,
            massive_rx_fm__v,
            massive_rx_fm__gm,
            massive_fx_fm__x,
            massive_fx_fm__v,
            massive_fx_fm__gm,
            massive_rx_rm__x,
            massive_rx_rm__v,
            massive_rx_rm__gm,
            tracer_Valid_Steps,
            tracer_Planet_Xs,
            tracer_Planet_Vs,
            tracer_Planet_As,
            tracer_Asteroid_Xs,
            tracer_Planet_Xs_Warmup,
            tracer_Asteroid_Xs_Warmup,
            tracer_Dts_Warmup,
            tracer_Init_Times,
            tracer_Jump_Times,
            tracer_RAs,
            tracer_Decs,
            tracer_Astrometric_Uncertainties,
            tracer_Observer_Positions,
            tracer_Observed_Planet_Xs,
            tracer_Observed_Asteroid_Xs,
            massive_Valid_Steps,
            massive_Planet_Xs,
            massive_Planet_Vs,
            massive_Planet_As,
            massive_Asteroid_Xs,
            massive_Planet_Xs_Warmup,
            massive_Asteroid_Xs_Warmup,
            massive_Dts_Warmup,
            massive_Init_Times,
            massive_Jump_Times,
            massive_RAs,
            massive_Decs,
            massive_Astrometric_Uncertainties,
            massive_Observer_Positions,
            massive_Observed_Planet_Xs,
            massive_Observed_Asteroid_Xs,
            planet_gms,
            asteroid_gms,
            jax_local_device_count,
            massive_pmap_inds,
            ajk,
            bjk,
            yc,
            yd,
        ):
            tracer_x0s = jnp.concatenate((tracer_fx_rm__x, tracer_rx_rm__x))
            tracer_v0s = jnp.concatenate((tracer_fx_rm__v, tracer_rx_rm__v))
            massive_x0s = jnp.concatenate(
                (massive_fx_rm__x, massive_rx_fm__x, massive_fx_fm__x, massive_rx_rm__x)
            )
            massive_v0s = jnp.concatenate(
                (massive_fx_rm__v, massive_rx_fm__v, massive_fx_fm__v, massive_rx_rm__v)
            )
            massive_gms = jnp.concatenate(
                (
                    massive_fx_rm__gm,
                    massive_rx_fm__gm,
                    massive_fx_fm__gm,
                    massive_rx_rm__gm,
                )
            )

            # tracer particles
            fixed_tracer_inputs = (
                massive_x0s,
                massive_v0s,
                massive_gms,
                planet_gms,
                asteroid_gms,
                bjk,
                ajk,
                yc,
                yd,
            )
            num_tracer_particles = tracer_x0s.shape[0]
            num_devices = jax_local_device_count.shape[0]
            ex = num_tracer_particles % num_devices
            axes = tuple([0] * 18 + [None] * 9)
            if tracer_Init_Times[0].shape[0] > 0:
                (
                    overhanging_ras,
                    overhanging_decs,
                    overhanging_resids,
                    overhanging_loglike,
                ) = jax.pmap(tracer_likelihoods, in_axes=axes)(
                    tracer_x0s[:ex][:, None, :],
                    tracer_v0s[:ex][:, None, :],
                    tracer_Init_Times[0],
                    tracer_Jump_Times[0],
                    tracer_Valid_Steps[0],
                    tracer_Planet_Xs[0],
                    tracer_Planet_Vs[0],
                    tracer_Planet_As[0],
                    tracer_Asteroid_Xs[0],
                    tracer_Planet_Xs_Warmup[0],
                    tracer_Asteroid_Xs_Warmup[0],
                    tracer_Dts_Warmup[0],
                    tracer_Observer_Positions[0],
                    tracer_RAs[0],
                    tracer_Decs[0],
                    tracer_Observed_Planet_Xs[0],
                    tracer_Observed_Asteroid_Xs[0],
                    tracer_Astrometric_Uncertainties[0],
                    *fixed_tracer_inputs,
                )
                overhanging_ras = overhanging_ras.reshape(
                    (-1, overhanging_ras.shape[-1])
                )
                overhanging_decs = overhanging_decs.reshape(
                    (-1, overhanging_decs.shape[-1])
                )
                overhanging_resids = overhanging_resids.reshape(
                    (-1, overhanging_resids.shape[-1])
                )
                overhanging_loglike = jnp.sum(overhanging_loglike)
            else:
                s = tracer_Valid_Steps[1].shape[-1] + 1
                overhanging_ras = jnp.empty((0, s))
                overhanging_decs = jnp.empty((0, s))
                overhanging_resids = jnp.empty((0, s))
                overhanging_loglike = jnp.array(0.0)
            if tracer_Init_Times[1].shape[0] > 1:
                even_ras, even_decs, even_resids, even_loglike = jax.pmap(
                    tracer_likelihoods, in_axes=axes
                )(
                    tracer_x0s[ex:].reshape([num_devices, -1, tracer_x0s.shape[-1]]),
                    tracer_v0s[ex:].reshape([num_devices, -1, tracer_v0s.shape[-1]]),
                    tracer_Init_Times[1],
                    tracer_Jump_Times[1],
                    tracer_Valid_Steps[1],
                    tracer_Planet_Xs[1],
                    tracer_Planet_Vs[1],
                    tracer_Planet_As[1],
                    tracer_Asteroid_Xs[1],
                    tracer_Planet_Xs_Warmup[1],
                    tracer_Asteroid_Xs_Warmup[1],
                    tracer_Dts_Warmup[1],
                    tracer_Observer_Positions[1],
                    tracer_RAs[1],
                    tracer_Decs[1],
                    tracer_Observed_Planet_Xs[1],
                    tracer_Observed_Asteroid_Xs[1],
                    tracer_Astrometric_Uncertainties[1],
                    *fixed_tracer_inputs,
                )
                even_ras = even_ras.reshape((-1, even_ras.shape[-1]))
                even_decs = even_decs.reshape((-1, even_decs.shape[-1]))
                even_resids = even_resids.reshape((-1, even_resids.shape[-1]))
                even_loglike = jnp.sum(even_loglike)
            else:
                s = tracer_Valid_Steps[0].shape[-1] + 1
                even_ras = jnp.empty((0, s))
                even_decs = jnp.empty((0, s))
                even_resids = jnp.empty((0, s))
                even_loglike = jnp.array(0.0)

            tracer_ras = jnp.concatenate((overhanging_ras, even_ras))
            tracer_decs = jnp.concatenate((overhanging_decs, even_decs))
            tracer_resids = jnp.concatenate((overhanging_resids, even_resids))
            tracer_loglike = overhanging_loglike + even_loglike

            # massive particles
            fixed_massive_inputs = (
                massive_x0s,
                massive_v0s,
                massive_gms,
                massive_Valid_Steps,
                massive_Init_Times,
                massive_Jump_Times,
                massive_Planet_Xs,
                massive_Planet_Vs,
                massive_Planet_As,
                massive_Asteroid_Xs,
                massive_Planet_Xs_Warmup,
                massive_Asteroid_Xs_Warmup,
                massive_Dts_Warmup,
                massive_Observer_Positions,
                massive_Observed_Planet_Xs,
                massive_Observed_Asteroid_Xs,
                massive_RAs,
                massive_Decs,
                massive_Astrometric_Uncertainties,
                bjk,
                ajk,
                yc,
                yd,
                planet_gms,
                asteroid_gms,
            )
            axes = tuple([0] + [None] * 25)
            if massive_pmap_inds[0].shape[0] > 0:
                (
                    overhanging_ras,
                    overhanging_decs,
                    overhanging_resids,
                    overhanging_loglike,
                ) = jax.pmap(massive_likelihoods, in_axes=axes)(
                    massive_pmap_inds[0], *fixed_massive_inputs
                )
                overhanging_ras = overhanging_ras.reshape(
                    (-1, overhanging_ras.shape[-1])
                )
                overhanging_decs = overhanging_decs.reshape(
                    (-1, overhanging_decs.shape[-1])
                )
                overhanging_resids = overhanging_resids.reshape(
                    (-1, overhanging_resids.shape[-1])
                )
                overhanging_loglike = jnp.sum(overhanging_loglike)
            else:
                s = massive_Valid_Steps.shape[1] + 1
                (
                    overhanging_ras,
                    overhanging_decs,
                    overhanging_resids,
                    overhanging_loglike,
                ) = (
                    jnp.empty((0, s)),
                    jnp.empty((0, s)),
                    jnp.empty((0, s)),
                    jnp.array(0.0),
                )
            if massive_pmap_inds[1].shape[0] > 0:
                even_ras, even_decs, even_resids, even_loglike = jax.pmap(
                    massive_likelihoods, in_axes=axes
                )(massive_pmap_inds[1], *fixed_massive_inputs)
                even_ras = even_ras.reshape((-1, even_ras.shape[-1]))
                even_decs = even_decs.reshape((-1, even_decs.shape[-1]))
                even_resids = even_resids.reshape((-1, even_resids.shape[-1]))
                even_loglike = jnp.sum(even_loglike)
            else:
                s = massive_Valid_Steps.shape[1] + 1
                even_ras, even_decs, even_resids, even_loglike = (
                    jnp.empty((0, s)),
                    jnp.empty((0, s)),
                    jnp.empty((0, s)),
                    jnp.array(0.0),
                )

            massive_ras = jnp.concatenate((overhanging_ras, even_ras))
            massive_decs = jnp.concatenate((overhanging_decs, even_decs))
            massive_resids = jnp.concatenate((overhanging_resids, even_resids))
            massive_loglike = overhanging_loglike + even_loglike

            return (
                (tracer_ras, massive_ras),
                (tracer_decs, massive_decs),
                (tracer_resids, massive_resids),
                tracer_loglike + massive_loglike,
            )

        # Collect all of that into Partial functions with all of the constants baked in
        # unlike the single device version, cannot jit these- it doesn't play well with
        # pmap: https://github.com/google/jax/issues/2926
        def resids_function(free_params, fixed_params, padded_supporting_data):
            tmp = _resids(**free_params, **fixed_params, **padded_supporting_data)
            return tmp[0], tmp[1], tmp[2]

        resids_function = jax.tree_util.Partial(
            resids_function,
            fixed_params=self._fixed_params,
            padded_supporting_data=self._padded_supporting_data,
        )

        def loglike_function(free_params, fixed_params, padded_supporting_data):
            tmp = _resids(**free_params, **fixed_params, **padded_supporting_data)
            return tmp[3]

        loglike_function = jax.tree_util.Partial(
            loglike_function,
            fixed_params=self._fixed_params,
            padded_supporting_data=self._padded_supporting_data,
        )

        return resids_function, loglike_function

    def _collapse_dicts(self, free_params):
        pass

    ####################################################################################
    # User methods
    ####################################################################################

    def maximimze_likelihood(
        self, threshold=3 * u.arcsec, max_attempts=5, verbose=True, method="BFGS"
    ):
        threshold = threshold.to(u.arcsec).value
        pass

    def propagate(
        self,
        times,
        target_step_size=3.0,
        use_GR=True,
        sky_positions=False,
        observatory_locations=[],
    ):
        """
        Integrate the system to one or more user-specified times.

        Parameters:
            times (astropy.time.Time, or list of astropy.time.Time, or array-like)
                The times to propagate the system to. If given as a float or array of
                floats, they are interpreted as JD TDB times
            target_step_size (float, default=3.0):
                The target step size for the integrator in days. The integrator will
                never take steps larger than this, but will often take smaller steps
            use_GR (bool, default=True):
                Whether to correct for GR effects during the integration using the
                Parameterized Post-Newtonian [1]_ framework
            sky_positions (bool, default=False):
                Whether to return the sky positions of the particles in addition to
                their cartesian positions
            observatory_locations (str, or list of str with len = len(times), or jnp.ndarray(shape=(len(times), 3). default=[]):
                The observatories to use for computing the sky positions. The strings
                can be names of observatories or MPC observatory codes. If given as an
                array, components are understood to be in (x, y, z) cartesian
                coordinates of the observatory in AU. If given as a single string, the
                same observatory is used for all times

        Returns:
            Tuple[jnp.ndarray(shape=len(times), 3), jnp.ndarray(shape=len(times), 3)] or Tuple[jnp.ndarray(shape=len(times), 3), jnp.ndarray(shape=len(times), 3), astropy.coordinates.SkyCoord]:
            xs (jnp.ndarray(shape=(N, len(times), 3)) or jnp.ndarray(shape=(len(times), 3))):
                The cartesian positions of the N particles at the requested times. Units
                are AU. If the system has only one particle, the first dimension is
                dropped
            vs (jnp.ndarray(shape=(N, len(times), 3)) or jnp.ndarray(shape=(len(times), 3))):
                The cartesian velocities of the N particles at the requested times.
                Units are AU / day. If the system has only one particle, the first
                dimension is dropped
            sky_positions (astropy.coordinates.SkyCoord):
                The sky positions of the particles at the requested times. Only returned
                if sky_positions=True

        Examples:

            >>> import jorbit
            >>> from jorbit import Observations, Particle, System
            >>> import jax.numpy as jnp
            >>> import pandas as pd
            >>> import astropy.units as u
            >>> from astropy.coordinates import SkyCoord
            >>> from astropy.time import Time
            >>> d = pd.read_csv(jorbit.DATADIR + "(274301)_wikipedia_horizons_timeseries.txt")
            >>> times = jnp.array(d["time"])
            >>> o = SkyCoord(d["ra"], d["dec"], unit=u.rad)
            >>> all_obs = Observations(
            ...     observed_coordinates=o,
            ...     times=times,
            ...     observatory_locations="kitt peak",
            ...     astrometric_uncertainties=0.1 * u.arcsec,
            ...     verbose_downloading=False,
            ... )
            >>> Xs = jnp.array(d[["x", "y", "z"]])
            >>> Vs = jnp.array(d[["vx", "vy", "vz"]])
            >>> subset_obs = Observations(
            ...     observed_coordinates=o[100:110],
            ...     times=times[100:110],
            ...     observatory_locations="kitt peak",
            ...     astrometric_uncertainties=0.1 * u.arcsec,
            ...     verbose_downloading=True,
            ... )
            >>> asteroid = Particle(
            ...     x=Xs[100],
            ...     v=Vs[100],
            ...     elements=None,
            ...     gm=0,
            ...     time=all_obs.times[100],
            ...     observations=subset_obs,
            ...     earliest_time=Time("1980-01-01"),
            ...     latest_time=Time("2030-01-01"),
            ...     name="(274301) Wikipedia",
            ...     free_orbit=True,
            ...     free_gm=True,
            ... )
            >>> system = System([asteroid])
            >>> print(f"Length of Integration: {all_obs.times[200] - system.time} days")
            >>> x, v, coord = system.propagate(
            ...     all_obs.times[200], sky_positions=True, observatory_locations="Kitt Peak"
            ... )
            >>> print(f"Astrometric error: {coord.separation(o[200]).to(u.mas)}")
            >>> print(f"Cartesian error: {jnp.linalg.norm(x - Xs[200])*u.au.to(u.m)} m")

        """
        # get "times" into an array of (n_times,)
        if isinstance(times, type(Time("2023-01-01"))):
            times = jnp.array(times.tdb.jd)
        elif isinstance(times, list):
            times = jnp.array([t.tdb.jd for t in times])
        if times.shape == ():
            times = jnp.array([times])

        assert jnp.max(times) < self._latest_time, (
            "Requested propagation includes times beyond the latest time in considered"
            " in the ephemeris for this particle. Consider initially setting a broader"
            " time range for the ephemeris."
        )
        assert jnp.min(times) > self._earliest_time, (
            "Requested propagation includes times before the earliest time in"
            " considered in the ephemeris for this particle. Consider initially setting"
            " a broader time range for the ephemeris."
        )

        assert (
            jnp.abs(jnp.sum(jnp.sign(jnp.diff(times)))) == len(times) - 1
        ), "Requested propagation times must be monotonically increasing or decreasing"

        if sky_positions and observatory_locations == []:
            raise ValueError(
                "Must provide observatory locations if sky_positions=True. See"
                " Observations docstring for more info."
            )

        tmp = jnp.sort(jnp.concatenate((jnp.array([self._time]), times)))
        largest_jump = jnp.max(jnp.abs(jnp.diff(tmp)))
        jumps = int(
            max(
                jnp.ceil(largest_jump / target_step_size).astype(int),
                self._integrator_order + 2,
            )
        )  # can't let this be a Jax traced type, won't play nicely with prep function

        (
            Planet_xs,
            Planet_vs,
            Planet_as,
            Asteroid_xs,
            Planet_xs_warmup,
            Asteroid_xs_warmup,
            Dts_warmup,
        ) = prep_gj_integrator_multiple(
            t0=self._time,
            times=times,
            jumps=jumps,
            a_jk=coeffs_dict["a_jk"][str(self._integrator_order)],
            planet_params=self._planet_params,
            asteroid_params=self._asteroid_params,
        )

        xs, vs = gj_integrate_multiple(
            x0=self._xs,
            v0=self._vs,
            gms=self._gms,
            b_jk=coeffs_dict["b_jk"][str(self._integrator_order)],
            a_jk=coeffs_dict["a_jk"][str(self._integrator_order)],
            t0=self._time,
            times=times,
            valid_steps=jnp.array([Planet_xs.shape[2]] * Planet_xs.shape[0]),
            planet_xs=Planet_xs,
            planet_vs=Planet_vs,
            planet_as=Planet_as,
            asteroid_xs=Asteroid_xs,
            planet_xs_warmup=Planet_xs_warmup,
            asteroid_xs_warmup=Asteroid_xs_warmup,
            dts_warmup=Dts_warmup,
            warmup_C=coeffs_dict["yc"][str(self._warmup_order)],
            warmup_D=coeffs_dict["yd"][str(self._warmup_order)],
            planet_gms=self._planet_gms,
            asteroid_gms=self._asteroid_gms,
            use_GR=use_GR,
        )

        # move them
        self._xs = xs[:, -1, :]
        self._vs = vs[:, -1, :]
        self._time = times[-1]

        if sky_positions:
            if observatory_locations == []:
                raise ValueError(
                    "Must provide observatory locations if sky_positions=True. See"
                    " Observations docstring for more info."
                )
            pos = [SkyCoord(0 * u.deg, 0 * u.deg)] * len(times)
            obs = Observations(
                observed_coordinates=pos,
                times=times,
                observatory_locations=observatory_locations,
                astrometric_uncertainties=1 * u.arcsec,
                verbose_downloading=False,
                mpc_file=None,
            )

            observed_planet_xs, _, _ = planet_state(
                planet_params=self._planet_params,
                times=jnp.tile(times, xs.shape[0]),
                velocity=False,
                acceleration=False,
            )
            observed_asteroid_xs, _, _ = planet_state(
                planet_params=self._asteroid_params,
                times=jnp.tile(times, xs.shape[0]),
                velocity=False,
                acceleration=False,
            )

            ras, decs = on_sky(
                xs=xs.reshape((-1, 3)),
                vs=vs.reshape((-1, 3)),
                gms=jnp.zeros(xs.shape[0] * xs.shape[1]),
                observer_positions=obs.observer_positions,
                planet_xs=observed_planet_xs,
                asteroid_xs=observed_asteroid_xs,
                planet_gms=self._planet_gms,
                asteroid_gms=self._asteroid_gms,
            )

            s = SkyCoord(ra=ras * u.rad, dec=decs * u.rad)
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
        raise AttributeError(
            "cannot change xs directly- use propagate(), which will update the entire"
            " state of the system"
        ) from None

    @property
    def vs(self):
        if self._vs.shape[0] == 1:
            return self._vs[0]
        return self._vs

    @vs.setter
    def vs(self, value):
        raise AttributeError(
            "cannot change vs directly- use propagate(), which will update the entire"
            " state of the system"
        ) from None

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
    def particles(self):
        particles = []
        for i, p in enumerate(self._particles):
            particles.append(
                Particle(
                    x=self._xs[i],
                    v=self._vs[i],
                    elements=None,
                    gm=self._gms[i],
                    time=float(self._time),
                    observations=p.observations,
                    earliest_time=p.earliest_time,
                    latest_time=p.latest_time,
                    name=p.name,
                    free_orbit=p.free_orbit,
                    free_gm=p.free_gm,
                )
            )
        return [particles[i] for i in self._particle_indecies]

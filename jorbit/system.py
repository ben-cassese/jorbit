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
from tqdm import tqdm

from jorbit import Observations, Particle
from jorbit.data.constants import ALL_PLANETS, LARGE_ASTEROIDS
from jorbit.engine.utils import construct_perturbers
from jorbit.data import (
    STANDARD_PLANET_PARAMS,
    STANDARD_ASTEROID_PARAMS,
    STANDARD_PLANET_GMS,
    STANDARD_ASTEROID_GMS,
)
from jorbit.engine import (
    on_sky,
    sky_error,
    j_on_sky,
)


from jorbit.data.constants import (
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
)
from jorbit.data import (
    STANDARD_PLANET_PARAMS,
    STANDARD_ASTEROID_PARAMS,
    STANDARD_PLANET_GMS,
    STANDARD_ASTEROID_GMS,
)
from jorbit.engine.gauss_jackson_integrator import gj_integrate_multiple
from jorbit.engine.utils import prep_uneven_GJ_integrator
from jorbit.engine.ephemeris import planet_state


class System:
    def __init__(
        self,
        particles,
        planets=ALL_PLANETS,
        asteroids=LARGE_ASTEROIDS,
        integrator_order=8,
        likelihood_timestep=4.0,
        force_common_epoch=True,
        fit_planet_gms=False,
        fit_asteroid_gms=False,
    ):
        self._particles = particles  # need to re-order though, saved to compare times
        self._planets = planets
        self._asteroids = asteroids
        self._fit_planet_gms = fit_planet_gms
        self._fit_asteroid_gms = fit_asteroid_gms
        self._integrator_order = integrator_order
        self._likelihood_timestep = likelihood_timestep

        (
            self._earliest_time,
            self._latest_time,
            self._planet_params,
            self._asteroid_params,
            self._planet_gms,
            self._asteroid_gms,
        ) = self._initialize_planets()

        (
            self._free_params,
            self._fixed_params,
            self._ordered_tracer_obs,
            self._ordered_massive_obs,
            self._particles,
        ) = self._create_free_fixed_params()

        self._padded_supporting_data = self._prep_system_GJ_integrator()

        # self._generate_likelihood_funcs()
        # self._residual_func, self._likelihood_func = self._generate_likelihood_funcs()

        # xs = []
        # vs = []
        # gms = []
        # for p in self._particles:
        #     xs.append(p.x)
        #     vs.append(p.v)
        #     gms.append(p.gm)
        # self._xs = jnp.array(xs)
        # self._vs = jnp.array(vs)
        # self._gms = jnp.array(gms)

    def __repr__(self):
        return f"System with {len(self._xs)} particles"

    def __len__(self):
        return len(self._xs)

    ################################################################################
    # Misc general heler methods
    ################################################################################
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

    def _create_free_fixed_params(self):
        # f for Free, r for Rigid
        tracer_fx_rm = {"x": [], "v": [], "gm": [], "obs": [], "particles": []}
        tracer_rx_rm = {"x": [], "v": [], "gm": [], "obs": [], "particles": []}
        massive_fx_rm = {"x": [], "v": [], "gm": [], "obs": [], "particles": []}
        massive_rx_fm = {"x": [], "v": [], "gm": [], "obs": [], "particles": []}
        massive_fx_fm = {"x": [], "v": [], "gm": [], "obs": [], "particles": []}
        massive_rx_rm = {"x": [], "v": [], "gm": [], "obs": [], "particles": []}

        for p in self._particles:
            if p.free_orbit and p.free_gm:
                massive_fx_fm["x"].append(p.x)
                massive_fx_fm["v"].append(p.v)
                massive_fx_fm["gm"].append(p.gm)
                massive_fx_fm["obs"].append(p.observations)
                massive_fx_fm["particles"].append(p)
            elif p.free_orbit and not p.free_gm:
                if p.gm == 0:
                    tracer_fx_rm["x"].append(p.x)
                    tracer_fx_rm["v"].append(p.v)
                    tracer_fx_rm["gm"].append(p.gm)
                    tracer_fx_rm["obs"].append(p.observations)
                    tracer_fx_rm["particles"].append(p)
                else:
                    massive_fx_rm["x"].append(p.x)
                    massive_fx_rm["v"].append(p.v)
                    massive_fx_rm["gm"].append(p.gm)
                    massive_fx_rm["obs"].append(p.observations)
                    massive_fx_rm["particles"].append(p)
            elif not p.free_orbit and p.free_gm:
                massive_rx_fm["x"].append(p.x)
                massive_rx_fm["v"].append(p.v)
                massive_rx_fm["gm"].append(p.gm)
                massive_rx_fm["obs"].append(p.observations)
                massive_rx_fm["particles"].append(p)
            elif not p.free_orbit and not p.free_gm:
                if p.gm == 0:
                    tracer_rx_rm["x"].append(p.x)
                    tracer_rx_rm["v"].append(p.v)
                    tracer_rx_rm["gm"].append(p.gm)
                    tracer_rx_rm["obs"].append(p.observations)
                    tracer_rx_rm["particles"].append(p)
                else:
                    massive_rx_rm["x"].append(p.x)
                    massive_rx_rm["v"].append(p.v)
                    massive_rx_rm["gm"].append(p.gm)
                    massive_rx_rm["obs"].append(p.observations)
                    massive_rx_rm["particles"].append(p)

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

        return (
            free_params,
            fixed_params,
            ordered_tracer_obs,
            ordered_massive_obs,
            reordered_particles,
        )

    def _prep_system_GJ_integrator(
        self,
    ):
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
        }

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

        # can split the pre-computed data into pmappable chunks
        # do this for everything but astrometric uncertainties- for now we're not
        # going to parallelize that last conversion from residuals to likelihood
        num_devices = jax.local_device_count()
        num_tracer_particles = tracer_Valid_Steps.shape[0]
        # if there are more tracer particles than available devices, split them up
        # the first tuple entry should be (shape of excess particles, 1, ...)
        # the second should be (num_devices, ...)
        # that way all excess particles can be pmapped at once,
        # then all the others a chunked efficiently
        if num_tracer_particles > num_devices:
            ex = num_tracer_particles % num_devices

            def r(arr):
                return (
                    arr[:ex][:, None, :],
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
        }

        return padded_supporting_data

    def _generate_likelihood_funcs(self):
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
        ):
            jax_local_device_count = jax_local_device_count.shape[0]
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

            def pmappable_tracer_contribution(
                ptracer_x0s,
                ptracer_v0s,
                ptracer_Init_Times,
                ptracer_Jump_Times,
                ptracer_Valid_Steps,
                ptracer_Planet_Xs,
                ptracer_Planet_Vs,
                ptracer_Planet_As,
                ptracer_Asteroid_Xs,
                ptracer_Planet_Xs_Warmup,
                ptracer_Asteroid_Xs_Warmup,
                ptracer_Dts_Warmup,
                ptracer_Observer_Positions,
                ptracer_RAs,
                ptracer_Decs,
                ptracer_Observed_Planet_Xs,
                ptracer_Observed_Asteroid_Xs,
            ):
                def _tracer_scan_func(carry, scan_over):
                    (
                        x0,
                        v0,
                        init_time,
                        jump_times,
                        valid_steps,
                        planet_xs,
                        planet_vs,
                        planet_as,
                        asteroid_xs,
                        planet_xs_warmup,
                        asteroid_xs_warmup,
                        dts_warmup,
                        observer_positions,
                        ras,
                        decs,
                        observed_planet_xs,
                        observed_asteroid_xs,
                    ) = scan_over

                    x0 = jnp.concatenate((jnp.array([x0]), massive_x0s))
                    v0 = jnp.concatenate((jnp.array([v0]), massive_v0s))
                    gms = jnp.concatenate((jnp.array([0.0]), massive_gms))

                    x, v = gj_integrate_multiple(
                        x0=x0,
                        v0=v0,
                        gms=gms,
                        valid_steps=valid_steps,
                        b_jk=GJ8_B,
                        a_jk=GJ8_A,
                        t0=init_time,
                        times=jump_times,
                        planet_xs=planet_xs,
                        planet_vs=planet_vs,
                        planet_as=planet_as,
                        asteroid_xs=asteroid_xs,
                        planet_xs_warmup=planet_xs_warmup,
                        asteroid_xs_warmup=asteroid_xs_warmup,
                        dts_warmup=dts_warmup,
                        warmup_C=Y4_C,
                        warmup_D=Y4_D,
                        planet_gms=STANDARD_PLANET_GMS,
                        asteroid_gms=STANDARD_ASTEROID_GMS,
                        use_GR=True,
                    )

                    # we only care about the first particle, the tracer
                    # add the first position back in
                    x = jnp.concatenate((x0[0][None], x[0, :, :]))
                    v = jnp.concatenate((v0[0][None], v[0, :, :]))

                    calc_ra, calc_dec = on_sky(
                        xs=x,
                        vs=v,
                        gms=jnp.zeros(x.shape[0]),
                        observer_positions=observer_positions,
                        planet_xs=observed_planet_xs,
                        asteroid_xs=observed_asteroid_xs,
                        planet_gms=planet_gms,
                        asteroid_gms=asteroid_gms,
                    )

                    resids = sky_error(
                        calc_ra=calc_ra,
                        calc_dec=calc_dec,
                        true_ra=ras,
                        true_dec=decs,
                    )

                    return None, (x, v, calc_ra, calc_dec, resids)

                tmp = jax.lax.scan(
                    _tracer_scan_func,
                    None,
                    (
                        ptracer_x0s,
                        ptracer_v0s,
                        ptracer_Init_Times,
                        ptracer_Jump_Times,
                        ptracer_Valid_Steps,
                        ptracer_Planet_Xs,
                        ptracer_Planet_Vs,
                        ptracer_Planet_As,
                        ptracer_Asteroid_Xs,
                        ptracer_Planet_Xs_Warmup,
                        ptracer_Asteroid_Xs_Warmup,
                        ptracer_Dts_Warmup,
                        ptracer_Observer_Positions,
                        ptracer_RAs,
                        ptracer_Decs,
                        ptracer_Observed_Planet_Xs,
                        ptracer_Observed_Asteroid_Xs,
                    ),
                )[1]
                tracer_xs = tmp[0]
                tracer_vs = tmp[1]
                tracer_ras = tmp[2]
                tracer_decs = tmp[3]
                tracer_resids = tmp[4]
                return tracer_xs, tracer_vs, tracer_ras, tracer_decs, tracer_resids

            def pmappable_massive_contribution(scan_inds):
                def _massive_scan_func(carry, scan_over):
                    ind = scan_over
                    x, v = gj_integrate_multiple(
                        x0=massive_x0s,
                        v0=massive_v0s,
                        gms=massive_gms,
                        valid_steps=massive_Valid_Steps[ind],
                        b_jk=GJ8_B,
                        a_jk=GJ8_A,
                        t0=massive_Init_Times[ind],
                        times=massive_Jump_Times[ind],
                        planet_xs=massive_Planet_Xs[ind],
                        planet_vs=massive_Planet_Vs[ind],
                        planet_as=massive_Planet_As[ind],
                        asteroid_xs=massive_Asteroid_Xs[ind],
                        planet_xs_warmup=massive_Planet_Xs_Warmup[ind],
                        asteroid_xs_warmup=massive_Asteroid_Xs_Warmup[ind],
                        dts_warmup=massive_Dts_Warmup[ind],
                        warmup_C=Y4_C,
                        warmup_D=Y4_D,
                        planet_gms=STANDARD_PLANET_GMS,
                        asteroid_gms=STANDARD_ASTEROID_GMS,
                        use_GR=True,
                    )

                    # add the first position back in
                    x = jnp.concatenate((massive_x0s[ind][None, :], x[ind, :, :]))
                    v = jnp.concatenate((massive_v0s[ind][None, :], v[ind, :, :]))

                    calc_ra, calc_dec = on_sky(
                        xs=x,
                        vs=v,
                        gms=jnp.ones(x.shape[0]) * massive_gms[ind],
                        observer_positions=massive_Observer_Positions[ind],
                        planet_xs=massive_Observed_Planet_Xs[ind],
                        asteroid_xs=massive_Observed_Asteroid_Xs[ind],
                        planet_gms=planet_gms,
                        asteroid_gms=asteroid_gms,
                    )

                    resids = sky_error(
                        calc_ra=calc_ra,
                        calc_dec=calc_dec,
                        true_ra=massive_RAs[ind],
                        true_dec=massive_Decs[ind],
                    )

                    return None, (x, v, calc_ra, calc_dec, resids)

                tmp = jax.lax.scan(
                    _massive_scan_func,
                    None,
                    scan_inds,
                )[1]
                massive_xs = tmp[0]
                massive_vs = tmp[1]
                massive_calc_RAs = tmp[2]
                massive_calc_Decs = tmp[3]
                massive_resids = tmp[4]
                return (
                    massive_xs,
                    massive_vs,
                    massive_calc_RAs,
                    massive_calc_Decs,
                    massive_resids,
                )

            #
            if tracer_x0s.shape[0] > 0:
                # if there are "excess" particles, either b/c there are fewer particles
                # than available devices, or b/c the number of particles is not divisible
                # by the number of devices
                if tracer_Valid_Steps[0].shape[0] > 0:
                    excess_results = jax.pmap(pmappable_tracer_contribution)(
                        tracer_x0s[0],
                        tracer_v0s[0],
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
                    )
                # if there are no excess particles, which means the number of particles
                # cleanly divides the number of devices, don't pmap over these empty
                # arrays
                else:
                    excess_results = (
                        jnp.empty((0, 0)),
                        jnp.empty((0, 0)),
                        jnp.empty((0, 0)),
                        jnp.empty((0, 0)),
                    )
                # if some particles exist in the second tuple entry, meaning the remaining
                # number of particles not covered above is cleanly divisible by the number
                # of devices, do another pmap over these
                if tracer_Valid_Steps[1].shape[0] > 0:
                    larger_mapped_results = jax.pmap(pmappable_tracer_contribution)(
                        tracer_x0s[1],
                        tracer_v0s[1],
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
                    )
                # if there are no particles in the second tuple entry, don't pmap over
                # these empty arrays
                else:
                    larger_mapped_results = (
                        jnp.empty((0, 0)),
                        jnp.empty((0, 0)),
                        jnp.empty((0, 0)),
                        jnp.empty((0, 0)),
                    )
                # combine the results from the two pmaps
                tracer_RAs = jnp.concatenate(
                    (excess_results[0], larger_mapped_results[0])
                )
                tracer_Decs = jnp.concatenate(
                    (excess_results[1], larger_mapped_results[1])
                )
                tracer_resids = jnp.concatenate(
                    (excess_results[2], larger_mapped_results[2])
                )
                tracer_likelihood = jnp.sum(excess_results[3]) + jnp.sum(
                    larger_mapped_results[3]
                )

            if massive_x0s.shape[0] > 0:
                pass
            else:
                massive_RAs = jnp.empty((0, 0))
                massive_Decs = jnp.empty((0, 0))
                massive_resids = jnp.empty((0, 0))
                massive_likelihood = jnp.empty((0, 0))

            return (
                jnp.concatenate((tracer_RAs, massive_RAs)),
                jnp.concatenate((tracer_Decs, massive_Decs)),
                jnp.concatenate((tracer_resids, massive_resids)),
                jnp.concatenate((tracer_likelihood, massive_likelihood)),
            )

            #     # tracer_ra, tracer_dec, tracer_resids
            # if len(massive_x0s) > 0:
            #     num_massive_particles = massive_x0s.shape[0]
            #     ex = num_massive_particles % jax_local_device_count
            #     massive_x0s = (massive_x0s[:ex], jnp.array(jnp.split(massive_x0s[ex:], jax_local_device_count)))
            #     massive_v0s = (massive_v0s[:ex], jnp.array(jnp.split(massive_v0s[ex:], jax_local_device_count)))
            #     excess_results = jax.pmap(massive_contribution)(massive_pmap_inds[0])
            #     pmmapped_results = jax.pmap(massive_contribution)(massive_pmap_inds[1])
            #     # massive_ra, massive_dec, massive_resids

        #     # put together the likelihoods
        #     loglike = 0.0
        #     if len(tracer_x0s) > 0 & len(massive_x0s) > 0:
        #         sigma2 = tracer_Astrometric_Uncertainties**2
        #         loglike += -0.5 * jnp.sum(tracer_resids**2 / sigma2)
        #         sigma2 = massive_Astrometric_Uncertainties**2
        #         loglike += -0.5 * jnp.sum(massive_resids**2 / sigma2)
        #         return (
        #             jnp.concatenate((tracer_ras, massive_ras)),
        #             jnp.concatenate((tracer_decs, massive_decs)),
        #             jnp.concatenate((tracer_resids, massive_resids)),
        #             loglike,
        #         )
        #     elif len(tracer_x0s) > 0:
        #         sigma2 = tracer_Astrometric_Uncertainties**2
        #         loglike += -0.5 * jnp.sum(tracer_resids**2 / sigma2)
        #         return tracer_ra, tracer_dec, tracer_resids, loglike
        #     elif len(massive_x0s) > 0:
        #         sigma2 = massive_Astrometric_Uncertainties**2
        #         loglike += -0.5 * jnp.sum(massive_resids**2 / sigma2)
        #         return massive_ra, massive_dec, massive_resids, loglike

        # # Collect all of that into Partial functions with all of the constants baked in
        # def resids_function(free_params, fixed_params, padded_supporting_data):
        #     tmp = _resids(**free_params, **fixed_params, **padded_supporting_data)
        #     return tmp[0], tmp[1], tmp[2]

        # resids_function = jax.jit(
        #     jax.tree_util.Partial(
        #         resids_function,
        #         fixed_params=self._fixed_params,
        #         padded_supporting_data=self._padded_supporting_data,
        #     )
        # )

        # def loglike_function(free_params, fixed_params, padded_supporting_data):
        #     tmp = _resids(**free_params, **fixed_params, **padded_supporting_data)
        #     return tmp[3]

        # loglike_function = jax.jit(
        #     jax.tree_util.Partial(
        #         loglike_function,
        #         fixed_params=self._fixed_params,
        #         padded_supporting_data=self._padded_supporting_data,
        #     )
        # )

        # return resids_function, loglike_function

    #     # initial stuff we can save/check right away
    #     self._time = particles[0].time
    #     earlys = []
    #     lates = []

    #     self._particle_names = []
    #     self._particle_observations = []
    #     self._particle_free_orbit = []
    #     self._particle_free_gm = []
    #     for i, p in enumerate(particles):
    #         earlys.append(p.earliest_time.tdb.jd)
    #         lates.append(p.latest_time.tdb.jd)
    #         assert (
    #             p.time == self._time
    #         ), "All particles must be initalized to the same time"

    #         if p.name == "":
    #             self._particle_names.append(f"Particle {i}")
    #         else:
    #             self._particle_names.append(p.name)

    #         num_unique_names = len(set(self._particle_names))
    #         assert num_unique_names == i + 1, (
    #             "2 or more identical names detected. If using custom names, make sure"
    #             " they are each unique and not 'Particle (int < # of particles in the"
    #             " system)'"
    #         )

    #         self._particle_observations.append(p.observations)
    #         self._particle_free_orbit.append(p.free_orbit)
    #         self._particle_free_gm.append(p.free_gm)

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
        use_GR=False,
        obey_large_step_limits=True,
        sky_positions=False,
        observatory_locations=[],
    ):
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
        pass

        # move them

    #     self._xs = xs
    #     self._vs = vs
    #     self._time = final_times[-1]

    #     if sky_positions:
    #         if observatory_locations == []:
    #             raise ValueError(
    #                 "Must provide observatory locations if sky_positions=True. See"
    #                 " Observations docstring for more info."
    #             )
    #         pos = [SkyCoord(0 * u.deg, 0 * u.deg)] * len(times)
    #         obs = Observations(
    #             positions=pos,
    #             times=times,
    #             observatory_locations=observatory_locations,
    #             astrometric_uncertainties=[10 * u.mas] * len(times),
    #         )

    #         ras, decs = j_on_sky(
    #             xs=xs.reshape(-1, xs.shape[-1]),
    #             vs=vs.reshape(-1, vs.shape[-1]),
    #             gms=jnp.tile(self._gms, len(times)),
    #             times=jnp.tile(times, len(xs)),
    #             observer_positions=jnp.array(list(obs.observer_positions) * len(xs)),
    #             planet_params=self._planet_params,
    #             asteroid_params=self._asteroid_params,
    #             planet_gms=self._planet_gms,
    #             asteroid_gms=self._asteroid_gms,
    #         )

    #         s = SkyCoord(ra=ras * u.rad, dec=decs * u.rad)
    #         if xs.shape[0] == 1:
    #             return xs[0], vs[0], s
    #         return xs, vs, s.reshape((xs.shape[0], xs.shape[1]))

    #     if xs.shape[0] == 1:
    #         return xs[0], vs[0]
    #     return xs, vs

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

    # @property
    # def particles(self):
    #     particles = []
    #     for i, n in enumerate(self._particle_names):
    #         particles.append(
    #             Particle(
    #                 x=self._xs[i],
    #                 v=self._vs[i],
    #                 elements=None,
    #                 gm=self._gms[i],
    #                 time=float(self._time),
    #                 observations=self._particle_observations[i],
    #                 earliest_time=self._earliest_time,
    #                 latest_time=self._latest_time,
    #                 name=n,
    #                 free_orbit=self._particle_free_orbit[i],
    #                 free_gm=self._particle_free_gm[i],
    #             )
    #         )
    #     return particles

# the processing of the .bsp file partially relies on, then is heavily influenced by,
# the implementation in the jplephem package:
# https://github.com/brandon-rhodes/python-jplephem/blob/master/jplephem/spk.py

import jax

jax.config.update("jax_enable_x64", True)
import warnings

import astropy.units as u
import jax.numpy as jnp
from astropy.time import Time

warnings.filterwarnings("ignore", module="erfa")

from jorbit.data.constants import (
    ALL_PLANET_IDS,
    ALL_PLANET_LOG_GMS,
    ALL_PLANET_NAMES,
    DEFAULT_ASTEROID_EPHEMERIS_URL,
    DEFAULT_PLANET_EPHEMERIS_URL,
    LARGE_ASTEROID_IDS,
    LARGE_ASTEROID_LOG_GMS,
    LARGE_ASTEROID_NAMES,
)
from jorbit.ephemeris.ephemeris_processors import (
    EphemerisPostProcessor,
    EphemerisProcessor,
)
from jorbit.ephemeris.process_bsp import extract_data, merge_data


class Ephemeris:
    def __init__(
        self,
        ssos="default planets",
        earliest_time=Time("1980-01-01"),
        latest_time=Time("2050-01-01"),
        postprocessing_func=None,
    ):

        if ssos == "default planets":
            ssos = [
                {
                    "ephem_file": DEFAULT_PLANET_EPHEMERIS_URL,
                    "names": ALL_PLANET_NAMES,
                    "targets": [ALL_PLANET_IDS[name] for name in ALL_PLANET_NAMES],
                    "centers": [0] * len(ALL_PLANET_IDS),
                    "log_gms": ALL_PLANET_LOG_GMS,
                }
            ]
        elif ssos == "default solar system":
            ssos = [
                {
                    "ephem_file": DEFAULT_PLANET_EPHEMERIS_URL,
                    "names": ALL_PLANET_NAMES,
                    "targets": [ALL_PLANET_IDS[name] for name in ALL_PLANET_NAMES],
                    "centers": [0] * len(ALL_PLANET_IDS),
                    "log_gms": ALL_PLANET_LOG_GMS,
                },
                {
                    "ephem_file": DEFAULT_ASTEROID_EPHEMERIS_URL,
                    "names": LARGE_ASTEROID_NAMES,
                    "targets": [
                        LARGE_ASTEROID_IDS[name] for name in LARGE_ASTEROID_NAMES
                    ],
                    "centers": [10] * len(LARGE_ASTEROID_IDS),
                    "log_gms": LARGE_ASTEROID_LOG_GMS,
                },
            ]

            def postprocessing_func(
                x, v
            ):  # , a): # the asteroids are all relative to the sun, not the barycenter
                x = x.at[-16:].set(x[-16:] + x[0])
                v = v.at[-16:].set(v[-16:] + v[0])
                # a = a.at[-16:].set(0.0)
                return x, v  # , a

            postprocessing_func = jax.tree_util.Partial(postprocessing_func)

        combined_names = []
        combined_log_gms = []
        for sso_group in ssos:
            combined_names += sso_group["names"]
            for n in sso_group["names"]:
                combined_log_gms.append(sso_group["log_gms"][n])
        self._combined_names = combined_names
        self._combined_log_gms = combined_log_gms

        ephs = []
        for sso_group in ssos:
            inits, intlens, coeffs = [], [], []
            for target, center in zip(
                sso_group["targets"], sso_group["centers"], strict=False
            ):
                init, intlen, coeff = extract_data(
                    center, target, sso_group["ephem_file"], earliest_time, latest_time
                )
                inits.append(init)
                intlens.append(intlen)
                coeffs.append(coeff)
            init, intlen, coeff = merge_data(
                inits, intlens, coeffs, earliest_time, latest_time
            )
            gms = []
            for n in sso_group["names"]:
                gms.append(sso_group["log_gms"][n])
            gms = jnp.array(gms)
            ephs.append(EphemerisProcessor(init, intlen, coeff, gms))
        self.ephs = tuple(ephs)

        if len(self.ephs) == 1:
            self.processor = self.ephs[0]
        else:
            self.processor = EphemerisPostProcessor(self.ephs, postprocessing_func)

    def state(self, time: Time):
        x, v = self.processor.state(time.tdb.jd)
        s = {}
        for n in range(len(self._combined_names)):
            s[self._combined_names[n]] = {
                "x": x[n] * u.au,
                "v": v[n] * u.au / u.day,
                # "a": a[n] * u.au / u.day**2,
                "log_gm": self._combined_log_gms[n],
            }
        return s

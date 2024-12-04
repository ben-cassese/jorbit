# the processing of the .bsp file partially relies on, then is heavily influenced by,
# the implementation in the jplephem package:
# https://github.com/brandon-rhodes/python-jplephem/blob/master/jplephem/spk.py

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import astropy.units as u
from astropy.time import Time
from astropy.utils.data import download_file
from jplephem.spk import SPK

import warnings

warnings.filterwarnings("ignore", module="erfa")

from jorbit.ephemeris.functional_ephemeris import FunctionalEphemeris
from jorbit.data.constants import (
    DEFAULT_PLANET_EPHEMERIS_URL,
    DEFAULT_ASTEROID_EPHEMERIS_URL,
    ALL_PLANET_IDS,
    ALL_PLANET_LOG_GMS,
    LARGE_ASTEROID_IDS,
    LARGE_ASTEROID_LOG_GMS,
)


class Ephemeris:
    def __init__(
        self,
        earliest_time=Time("1980-01-01"),
        latest_time=Time("2100-01-01"),
        ssos="default planets",
        ephem_file=DEFAULT_PLANET_EPHEMERIS_URL,
    ):
        self.earliest_time = earliest_time
        self.latest_time = latest_time
        self.ssos = ssos
        self.ephem_file = ephem_file

        self.ssos, self.sso_ids, self.sso_log_gms = self.setup_ssos(ssos)

        self.validation_checks()

        self.FunctionalEphemeris = self.setup_functional()

    def setup_ssos(self, ssos):
        if ssos == "default planets":
            ssos = list(ALL_PLANET_IDS.keys())
            ids = ALL_PLANET_IDS
            log_gms = ALL_PLANET_LOG_GMS
        elif ssos == "large asteroids":
            ssos = list(LARGE_ASTEROID_IDS.keys())
            ids = LARGE_ASTEROID_IDS
            log_gms = LARGE_ASTEROID_LOG_GMS
        else:
            ssos = ssos
            ids = []
            log_gms = []
            for obj in ssos:
                if obj.lower() in ALL_PLANET_IDS.keys():
                    ids.append(ALL_PLANET_IDS[obj.lower()])
                    log_gms.append(ALL_PLANET_LOG_GMS[obj.lower()])
                elif obj.lower() in LARGE_ASTEROID_IDS.keys():
                    ids.append(LARGE_ASTEROID_IDS[obj.lower()])
                    log_gms.append(LARGE_ASTEROID_LOG_GMS[obj.lower()])
                else:
                    raise ValueError(f"{obj} is not a recognized object")
            ids = dict(zip(ssos, ids))
            log_gms = dict(zip(ssos, log_gms))

        return ssos, ids, log_gms

    def validation_checks(self):
        if self.earliest_time > self.latest_time:
            raise ValueError("earliest_time must be before latest_time")

        kernel = SPK.open(download_file(self.ephem_file, cache=True))
        for i, name in enumerate(self.ssos):
            found = False
            for seg in kernel.segments:
                if seg.target == self.sso_ids[name]:
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"Object '{self.ssos[i]}' (id={self.sso_ids[name]}) is not found included in this ephemeris file, {self.ephem_file}"
                )

    def setup_functional(self):
        kernel = SPK.open(download_file(self.ephem_file, cache=True))

        sso_init = []
        sso_intlen = []
        sso_coeffs = []
        for name in self.ssos:
            id = self.sso_ids[name]
            for seg in kernel.segments:
                if seg.target != id:
                    continue
                init, intlen, coeff = seg._data
                sso_init.append(jnp.array(init))
                sso_intlen.append(jnp.array(intlen))
                sso_coeffs.append(jnp.array(coeff))

        sso_init = jnp.array(sso_init)
        sso_intlen = jnp.array(sso_intlen)
        # sso_coeffs = jnp.array(sso_coeffs)

        init0 = sso_init[0]
        for i in sso_init:
            assert i == init0

        # Trim the timespans down to the earliest and latest times
        longest_intlen = jnp.max(sso_intlen)
        ratios = longest_intlen / sso_intlen
        early_indecies = []
        late_indecies = []
        for i in range(len(sso_init)):
            component_count, coefficient_count, n = sso_coeffs[i].shape
            index, offset = jnp.divmod(
                (self.earliest_time.tdb.jd - 2451545.0) * 86400.0 - sso_init[i],
                sso_intlen[i],
            )
            omegas = index == n
            index = jnp.where(omegas, index - 1, index)
            early_indecies.append(index)

            index, offset = jnp.divmod(
                (self.latest_time.tdb.jd - 2451545.0) * 86400.0 - sso_init[i],
                sso_intlen[i],
            )
            omegas = index == n
            index = jnp.where(omegas, index - 1, index)
            late_indecies.append(index)

        early_indecies = (
            jnp.ones(len(early_indecies)) * jnp.min(jnp.array(early_indecies)) * ratios
        ).astype(int)
        new_inits = sso_init + early_indecies * sso_intlen
        late_indecies = (
            jnp.ones(len(late_indecies)) * jnp.min(jnp.array(late_indecies)) * ratios
        ).astype(int)
        trimmed_coeffs = []
        for i in range(len(sso_init)):
            trimmed_coeffs.append(
                sso_coeffs[i][:, :, early_indecies[i] : late_indecies[i]]
            )

        # Add extra Chebyshev coefficients (zeros) to make the number of
        # coefficients at each time slice the same across all planets
        coeff_shapes = []
        for i in trimmed_coeffs:
            coeff_shapes.append(i.shape)
        coeff_shapes = jnp.array(coeff_shapes)
        most_coefficients, _, most_time_slices = jnp.max(coeff_shapes, axis=0)

        padded_coefficients = []
        for c in trimmed_coeffs:
            c = jnp.pad(c, ((most_coefficients - c.shape[0], 0), (0, 0), (0, 0)))
            padded_coefficients.append(c)

        # This is a little sketchy- tile each planet so that they all have the same
        # number of time slices. This means that for planets with longer original intlens,
        # we could technically feed in times outside the original timespan and get a false result
        # But, by keeping their original intlens intact, if we feed in a time within
        # the timespan, we should just always stay in the first half, quarter, whatever
        shortest_intlen = jnp.min(sso_intlen)
        padded_intlens = jnp.ones(len(sso_intlen)) * shortest_intlen
        extra_padded = []
        for i in range(len(padded_coefficients)):
            extra_padded.append(
                jnp.tile(padded_coefficients[i], int(sso_intlen[i] / shortest_intlen))
            )

        new_coeff = jnp.array(extra_padded)

        log_gms = []
        for name in self.ssos:
            log_gms.append(self.sso_log_gms[name])
        log_gms = jnp.array(log_gms)
        return FunctionalEphemeris(new_inits, sso_intlen, new_coeff, log_gms)

    def state(self, time):
        xs, vs, accs = self.FunctionalEphemeris.state(time.tdb.jd)
        d = {}
        for i, id in enumerate(self.sso_ids):
            d[id] = {
                "x": xs[i],
                "v": vs[i],
                "a": accs[i],
            }
        return d

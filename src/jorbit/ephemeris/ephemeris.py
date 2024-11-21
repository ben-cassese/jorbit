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

from jorbit.data.constants import (
    DEFAULT_PLANET_EPHEMERIS_URL,
    DEFAULT_ASTEROID_EPHEMERIS_URL,
)


@jax.tree_util.register_pytree_node_class
class ProcessedEphemeris:
    def __init__(self, init, intlen, coeffs, gms):
        self.init = init
        self.intlen = intlen
        self.coeffs = coeffs
        self.gms = gms

    def tree_flatten(self):
        children = (self.init, self.intlen, self.coeffs, self.gms)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @jax.jit
    def eval_cheby(self, coefficients, x):
        b_ii = jnp.zeros(3)
        b_i = jnp.zeros(3)

        def scan_func(X, a):
            b_i, b_ii = X
            tmp = b_i
            b_i = a + 2 * x * b_i - b_ii
            b_ii = tmp
            return (b_i, b_ii), b_i

        (b_i, b_ii), s = jax.lax.scan(scan_func, (b_i, b_ii), coefficients[:-1])
        return coefficients[-1] + x * b_i - b_ii, s

    @jax.jit
    def _individual_state(self, init, intlen, coeffs, tdb):
        tdb2 = 0.0  # leaving in case we ever decide to increase the time precision and use 2 floats
        _, _, n = coeffs.shape

        # 2451545.0 is the J2000 epoch in TDB
        index1, offset1 = jnp.divmod((tdb - 2451545.0) * 86400.0 - init, intlen)
        index2, offset2 = jnp.divmod(tdb2 * 86400.0, intlen)
        index3, offset = jnp.divmod(offset1 + offset2, intlen)
        index = (index1 + index2 + index3).astype(int)

        omegas = index == n
        index = jnp.where(omegas, index - 1, index)
        offset = jnp.where(omegas, offset + intlen, offset)

        coefficients = coeffs[:, :, index]

        s = 2.0 * offset / intlen - 1.0

        # Position
        x, As = self.eval_cheby(coefficients, s)  # in km here

        # Velocity
        Q = self.eval_cheby(2 * As, s)
        v = Q[0] - As[-1]
        v /= intlen
        v *= 2.0  # in km/s here

        # Acceleration
        a = self.eval_cheby(4 * Q[1], s)[0] - 2 * Q[1][-1]
        a /= intlen**2
        a *= 4.0  # in km/s^2 here

        # Convert to AU, AU/day, AU/day^2
        return (
            x.T * 6.684587122268446e-09,
            v.T * 0.0005775483273639937,
            a.T * 49.900175484249054,
        )

    @jax.jit
    def state(self, tdb):
        x, v, a = jax.vmap(self._individual_state, in_axes=(0, 0, 0, None))(
            self.init, self.intlen, self.coeffs, tdb
        )
        return x, v, a


class Ephemeris:
    def __init__(
        self,
        earliest_time=Time("1980-01-01"),
        latest_time=Time("2100-01-01"),
        ephem_file=DEFAULT_PLANET_EPHEMERIS_URL,
    ):
        self.earliest_time = earliest_time
        self.latest_time = latest_time
        self.ephem_file = ephem_file

        self.planet_ids = [10, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        self.initial_checks()
        self.ProcessedEphemeris = self.process()

    def initial_checks(self):
        pass

    def process(self):
        # come back to this to let users pick planets

        kernel = SPK.open(download_file(self.ephem_file, cache=True))

        planet_init = []
        planet_intlen = []
        planet_coeffs = []
        for id in self.planet_ids:
            for seg in kernel.segments:
                if seg.target != id:
                    continue
                init, intlen, coeff = seg._data
                planet_init.append(jnp.array(init))
                planet_intlen.append(jnp.array(intlen))
                planet_coeffs.append(jnp.array(coeff))

        planet_init = jnp.array(planet_init)
        planet_intlen = jnp.array(planet_intlen)
        # planet_coeffs = jnp.array(planet_coeffs)

        init0 = planet_init[0]
        for i in planet_init:
            assert i == init0

        # Trim the timespans down to the earliest and latest times
        longest_intlen = jnp.max(planet_intlen)
        ratios = longest_intlen / planet_intlen
        early_indecies = []
        late_indecies = []
        for i in range(len(planet_init)):
            component_count, coefficient_count, n = planet_coeffs[i].shape
            index, offset = jnp.divmod(
                (self.earliest_time.tdb.jd - 2451545.0) * 86400.0 - planet_init[i],
                planet_intlen[i],
            )
            omegas = index == n
            index = jnp.where(omegas, index - 1, index)
            early_indecies.append(index)

            index, offset = jnp.divmod(
                (self.latest_time.tdb.jd - 2451545.0) * 86400.0 - planet_init[i],
                planet_intlen[i],
            )
            omegas = index == n
            index = jnp.where(omegas, index - 1, index)
            late_indecies.append(index)

        early_indecies = (
            jnp.ones(len(early_indecies)) * jnp.min(jnp.array(early_indecies)) * ratios
        ).astype(int)
        new_inits = planet_init + early_indecies * planet_intlen
        late_indecies = (
            jnp.ones(len(late_indecies)) * jnp.min(jnp.array(late_indecies)) * ratios
        ).astype(int)
        trimmed_coeffs = []
        for i in range(len(planet_init)):
            trimmed_coeffs.append(
                planet_coeffs[i][:, :, early_indecies[i] : late_indecies[i]]
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
        shortest_intlen = jnp.min(planet_intlen)
        padded_intlens = jnp.ones(len(planet_intlen)) * shortest_intlen
        extra_padded = []
        for i in range(len(padded_coefficients)):
            extra_padded.append(
                jnp.tile(
                    padded_coefficients[i], int(planet_intlen[i] / shortest_intlen)
                )
            )

        new_coeff = jnp.array(extra_padded)

        gms = jnp.ones(len(self.planet_ids)) * 1.0  # need to fill in real ones
        return ProcessedEphemeris(new_inits, planet_intlen, new_coeff, gms)

    def state(self, time):
        xs, vs, accs = self.ProcessedEphemeris.state(time.tdb.jd)
        d = {}
        for i, id in enumerate(self.planet_ids):
            d[id] = {
                "x": xs[i],
                "v": vs[i],
                "a": accs[i],
            }
        return d

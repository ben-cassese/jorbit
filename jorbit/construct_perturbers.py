import jax
from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, lax
import pickle

from jplephem.spk import SPK
import astropy.units as u
from astropy.time import Time
from astropy.utils.data import download_file

from jplephem.spk import SPK
import pandas as pd

from .data.constants import (
    all_planets,
    all_planet_nums,
    all_planet_gms,
    large_asteroids,
    large_asteroid_nums,
    large_asteroid_gms,
)


def construct_perturbers(
    planets=all_planets,
    asteroids=large_asteroids,
    earliest_time=Time("1980-01-01"),
    latest_time=Time("2100-01-01"),
):
    assert earliest_time.tdb.jd < latest_time.tdb.jd
    
    assert earliest_time.tdb.jd > 2287184.5, 'The DE440 ephemeris only covers between 1549-12-31 and 2650-01-25. Please adjust earliest and latest times.' 
    assert latest_time.tdb.jd < 2688976.5, 'The DE440 ephemeris only covers between 1549-12-31 and 2650-01-25. Please adjust earliest and latest times.'

    assert isinstance(planets, list)
    assert "sun" not in planets
    assert len(list(set(planets))) == len(planets)
    planets = ["sun"] + planets  # Should always be first

    assert isinstance(asteroids, list)
    assert len(list(set(planets))) == len(planets)

    planet_ids = []
    planet_gms = []
    for p in planets:
        try:
            planet_ids.append(all_planet_nums[p.lower()])
            planet_gms.append(all_planet_gms[p.lower()])
        except:
            print(f"{p} not a valid planet name, not included in integration")

    planet_ephem = "https://ssd.jpl.nasa.gov//ftp/eph/planets/bsp/de440.bsp"
    kernel = SPK.open(download_file(planet_ephem, cache=True))

    planet_init = []
    planet_intlen = []
    planet_coeffs = []
    for id in planet_ids:
        for seg in kernel.segments:
            if seg.target != id:
                continue
            init, intlen, coeff = seg._data
            planet_init.append(jnp.array(init))
            planet_intlen.append(jnp.array(intlen))
            planet_coeffs.append(jnp.array(coeff))

    # ***
    # Was previously using the large asteroids file to allow for outer solar system,
    # but swapping for now to the n16 file and GMs from the same source as the planets
    # to compare more directly to ASSIST
    # ***
    # ############################################################################
    # # table created from:
    # # https://ssd.jpl.nasa.gov/ftp/eph/small_bodies/asteroids_de441/SB441_IOM392R-21-005_perturbers.pdf

    # asteroid_info = pd.read_csv('data/Asteroid_Perturbers.csv')
    # ids = []
    # for body in asteroids:
    #     try:
    #         if isinstance(body, str):
    #             body = body.lower()
    #             ids.append(asteroid_info[asteroid_info['Name/Provisional designation'] == body]['Spice ID'].values[0])
    #             gms.append(asteroid_info[asteroid_info['Name/Provisional designation'] == body]['GM - au3 /d2'].values[0])
    #         elif isinstance(body, int):
    #             ids.append(asteroid_info[asteroid_info['Number'] == body]['Spice ID'].values[0])
    #             gms.append(asteroid_info[asteroid_info['Number'] == body]['GM - au3 /d2'].values[0])
    #     except:
    #         print(f'{body} not a valid asteroid name or number, not included in integration')

    # asteroid_ephem = 'https://ssd.jpl.nasa.gov/ftp/eph/small_bodies/asteroids_de441/sb441-n373.bsp'
    # kernel = SPK.open(download_file(asteroid_ephem, cache=True))
    if len(asteroids) == 0:
        asteroids = ["ceres"]
        no_asteroids = True
    else:
        no_asteroids = False

    asteroid_ids = []
    asteroid_gms = []
    for p in asteroids:
        try:
            asteroid_ids.append(large_asteroid_nums[p.lower()])
            asteroid_gms.append(large_asteroid_gms[p.lower()])
        except:
            print(f"{p} not a valid asteroid name, not included in integration")
    # asteroid_ephem = 'https://ssd.jpl.nasa.gov/ftp/eph/small_bodies/asteroids_de441/sb441-n373.bsp'
    asteroid_ephem = (
        "https://ssd.jpl.nasa.gov/ftp/eph/small_bodies/asteroids_de441/sb441-n16.bsp"
    )
    kernel = SPK.open(download_file(asteroid_ephem, cache=True))

    asteroid_init = []
    asteroid_intlen = []
    asteroid_coeffs = []
    for id in asteroid_ids:
        for seg in kernel.segments:
            if seg.target != id:
                continue
            if seg.start_jd != 1999474.5:
                continue
            init, intlen, coeff = seg._data
            asteroid_init.append(jnp.array(init))
            asteroid_intlen.append(jnp.array(intlen))
            asteroid_coeffs.append(jnp.array(coeff))

    ############################################################################
    # the vectorizing rewrite

    planet_init, planet_intlen, planet_coeffs = vectorize(
        Init=jnp.array(planet_init),
        Intlen=jnp.array(planet_intlen),
        Coeff=planet_coeffs,
        earliest_time=earliest_time.tdb.jd - 60,
        latest_time=latest_time.tdb.jd + 60,
    )

    asteroid_init, asteroid_intlen, asteroid_coeffs = vectorize(
        Init=jnp.array(asteroid_init),
        Intlen=jnp.array(asteroid_intlen),
        Coeff=asteroid_coeffs,
        earliest_time=earliest_time.tdb.jd - 60,
        latest_time=latest_time.tdb.jd + 60,
    )
    if no_asteroids:
        asteroid_init *= 0
        asteroid_intlen *= 0
        asteroid_coeffs *= 0
        asteroid_gms = [0]
    ############################################################################
    # gms = planet_gms + asteroid_gms
    assert len(planet_gms) + len(asteroid_gms) == len(planet_init) + len(asteroid_init)
    return (
        (planet_init, planet_intlen, planet_coeffs),
        (asteroid_init, asteroid_intlen, asteroid_coeffs),
        jnp.array(planet_gms),
        jnp.array(asteroid_gms),
    )


def vectorize(Init, Intlen, Coeff, earliest_time, latest_time):
    init0 = Init[0]
    for i in Init:
        assert i == init0

    # Trim the timespans down to the earliest and latest times
    longest_intlen = jnp.max(Intlen)
    ratios = longest_intlen / Intlen
    early_indecies = []
    late_indecies = []
    for i in range(len(Init)):
        component_count, coefficient_count, n = Coeff[i].shape
        index, offset = jnp.divmod(
            (earliest_time - 2451545.0) * 86400.0 - Init[i], Intlen[i]
        )
        omegas = index == n
        index = jnp.where(omegas, index - 1, index)
        early_indecies.append(index)

        index, offset = jnp.divmod(
            (latest_time - 2451545.0) * 86400.0 - Init[i], Intlen[i]
        )
        omegas = index == n
        index = jnp.where(omegas, index - 1, index)
        late_indecies.append(index)

    early_indecies = (
        jnp.ones(len(early_indecies)) * jnp.min(jnp.array(early_indecies)) * ratios
    ).astype(int)
    new_inits = Init + early_indecies * Intlen
    late_indecies = (
        jnp.ones(len(late_indecies)) * jnp.min(jnp.array(late_indecies)) * ratios
    ).astype(int)
    trimmed_coeffs = []
    for i in range(len(Init)):
        trimmed_coeffs.append(Coeff[i][:, :, early_indecies[i] : late_indecies[i]])

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
    shortest_intlen = jnp.min(Intlen)
    padded_intlens = jnp.ones(len(Intlen)) * shortest_intlen
    extra_padded = []
    for i in range(len(padded_coefficients)):
        extra_padded.append(
            jnp.tile(padded_coefficients[i], int(Intlen[i] / shortest_intlen))
        )

    new_coeff = jnp.array(extra_padded)

    return new_inits, Intlen, new_coeff


(
    STANDARD_PLANET_PARAMS,
    STANDARD_ASTEROID_PARAMS,
    STANDARD_PLANET_GMS,
    STANDARD_ASTEROID_GMS,
) = construct_perturbers()

STANDARD_SUN_PARAMS = [jnp.array([STANDARD_PLANET_PARAMS[i][0]]) for i in range(3)]

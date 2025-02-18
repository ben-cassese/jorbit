"""Test that the processing/manipulation of the DE440 ephmeris agrees with Horizons."""

import jax

jax.config.update("jax_enable_x64", True)
import astropy.units as u
import jax.numpy as jnp
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from jorbit.ephemeris import Ephemeris


def test_planets_ephemeris() -> None:
    """Test that the planets agree with Horizons to w/in 1 m, 1 m/day."""
    t0 = Time("2024-12-01 00:00")

    eph = Ephemeris(ssos="default planets")
    jorbit_ephem = eph.state(t0)

    horizon_names = [
        "sun",
        "mercury barycenter",
        "venus barycenter",
        "earth-moon barycenter",
        "mars barycenter",
        "jupiter barycenter",
        "saturn barycenter",
        "uranus barycenter",
        "neptune barycenter",
        "pluto barycenter",
    ]

    jorbit_names = [
        "sun",
        "mercury",
        "venus",
        "earth",
        "mars",
        "jupiter",
        "saturn",
        "uranus",
        "neptune",
        "pluto",
    ]

    for i in range(len(horizon_names)):
        obj = Horizons(id=horizon_names[i], location="@0", epochs=[t0.tdb.jd])
        vecs = obj.vectors(refplane="earth")
        x0 = jnp.array([vecs["x"][0], vecs["y"][0], vecs["z"][0]])
        v0 = jnp.array([vecs["vx"][0], vecs["vy"][0], vecs["vz"][0]])

        x_err = x0 - jorbit_ephem[jorbit_names[i]]["x"]
        v_err = v0 - jorbit_ephem[jorbit_names[i]]["v"]

        assert (jnp.linalg.norm(x_err) * u.au) < 1 * u.m
        assert (jnp.linalg.norm(v_err) * u.au / u.day) < (1 * u.m / u.day)


def test_asteroids_ephemeris() -> None:
    """Test that the asteroids agree with Horizons to w/in 200 km, 2 km/day."""
    t0 = Time("2024-12-01 00:00")

    eph = Ephemeris(ssos="default solar system")
    jorbit_ephem = eph.state(t0)

    jorbit_names = [
        "ceres",
        "pallas",
        "juno",
        "vesta",
        "iris",
        "hygiea",
        "eunomia",
        "psyche",
        "euphrosyne",
        "europa",
        "cybele",
        "sylvia",
        "thisbe",
        "camilla",
        "davida",
        "interamnia",
    ]

    horizons_names = [
        "1",
        "2",
        "3",
        "4",
        "7",
        "10",
        "15",
        "16",
        "31",
        "52",
        "65",
        "87",
        "88",
        "107",
        "511",
        "704",
    ]

    for i in range(len(jorbit_names)):
        obj = Horizons(
            id=horizons_names[i], location="@0", epochs=[t0.tdb.jd], id_type="smallbody"
        )
        vecs = obj.vectors(refplane="earth")
        x0 = jnp.array([vecs["x"][0], vecs["y"][0], vecs["z"][0]])
        v0 = jnp.array([vecs["vx"][0], vecs["vy"][0], vecs["vz"][0]])

        x_err = x0 - jorbit_ephem[jorbit_names[i]]["x"]
        v_err = v0 - jorbit_ephem[jorbit_names[i]]["v"]

        assert (jnp.linalg.norm(x_err) * u.au) < 200 * u.km
        assert (jnp.linalg.norm(v_err) * u.au / u.day) < (2 * u.km / u.day)

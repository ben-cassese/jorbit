"""Test the Particle class."""

import jax

jax.config.update("jax_enable_x64", True)

import astropy.units as u
import jax.numpy as jnp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from jorbit import Observations, Particle
from jorbit.utils.horizons import (
    horizons_bulk_astrometry_query,
    horizons_bulk_vector_query,
)
from jorbit.utils.states import KeplerianState


def test_integrate() -> None:
    """Test that the integrate method mostly matches Horizons."""
    np.random.seed(0)
    times = Time("2025-01-01") + np.arange(0, 100, 5) * u.day

    for _i in range(10):
        sso = str(np.random.randint(1_000, 500_000))

        horizons_results = horizons_bulk_vector_query(
            target=sso,
            center="500@0",
            times=times,
            disable_astroquery=False,
        )
        horizons_x = jnp.column_stack(
            [
                horizons_results["x"].values,
                horizons_results["y"].values,
                horizons_results["z"].values,
            ]
        )

        p = Particle.from_horizons(name=sso, time=Time("2025-01-01"))

        positions, velocities = p.integrate(times)

        assert (
            jnp.max(jnp.linalg.norm(horizons_x - positions, axis=1)) * u.au.to(u.m)
            < 3 * u.m
        )


def test_ephemeris() -> None:
    """Test that the ephemeris method mostly matches Horizons."""
    np.random.seed(1)
    times = Time("2025-01-01") + np.arange(0, 100, 5) * u.day

    for _i in range(5):
        sso = str(np.random.randint(1_000, 500_000))

        horizons_results = horizons_bulk_astrometry_query(
            target=sso,
            center="695@399",
            times=times,
            disable_astroquery=False,
        )

        p = Particle.from_horizons(name=sso, time=Time("2025-01-01"))

        eph = p.ephemeris(times=times, observer="kitt peak")

        assert (
            np.max(
                SkyCoord(horizons_results["RA"], horizons_results["DEC"], unit=u.deg)
                .separation(eph)
                .to(u.mas)
            )
            < 1 * u.mas
        )


def test_max_likelihood() -> None:
    """Test that the max_likelihood method produces <10 mas residuals w/ Horizons."""
    np.random.seed(2)
    sso = str(np.random.randint(1_000, 500_000))
    nights = [
        Time("2025-01-01 07:00"),
        Time("2025-01-02 07:00"),
        Time("2025-01-05 07:00"),
    ]

    times = []
    for n in nights:
        times.extend([n + i * 1 * u.hour for i in range(3)])
    times = Time(times)

    obj = Horizons(id=sso, location="695@399", epochs=times.utc.jd)
    pts = obj.ephemerides(extra_precision=True, quantities="1")

    coords = SkyCoord(pts["RA"], pts["DEC"], unit=(u.deg, u.deg))
    times = Time(pts["datetime_jd"], format="jd", scale="utc")

    obs = Observations(
        observed_coordinates=coords,
        times=times,
        observatories="kitt peak",
        astrometric_uncertainties=1 * u.arcsec,
    )

    obj = Horizons(id=sso, location="500@0", epochs=times.tdb.jd[0])
    vecs = obj.vectors(refplane="earth")
    true_x0 = jnp.array([vecs["x"], vecs["y"], vecs["z"]]).T[0]
    true_v0 = jnp.array([vecs["vx"], vecs["vy"], vecs["vz"]]).T[0]

    p_perturbed = Particle(
        x=true_x0 + jnp.ones(3) * 1e-1,
        v=true_v0 - jnp.ones(3) * 1e-3,
        time=times[0],
        name="",
        observations=obs,
    )

    p_best_fit = p_perturbed.max_likelihood(verbose=False)

    res_best_fit = p_best_fit.residuals(p_best_fit._keplerian_state)

    res_mags = jnp.linalg.norm(res_best_fit, axis=1) * u.arcsec

    assert np.all(res_mags < 10 * u.mas)


def test_different_gravity() -> None:
    """Test that the integrate method runs with different gravity settings."""
    p = Particle.from_horizons(
        name="274301", time=Time("2025-01-01"), gravity="newtonian planets"
    )
    _ = p.integrate(Time("2025-01-02"))

    p = Particle.from_horizons(
        name="274301", time=Time("2025-01-01"), gravity="newtonian solar system"
    )
    _ = p.integrate(Time("2025-01-02"))

    p = Particle.from_horizons(
        name="274301", time=Time("2025-01-01"), gravity="gr planets"
    )
    _ = p.integrate(Time("2025-01-02"))

    p = Particle.from_horizons(
        name="274301", time=Time("2025-01-01"), gravity="gr solar system"
    )
    _ = p.integrate(Time("2025-01-02"))

    p = Particle.from_horizons(
        name="274301", time=Time("2025-01-01"), gravity="default solar system"
    )
    _ = p.integrate(Time("2025-01-02"))


def test_different_inits() -> None:
    """Test that the different ways to initialize a Particle object work."""
    p = Particle.from_horizons(name="274301", time=Time("2025-01-01"))
    _ = p.integrate(Time("2025-01-02"))

    # directly supply state vectors in barycentric ICRS coordinates, units of AU and AU/day
    p = Particle(
        name="(274301) Wikipedia",
        x=jnp.array([-2.003779703686627, 1.780533558134481, 0.5203350526739642]),
        v=jnp.array(
            [-0.006668390915419885, -0.006621147093559814, -0.002036640485149475]
        ),
        time=Time("2025-01-01"),
    )
    _ = p.integrate(Time("2025-01-02"))

    # use ecliptic orbital elements
    k = KeplerianState(
        semi=jnp.array([2.3785863410573236]),
        ecc=jnp.array([0.14924976664546713]),
        inc=jnp.array([6.733641114294506]),
        Omega=jnp.array([183.37291068678854]),
        omega=jnp.array([140.26341029272996]),
        nu=jnp.array([173.59627946476093]),
        time=Time("2025-01-01").tdb.jd,
    )
    p = Particle(name="(274301) Wikipedia", state=k)
    _ = p.integrate(Time("2025-01-02"))

    c = k.to_cartesian()
    p = Particle(name="(274301) Wikipedia", state=c)
    _ = p.integrate(Time("2025-01-02"))


def test_properties() -> None:
    """Test that the properties of a Particle object work."""
    p = Particle(
        name="(274301) Wikipedia",
        x=jnp.array([-2.003779703686627, 1.780533558134481, 0.5203350526739642]),
        v=jnp.array(
            [-0.006668390915419885, -0.006621147093559814, -0.002036640485149475]
        ),
        time=Time("2025-01-01"),
    )

    _ = repr(p)
    _ = p.cartesian_state
    _ = p.keplerian_state

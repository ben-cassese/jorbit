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
from jorbit.data.constants import SPEED_OF_LIGHT
from jorbit.system import System
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

        positions, _velocities = p.integrate(times)

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
        acceleration_func_kwargs={"c2": SPEED_OF_LIGHT**2},
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


def test_keplerian_integrate() -> None:
    """Test that keplerian propagation is self-consistent (forward-backward roundtrip)."""
    p = Particle.from_horizons(
        name="274301", time=Time("2025-01-01"), gravity="keplerian"
    )

    # Forward propagation
    times_fwd = Time("2025-01-01") + np.arange(1, 31) * u.day
    positions, velocities = p.integrate(times_fwd)
    assert positions.shape == (30, 3)
    assert velocities.shape == (30, 3)

    # Roundtrip: propagate forward 10 days then back to epoch
    pos_fwd, vel_fwd = p.integrate(Time("2025-01-11"))
    from jorbit.utils.states import CartesianState

    state_fwd = CartesianState(
        x=pos_fwd,
        v=vel_fwd,
        time=Time("2025-01-11").tdb.jd,
        acceleration_func_kwargs={"c2": SPEED_OF_LIGHT**2},
    )
    pos_back, vel_back = p.integrate(Time("2025-01-01"), state=state_fwd)

    assert jnp.linalg.norm(pos_back[0] - p._x) * u.au.to(u.m) < 1 * u.m
    assert (
        jnp.linalg.norm(vel_back[0] - p._v) * (u.au / u.day).to(u.m / u.s)
        < 1e-4 * u.m / u.s
    )


def test_keplerian_ephemeris() -> None:
    """Test that keplerian ephemeris is close to Horizons for short timescales."""
    p_nbody = Particle.from_horizons(
        name="274301", time=Time("2025-01-01"), gravity="default solar system"
    )
    p_kepler = Particle.from_horizons(
        name="274301", time=Time("2025-01-01"), gravity="keplerian"
    )

    times = Time("2025-01-01") + np.array([1, 5, 10]) * u.day
    eph_nbody = p_nbody.ephemeris(times, "kitt peak")
    eph_kepler = p_kepler.ephemeris(times, "kitt peak")

    seps = eph_nbody.separation(eph_kepler).to(u.arcsec)
    # keplerian should be within a few arcsec of N-body over 10 days
    assert np.all(seps < 10 * u.arcsec)


def test_keplerian_max_likelihood() -> None:
    """Test that max_likelihood works for keplerian particles with self-consistent obs."""
    p_true = Particle.from_horizons(
        name="274301", time=Time("2025-01-01"), gravity="keplerian"
    )

    # Generate self-consistent keplerian observations
    times = Time("2025-01-01") + [1, 3, 5, 7, 10, 14, 20, 30] * u.day
    eph = p_true.ephemeris(times, "kitt peak")
    obs = Observations(
        observed_coordinates=eph,
        times=times,
        observatories="kitt peak",
        astrometric_uncertainties=1 * u.arcsec,
    )

    # Perturbed particle
    p_perturbed = Particle(
        x=p_true._x + jnp.ones(3) * 1e-4,
        v=p_true._v - jnp.ones(3) * 1e-6,
        time=Time("2025-01-01"),
        observations=obs,
        gravity="keplerian",
    )

    p_fit = p_perturbed.max_likelihood(verbose=False)
    res_fit = p_fit.residuals(p_fit._cartesian_state)
    res_mags = jnp.linalg.norm(res_fit, axis=1) * u.arcsec

    assert np.all(res_mags < 1 * u.mas)
    assert p_fit._is_keplerian


def test_keplerian_properties() -> None:
    """Test properties and init for keplerian particles."""
    p = Particle(
        name="test_keplerian",
        x=jnp.array([-2.0, 1.78, 0.52]),
        v=jnp.array([-0.0067, -0.0066, -0.002]),
        time=Time("2025-01-01"),
        gravity="keplerian",
    )

    assert "Particle" in repr(p)
    assert p._is_keplerian
    assert p.gravity == "keplerian"
    _ = p.cartesian_state
    _ = p.keplerian_state

    # No observations → no likelihood
    assert p.loglike is None
    assert p.residuals is None
    assert p.static_residuals is None

    # Init from KeplerianState
    k = KeplerianState(
        semi=jnp.array([2.3785863410573236]),
        ecc=jnp.array([0.14924976664546713]),
        inc=jnp.array([6.733641114294506]),
        Omega=jnp.array([183.37291068678854]),
        omega=jnp.array([140.26341029272996]),
        nu=jnp.array([173.59627946476093]),
        time=Time("2025-01-01").tdb.jd,
        acceleration_func_kwargs={"c2": SPEED_OF_LIGHT**2},
    )
    p2 = Particle(name="from_elements", state=k, gravity="keplerian")
    _ = p2.integrate(Time("2025-01-02"))


def test_system_keplerian_integrate() -> None:
    """Test that System keplerian integration has correct shapes and roundtrips."""
    p1 = Particle.from_horizons(
        name="274301", time=Time("2025-01-01"), gravity="keplerian"
    )
    p2 = Particle.from_horizons(name="1", time=Time("2025-01-01"), gravity="keplerian")
    sys = System(particles=[p1, p2], gravity="keplerian")

    assert sys._is_keplerian
    assert sys.gravity == "keplerian"

    times_fwd = Time("2025-01-01") + np.arange(1, 11) * u.day
    positions, velocities = sys.integrate(times_fwd)
    assert positions.shape == (10, 2, 3)
    assert velocities.shape == (10, 2, 3)

    # Verify positions at epoch match initial state
    pos_back, _ = sys.integrate(Time("2025-01-01"))

    for i, p in enumerate([p1, p2]):
        assert jnp.linalg.norm(pos_back[0, i] - p._x) * u.au.to(u.m) < 1.0


def test_system_keplerian_ephemeris() -> None:
    """Test System keplerian ephemeris matches individual Particle ephemerides."""
    p1 = Particle.from_horizons(
        name="274301", time=Time("2025-01-01"), gravity="keplerian"
    )
    p2 = Particle.from_horizons(name="1", time=Time("2025-01-01"), gravity="keplerian")
    sys = System(particles=[p1, p2], gravity="keplerian")

    times = Time("2025-01-01") + np.array([1, 5, 10]) * u.day
    eph_sys = sys.ephemeris(times, "kitt peak")

    # Compare against individual particle ephemerides
    eph_p1 = p1.ephemeris(times, "kitt peak")
    eph_p2 = p2.ephemeris(times, "kitt peak")

    # System shape should be (N_particles, N_times)
    assert eph_sys.ra.shape == (2, 3)

    # Each particle's ephemeris should match exactly
    for t_idx in range(3):
        sep1 = eph_sys[0, t_idx].separation(eph_p1[t_idx]).to(u.arcsec)
        sep2 = eph_sys[1, t_idx].separation(eph_p2[t_idx]).to(u.arcsec)
        assert sep1 < 0.001 * u.arcsec
        assert sep2 < 0.001 * u.arcsec


def test_elongation_angle() -> None:
    """Test that the elongation angle calculation is correct."""
    # make sure it agrees with Horizons
    t = Time("2026-01-01")
    obj = Horizons(id="274301", location="695@399", epochs=t.utc.jd)
    eph = obj.ephemerides(quantities="1,23")
    horizons_angle = eph["elong"][0]

    # now use jorbit
    p = Particle.from_horizons(
        name="274301",
        time=t,
    )
    angles = p.is_observable(times=t, observer="kitt peak", return_angle=True)

    assert np.isclose(angles[0], horizons_angle, atol=0.01)

    # also make sure an array of times doesn't crash
    times = Time(jnp.linspace(t.tdb.jd, t.tdb.jd + 10, 5), format="jd", scale="tdb")
    ephem = p.ephemeris(times=times, observer="kitt peak")
    mask = p.is_observable(times=times, observer="kitt peak", ephem=ephem)
    assert mask.shape == (5,)
    assert mask[0] == (angles[0] > np.deg2rad(20))

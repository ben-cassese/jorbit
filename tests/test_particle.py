"""Test the Particle class."""

import os

import jax

jax.config.update("jax_enable_x64", True)

import astropy.units as u
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from jorbit import Observations, Particle
from jorbit.data.constants import SPEED_OF_LIGHT
from jorbit.integrators.budgeted import FORCED_LANDING_STEP_BUDGET
from jorbit.integrators.ias15 import IAS15_MAX_DYNAMIC_STEPS
from jorbit.system import System
from jorbit.utils.horizons import (
    horizons_bulk_astrometry_query,
    horizons_bulk_vector_query,
)
from jorbit.utils.kepler import kepler
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

    assert np.all(res_mags < 20 * u.mas)


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
        time_reference=Time("2025-01-01").tdb.jd,
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

    # The forward-propagated state is at Jan 11; we pass it via state= to
    # integrate back to Jan 1. relative_time must match the *exact* offset
    # the forward integration used (Jan 11 - Jan 1 in TDB is 10.0 + ~3 ns
    # because the TDB-TT correction varies slightly); using the Particle's
    # own conversion guarantees forward/backward symmetry.
    state_fwd = CartesianState(
        x=pos_fwd,
        v=vel_fwd,
        relative_time=p._times_to_offsets(Time("2025-01-11")),
        time_reference=p._t_ref_jd,
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
        time_reference=Time("2025-01-01").tdb.jd,
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


def _assist_position_at(
    x0: np.ndarray,
    v0: np.ndarray,
    epoch_jd: float,
    target_jd: float,
    planets: str,
    asteroids: str,
) -> np.ndarray:
    """ASSIST barycentric position [AU] of a test particle (x0, v0 at epoch_jd TDB).

    Force model matches jorbit's "default solar system": Sun + planets + 16 asteroids
    + parameterized post-Newtonian GR, no harmonics or non-gravitational terms. Callers
    must have imported rebound before assist (done via importorskip in the test) so
    assist's C extension can find librebound.
    """
    import assist
    import rebound

    sim = rebound.Simulation()
    eph = assist.Ephem(planets, asteroids)
    sim.add(
        rebound.Particle(
            x=float(x0[0]),
            y=float(x0[1]),
            z=float(x0[2]),
            vx=float(v0[0]),
            vy=float(v0[1]),
            vz=float(v0[2]),
        )
    )
    sim.t = float(epoch_jd) - eph.jd_ref
    extras = assist.Extras(sim, eph)
    extras.forces = ["SUN", "PLANETS", "ASTEROIDS", "GR_EIH"]
    extras.gr_eih_sources = 11
    extras.integrate_or_interpolate(float(target_jd) - eph.jd_ref)
    p = sim.particles[0]
    return np.array([p.x, p.y, p.z])


@pytest.mark.slow
def test_long_integrate_vs_assist() -> None:
    """Regression: long integrations must not truncate and must match ASSIST both ways.

    A synthetic a=0.1 AU, e=0.2 orbit (period ~11.5 days) integrated 20 years in each
    direction is ~630 revolutions taking >20000 adaptive steps as ONE interval. That
    overflows both backend caps, so the host-side orchestration in
    jorbit.integrators.budgeted must kick in, and both directions are exercised:

    - integrate (forced landing): overflows the 10000-iteration per-interval cap, so
      budgeted_forced_landing subdivides via insert_budget_dummy_times and re-runs.
      Guards the interior[::-1] reversal that mis-placed dummy landings for backward runs
      (which raised RuntimeError on the re-run).
    - integrate_or_interpolate (dense output): overflows the 15000-step buffer, so
      stitched_interpolate stitches multiple chunks. Guards the searchsorted-on-descending
      t_step_starts bug in precompute_interpolation_indices, which returned a -1 step index
      for backward queries and yielded (0, 0, 0) positions.

    A sign or ordering error puts the particle at the wrong orbital phase, an AU-scale
    error; jorbit and ASSIST agree to <15 m here, so 1 km leaves a wide margin.
    """
    pytest.importorskip("rebound")
    pytest.importorskip("assist")
    planets = os.path.expanduser("~/Downloads/linux_p1550p2650.440")
    asteroids = os.path.expanduser("~/Downloads/sb441-n16.bsp")
    if not (os.path.exists(planets) and os.path.exists(asteroids)):
        pytest.skip("ASSIST ephemeris files not available locally")

    t_ref = 2462502.5
    epoch_jd = float(Time(46066.0, format="mjd", scale="utc").tdb.jd) + 20.0 * 365.25
    nu = float(kepler(jnp.asarray(np.radians(180.0)), jnp.asarray(0.2)))
    state = KeplerianState(
        semi=jnp.asarray([0.1]),
        ecc=jnp.asarray([0.2]),
        inc=jnp.asarray([10.0]),
        Omega=jnp.asarray([0.0]),
        omega=jnp.asarray([0.0]),
        nu=jnp.asarray([np.degrees(nu)]),
        acceleration_func_kwargs={"c2": SPEED_OF_LIGHT**2},
        time_reference=jnp.asarray(t_ref),
        relative_time=jnp.asarray(epoch_jd - t_ref),
    )
    cart = state.to_cartesian()
    x0 = np.asarray(cart.x).reshape(-1)
    v0 = np.asarray(cart.v).reshape(-1)

    particle = Particle(
        state=KeplerianState(
            semi=jnp.asarray([2.5]),
            ecc=jnp.asarray([0.1]),
            inc=jnp.asarray([5.0]),
            Omega=jnp.asarray([80.0]),
            omega=jnp.asarray([40.0]),
            nu=jnp.asarray([0.0]),
            acceleration_func_kwargs={"c2": SPEED_OF_LIGHT**2},
            time_reference=t_ref,
        ),
        gravity="default solar system",
    )

    span_days = 20.0 * 365.25
    # (method, min steps proving the no-truncation orchestration was exercised): integrate
    # subdivides once it overflows the per-interval cap; integrate_or_interpolate stitches
    # once it overflows the dense-output buffer.
    methods = [
        ("integrate", FORCED_LANDING_STEP_BUDGET),
        ("integrate_or_interpolate", IAS15_MAX_DYNAMIC_STEPS),
    ]
    for method, min_steps in methods:
        fn = getattr(particle, method)
        for label, target_jd in [
            ("backward", epoch_jd - span_days),
            ("forward", epoch_jd + span_days),
        ]:
            pos, _, steps = fn(
                times=Time(target_jd, format="jd", scale="tdb"),
                state=state,
                return_steps=True,
            )
            assert int(np.asarray(steps).sum()) > min_steps

            assist_x = _assist_position_at(
                x0, v0, epoch_jd, target_jd, planets, asteroids
            )
            err_m = float(np.linalg.norm(np.asarray(pos[0]) - assist_x)) * u.au.to(u.m)
            assert (
                err_m < 1_000
            ), f"{method} {label} vs ASSIST: {err_m:.1f} m exceeds 1 km"


def test_static_residuals() -> None:
    """Regression test: static_residuals must not return NaN when particle epoch = first obs.

    precompute_likelihood_data seeds IAS15 with dt = obs_times[0] - t0.  When the
    particle epoch equals the first observation time (the common case) that gap is 0,
    which causes IAS15 to divide by zero and produce NaN for every precomputed step.
    All subsequent static_residuals calls then return NaN.
    """
    nights = [
        Time("2025-01-01 07:00"),
        Time("2025-01-02 07:00"),
        Time("2025-01-05 07:00"),
    ]
    times = []
    for n in nights:
        times.extend([n + i * 1 * u.hour for i in range(3)])
    times = Time(times)

    obj = Horizons(id="274301", location="695@399", epochs=times.utc.jd)
    pts = obj.ephemerides(extra_precision=True, quantities="1")
    coords = SkyCoord(pts["RA"], pts["DEC"], unit=(u.deg, u.deg))
    times = Time(pts["datetime_jd"], format="jd", scale="utc")

    obs = Observations(
        observed_coordinates=coords,
        times=times,
        observatories="kitt peak",
        astrometric_uncertainties=1 * u.arcsec,
    )

    obj = Horizons(id="274301", location="500@0", epochs=times.tdb.jd[0])
    vecs = obj.vectors(refplane="earth")
    true_x0 = jnp.array([vecs["x"], vecs["y"], vecs["z"]]).T[0]
    true_v0 = jnp.array([vecs["vx"], vecs["vy"], vecs["vz"]]).T[0]

    # Epoch = first observation time: previously caused dt_seed=0 → NaN.
    p = Particle(
        x=true_x0,
        v=true_v0,
        time=times[0],
        observations=obs,
    )

    # Residuals must be finite.
    res = p.static_residuals(p.cartesian_state)
    assert jnp.all(jnp.isfinite(res)), "static_residuals returned NaN or Inf"

    # For the true (noiseless Horizons) orbit the residuals must be sub-mas.
    assert float(jnp.max(jnp.abs(res))) < 1e-3  # 1 mas threshold

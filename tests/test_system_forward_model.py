"""Tests for the System-level fast batched forward model (System + shared Observations)."""

import jax

jax.config.update("jax_enable_x64", True)

import astropy.units as u
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time

from jorbit import Observations, Particle, System

EPOCH = Time("2025-01-01")


def _build_particle_and_obs() -> tuple[Particle, Observations, SkyCoord, Time]:
    """A Particle with self-consistent astrometry (its own N-body ephemeris)."""
    p_true = Particle.from_horizons(
        name="274301", time=EPOCH, gravity="default solar system"
    )
    obs_times = EPOCH + np.array([1, 3, 5, 7, 10, 14, 20, 30, 45, 60]) * u.day
    eph_true = p_true.ephemeris(obs_times, "kitt peak")
    obs = Observations(
        observed_coordinates=eph_true,
        times=obs_times,
        observatories="kitt peak",
        astrometric_uncertainties=0.1 * u.arcsec,
    )
    p = Particle(
        x=p_true._x,
        v=p_true._v,
        time=EPOCH,
        observations=obs,
        gravity="default solar system",
    )
    return p, obs, eph_true, obs_times


def test_system_forward_model_accuracy_and_likelihood() -> None:
    """model_radec matches Particle.ephemeris; loglike/residuals match Particle."""
    p, obs, eph_true, _ = _build_particle_and_obs()
    system = System(particles=[p], observations=obs, gravity="default solar system")

    assert system.loglike is not None
    assert system.residuals is not None
    assert system.chi2 is not None
    assert system.model_radec is not None

    true_state = jnp.concatenate([jnp.asarray(p._x), jnp.asarray(p._v)])[None, :]

    # 1. model_radec vs the trusted single-particle ephemeris path.
    ras, decs = system.model_radec(true_state)
    model_sc = SkyCoord(
        ra=np.asarray(ras[0]), dec=np.asarray(decs[0]), unit=u.rad, frame="icrs"
    )
    seps = model_sc.separation(eph_true).to(u.mas)
    assert np.all(seps < 1 * u.mas)

    # 2. loglike / residuals cross-check against Particle (same covariance convention).
    ll_sys = float(system.loglike(true_state)[0])
    ll_particle = float(p.loglike(p._cartesian_state))
    assert abs(ll_sys - ll_particle) < 1e-3

    res_sys = np.asarray(system.residuals(true_state)[0])
    res_particle = np.asarray(p.residuals(p._cartesian_state))
    assert np.abs(res_sys - res_particle).max() < 1e-6


def test_system_forward_model_batches_over_states() -> None:
    """A (P, 6) batch of candidate states yields per-particle finite outputs."""
    p, obs, _, obs_times = _build_particle_and_obs()
    system = System(particles=[p], observations=obs, gravity="default solar system")

    true_state = jnp.concatenate([jnp.asarray(p._x), jnp.asarray(p._v)])
    rng = np.random.default_rng(0)
    batch = jnp.asarray(
        np.asarray(true_state)[None, :] + 1e-9 * rng.standard_normal((16, 6))
    )

    ll = system.loglike(batch)
    assert ll.shape == (16,)
    assert np.all(np.isfinite(ll))

    res = system.residuals(batch)
    assert res.shape == (16, len(obs_times), 2)
    assert np.all(np.isfinite(res))


def test_system_forward_model_ias15_max_steps() -> None:
    """An arc-sized buffer is bit-identical; an undersized one fails loudly."""
    p, obs, _, _ = _build_particle_and_obs()
    system = System(particles=[p], observations=obs, gravity="default solar system")
    system_small = System(
        particles=[p],
        observations=obs,
        gravity="default solar system",
        ias15_max_steps=64,
    )
    system_tiny = System(
        particles=[p],
        observations=obs,
        gravity="default solar system",
        ias15_max_steps=2,
    )

    true_state = jnp.concatenate([jnp.asarray(p._x), jnp.asarray(p._v)])[None, :]

    # 64 steps comfortably cover the 60-day arc: outputs are bit-identical.
    ll_ref = np.asarray(system.loglike(true_state))
    assert np.array_equal(np.asarray(system_small.loglike(true_state)), ll_ref)
    ras_ref, decs_ref = system.model_radec(true_state)
    ras_small, decs_small = system_small.model_radec(true_state)
    assert np.array_equal(np.asarray(ras_small), np.asarray(ras_ref))
    assert np.array_equal(np.asarray(decs_small), np.asarray(decs_ref))

    # 2 steps cannot: truncation is loud (-inf loglike, NaN-poisoned radec).
    assert np.all(np.asarray(system_tiny.loglike(true_state)) == -np.inf)
    ras_tiny, _ = system_tiny.model_radec(true_state)
    assert np.any(np.isnan(np.asarray(ras_tiny)))


def test_system_forward_model_none_when_unavailable() -> None:
    """The fast attributes are None without observations or on the keplerian path."""
    p, obs, _, _ = _build_particle_and_obs()

    sys_no_obs = System(particles=[p], gravity="default solar system")
    assert sys_no_obs.loglike is None
    assert sys_no_obs.residuals is None
    assert sys_no_obs.chi2 is None
    assert sys_no_obs.model_radec is None
    assert sys_no_obs.model_radec_with_status is None

    p_kep = Particle.from_horizons(name="274301", time=EPOCH, gravity="keplerian")
    sys_kep = System(particles=[p_kep], observations=obs, gravity="keplerian")
    assert sys_kep.loglike is None
    assert sys_kep.model_radec is None
    assert sys_kep.model_radec_with_status is None


def test_model_radec_with_status_identifies_truncation() -> None:
    """The status twin matches model_radec exactly and locates the NaNs."""
    p, obs, _, _ = _build_particle_and_obs()
    system = System(particles=[p], observations=obs, gravity="default solar system")
    system_tiny = System(
        particles=[p],
        observations=obs,
        gravity="default solar system",
        ias15_max_steps=2,
    )
    true_state = jnp.concatenate([jnp.asarray(p._x), jnp.asarray(p._v)])[None, :]

    # A buffer that covers the arc: nothing truncates, and the coordinates are a pure
    # superset of model_radec's rather than a re-derivation.
    ras_ref, decs_ref = system.model_radec(true_state)
    ras, decs, reached, ltt_covered = system.model_radec_with_status(true_state)
    assert np.all(np.asarray(reached))
    assert np.all(np.asarray(ltt_covered))
    assert np.array_equal(np.asarray(ras), np.asarray(ras_ref))
    assert np.array_equal(np.asarray(decs), np.asarray(decs_ref))

    # A 2-step buffer cannot: the mask is False exactly where the coordinates are NaN,
    # which is what separates truncation from a NaN acceleration downstream.
    ras_tiny, decs_tiny, reached_tiny, ltt_tiny = system_tiny.model_radec_with_status(
        true_state
    )
    reached_tiny = np.asarray(reached_tiny)
    assert not np.all(reached_tiny)
    assert np.array_equal(np.isnan(np.asarray(ras_tiny)), ~reached_tiny[None, :])
    assert np.array_equal(np.isnan(np.asarray(decs_tiny)), ~reached_tiny[None, :])
    # The two masks stay independent: an observation the integration never reached has no
    # meaningful light travel time, so ltt_covered must not also drop there. Otherwise both
    # bits fall together and the caller loses the ability to tell the causes apart, which
    # is the whole reason this callable exists.
    assert np.all(np.asarray(ltt_tiny))


def test_probe_span() -> None:
    """probe_span predicts the truncation the forward model would hit."""
    p, obs, _, obs_times = _build_particle_and_obs()
    system = System(particles=[p], observations=obs, gravity="default solar system")
    system_tiny = System(
        particles=[p],
        observations=obs,
        gravity="default solar system",
        ias15_max_steps=2,
    )

    would_truncate, steps = system.probe_span()
    assert not would_truncate
    assert steps > 0
    # Explicit times give the same answer as falling back to the observations.
    assert system.probe_span(obs_times) == (would_truncate, steps)

    would_truncate_tiny, steps_tiny = system_tiny.probe_span()
    assert would_truncate_tiny
    assert steps_tiny > 0

    # Paths that land on every requested time by construction have no buffer to
    # overflow, so they report no truncation without integrating.
    p_kep = Particle.from_horizons(name="274301", time=EPOCH, gravity="keplerian")
    sys_kep = System(particles=[p_kep], observations=obs, gravity="keplerian")
    assert sys_kep.probe_span() == (False, 0)

    # No observations to fall back on and no times given is a user error, not a guess.
    sys_no_obs = System(particles=[p], gravity="default solar system")
    with pytest.raises(ValueError, match=r"no.*attached Observations"):
        sys_no_obs.probe_span()
    assert sys_no_obs.probe_span(obs_times) == (would_truncate, steps)

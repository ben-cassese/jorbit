"""Tests for the Observations container."""

import jax

jax.config.update("jax_enable_x64", True)

import astropy.units as u
import jax.numpy as jnp
from astropy.coordinates import SkyCoord
from astropy.time import Time

from jorbit import Observations


def _make_observations(day_offsets: list[float]) -> Observations:
    offsets = jnp.asarray(day_offsets)
    count = len(day_offsets)
    coords = SkyCoord(
        ra=0.1 + 0.01 * offsets,
        dec=0.2 + 0.01 * offsets,
        unit=u.rad,
    )
    times = Time("2025-01-01").tdb.jd + offsets
    observatories = jnp.arange(count * 3, dtype=float).reshape(count, 3)
    uncertainties = jnp.ones(count) * u.arcsec

    return Observations(
        observed_coordinates=coords,
        times=times,
        observatories=observatories,
        astrometric_uncertainties=uncertainties,
    )


def test_integer_getitem_preserves_single_observation_shapes() -> None:
    """Integer indexing should return a length-one Observations object."""
    obs = _make_observations([0, 1, 2])

    first = obs[0]

    assert len(first) == 1
    assert first.ra.shape == (1,)
    assert first.dec.shape == (1,)
    assert first.times.shape == (1,)
    assert first.observatories.shape == (1, 3)
    assert first.astrometric_uncertainties.shape == (1,)


def test_added_observations_support_fit_seed_indexing_pattern() -> None:
    """Combined observations should support the indexing used for fit seeds."""
    obs = _make_observations([0, 2]) + _make_observations([1])
    mid_idx = jnp.argmin(jnp.abs(obs.times - jnp.mean(obs.times)))

    seed_obs = obs[0] + obs[mid_idx] + obs[-1]

    assert len(seed_obs) == 3
    assert seed_obs.observatories.shape == (3, 3)

"""Compare dynamic loglike vs static_residuals-based loglike for consistency."""

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

# Fixed seed for reproducibility, pick 10 asteroid IDs
_rng = np.random.RandomState(42)
SSO_IDS = [str(_rng.randint(1_000, 500_000)) for _ in range(10)]


def _make_observations_and_particle(sso) -> Particle:
    """Create observations and a Particle from Horizons for a given SSO."""
    nights = [
        Time("2025-01-01 07:00"),
        Time("2025-01-02 07:00"),
    ]
    times = []
    for n in nights:
        times.extend([n, n + 1 * u.hour])
    times = Time(times)

    obj = Horizons(id=sso, location="695@399", epochs=times.utc.jd)
    pts = obj.ephemerides(extra_precision=True, quantities="1")

    coords = SkyCoord(pts["RA"], pts["DEC"], unit=(u.deg, u.deg))
    obs_times = Time(pts["datetime_jd"], format="jd", scale="utc")

    obs = Observations(
        observed_coordinates=coords,
        times=obs_times,
        observatories="kitt peak",
        astrometric_uncertainties=1 * u.arcsec,
    )

    p = Particle.from_horizons(name=sso, time=obs_times[0], observations=obs)
    return p


def _static_loglike(p) -> jnp.ndarray:
    """Compute log-likelihood from static_residuals using the same formula as _loglike."""
    xis_etas = p.static_residuals(p._cartesian_state)
    inv_cov = p._observations.inv_cov_matrices
    log_dets = p._observations.cov_log_dets

    quad = jnp.einsum("bi,bij,bj->b", xis_etas, inv_cov, xis_etas)
    ll = jnp.sum(-0.5 * (2 * jnp.log(2 * jnp.pi) + log_dets + quad))
    return ll


@pytest.mark.slow
@pytest.mark.parametrize("sso", SSO_IDS)
def test_loglike_comparison(sso) -> None:
    """Verify dynamic loglike and static_residuals-based loglike agree."""
    p = _make_observations_and_particle(sso)

    dynamic_ll = float(p.loglike(p._cartesian_state))
    static_ll = float(_static_loglike(p))

    print(f"\nSSO {sso}:")
    print(f"  dynamic loglike = {dynamic_ll:.6f}")
    print(f"  static  loglike = {static_ll:.6f}")
    print(f"  difference      = {abs(dynamic_ll - static_ll):.6e}")

    # Both methods should produce very similar log-likelihoods.
    # Placeholder tolerance — tighten after inspecting initial results.
    assert np.isclose(dynamic_ll, static_ll, atol=1.0), (
        f"SSO {sso}: dynamic ({dynamic_ll:.6f}) vs static ({static_ll:.6f}) "
        f"differ by {abs(dynamic_ll - static_ll):.6e}"
    )

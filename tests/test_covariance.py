"""Tests for the uncertainty=True path of Particle.ephemeris."""

import jax

jax.config.update("jax_enable_x64", True)

import astropy.units as u
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time

from jorbit import Particle
from jorbit.utils.states import CartesianState

RAD2ARCSEC = 180.0 / np.pi * 3600.0


def _build_test_particle() -> Particle:
    return Particle.from_horizons(name="274301", time=Time("2025-01-01"))


def _attach_cart_cov(state: CartesianState, cov: jnp.ndarray) -> CartesianState:
    return state.replace(cov=cov)


def test_ephemeris_uncertainty_cartesian() -> None:
    """Cartesian-cov path returns a well-formed EphemerisWithUncertainty."""
    p = _build_test_particle()
    t0 = Time("2025-01-01")
    times = t0 + np.arange(0, 101, 10) * u.day
    N = len(times)

    sigma_p = 1e-6  # AU (~150 m)
    sigma_v = 1e-7  # AU/day
    cov = jnp.diag(jnp.array([sigma_p**2] * 3 + [sigma_v**2] * 3))
    cs = _attach_cart_cov(p.cartesian_state, cov)

    result = p.ephemeris(times, "palomar", state=cs, uncertainty=True)

    assert isinstance(result, tuple)
    assert isinstance(result[0], SkyCoord)
    assert result[0].shape == (N,)
    assert result[1].shape == (N, 2, 2)

    cov_radec = np.asarray(result[1])
    assert np.all(np.isfinite(cov_radec))

    # Symmetric (machine precision)
    sym_err = np.max(np.abs(cov_radec - np.transpose(cov_radec, (0, 2, 1))))
    assert sym_err < 1e-15

    # Positive semi-definite at every time
    eigvals = np.linalg.eigvalsh(cov_radec)
    assert np.all(eigvals >= -1e-18)

    # Diagonals are non-negative and grow monotonically in time
    diag = np.diagonal(cov_radec, axis1=1, axis2=2)  # shape (N, 2)
    assert np.all(diag >= 0)
    assert np.all(np.diff(diag, axis=0) >= -1e-18)


def test_ephemeris_uncertainty_keplerian() -> None:
    """Keplerian-cov path: covariance is propagated from element space."""
    p = Particle.from_horizons(
        name="274301", time=Time("2025-01-01"), gravity="keplerian"
    )
    t0 = Time("2025-01-01")
    times = t0 + np.arange(0, 101, 10) * u.day
    N = len(times)

    kep = p.keplerian_state
    # Small, ad-hoc diagonal cov in element space.
    cov_kep = jnp.diag(jnp.array([1e-10, 1e-10, 1e-8, 1e-8, 1e-8, 1e-6]))
    kep_with_cov = kep.replace(cov=cov_kep)

    result = p.ephemeris(times, "palomar", state=kep_with_cov, uncertainty=True)
    assert isinstance(result, tuple)
    assert result[1].shape == (N, 2, 2)

    cov_radec = np.asarray(result[1])
    assert np.all(np.isfinite(cov_radec))
    eigvals = np.linalg.eigvalsh(cov_radec)
    assert np.all(eigvals >= -1e-18)


def test_ephemeris_uncertainty_finite_difference() -> None:
    """The propagated sigma agrees with a finite-difference perturbation.

    Build a cov that puts all variance into x[0], so the propagated
    sigma_RA at each time is exactly |dRA/dx0| * sigma_pos. Compare against
    a tiny finite-difference perturbation in x[0].
    """
    p = _build_test_particle()
    t0 = Time("2025-01-01")
    times = t0 + np.arange(0, 31, 10) * u.day  # short span, stay linear

    sigma_pos = 1e-8  # AU; small enough to be safely linear
    cov = jnp.zeros((6, 6)).at[0, 0].set(sigma_pos**2)
    cs = _attach_cart_cov(p.cartesian_state, cov)

    result = p.ephemeris(times, "palomar", state=cs, uncertainty=True)
    sigma_ra_prop = np.sqrt(np.asarray(result[1][:, 0, 0]))  # arcsec
    sigma_dec_prop = np.sqrt(np.asarray(result[1][:, 1, 1]))

    # Finite-difference perturbation in x[0].
    cs_pert = cs.replace(x=cs.x.at[0, 0].add(sigma_pos), cov=jnp.empty((0, 0)))
    eph_pert = p.ephemeris(times, "palomar", state=cs_pert)

    ra_nom = np.asarray(result[0].ra.rad)
    dec_nom = np.asarray(result[0].dec.rad)
    sigma_ra_fd = np.abs(np.asarray(eph_pert.ra.rad) - ra_nom) * RAD2ARCSEC
    sigma_dec_fd = np.abs(np.asarray(eph_pert.dec.rad) - dec_nom) * RAD2ARCSEC

    # In the linear regime the ratios are 1 to high precision.
    assert np.allclose(sigma_ra_prop, sigma_ra_fd, rtol=1e-3, atol=0)
    assert np.allclose(sigma_dec_prop, sigma_dec_fd, rtol=1e-3, atol=0)


def test_ephemeris_uncertainty_no_cov_raises() -> None:
    """uncertainty=True without a (6, 6) cov on the state raises ValueError."""
    p = _build_test_particle()
    t0 = Time("2025-01-01")
    times = t0 + np.arange(0, 11, 5) * u.day

    with pytest.raises(ValueError, match=r"\(6, 6\) covariance matrix"):
        p.ephemeris(times, "palomar", uncertainty=True)

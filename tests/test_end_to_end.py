"""End-to-end tests probing time handling and object combination edge cases.

These tests target the specific weaknesses identified in the May 2026 bug hunt:
precision preservation through __getitem__ / __add__, System epoch extraction,
heliocentric coordinate roundtrips, pre-built covariance matrices, and
consistency between the static and dynamic residual pipelines.
"""

import jax

jax.config.update("jax_enable_x64", True)

import astropy.units as u
import jax.numpy as jnp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time

from jorbit import Observations, Particle
from jorbit.data.constants import SPEED_OF_LIGHT
from jorbit.system import System
from jorbit.utils.states import (
    CartesianState,
    barycentric_to_heliocentric,
    heliocentric_to_barycentric,
)

# ---------------------------------------------------------------------------
# Shared synthetic data (no network calls; no Horizons)
# ---------------------------------------------------------------------------

_X0 = jnp.array([-2.003779703686627, 1.780533558134481, 0.5203350526739642])
_V0 = jnp.array([-0.006668390915419885, -0.006621147093559814, -0.002036640485149475])
_T0 = Time("2025-01-01")


def _make_obs_with_astropy_times(n_obs: int = 5) -> Observations:
    """Minimal Observations object with genuine astropy Time precision.

    Uses pre-computed observer positions so no Horizons query is needed.
    """
    times = _T0 + np.arange(n_obs) * u.day
    coords = SkyCoord(
        ra=np.linspace(0.10, 0.10 + 0.01 * (n_obs - 1), n_obs) * u.rad,
        dec=np.zeros(n_obs) * u.rad,
    )
    # Fake observer positions (we are only testing structure, not astrometry)
    observer_positions = jnp.zeros((n_obs, 3))
    return Observations(
        observed_coordinates=coords,
        times=times,
        observatories=observer_positions,
        astrometric_uncertainties=1.0 * u.arcsec,
    )


# ===========================================================================
# 1a.  Observations.__getitem__ should preserve times_astropy
# ===========================================================================


def test_observations_getitem_integer_preserves_times_astropy() -> None:
    """Integer-indexing a high-precision Observations must keep times_astropy."""
    obs = _make_obs_with_astropy_times(5)
    assert obs.times_astropy is not None, "baseline: times_astropy must be set"

    sliced = obs[2]
    assert sliced.times_astropy is not None, (
        "__getitem__(int) dropped times_astropy; downstream static_residuals "
        "will silently fall back to float arithmetic"
    )


def test_observations_getitem_slice_preserves_times_astropy() -> None:
    """Slice-indexing a high-precision Observations must keep times_astropy."""
    obs = _make_obs_with_astropy_times(5)

    sliced = obs[1:3]
    assert sliced.times_astropy is not None, "__getitem__(slice) dropped times_astropy"
    assert len(sliced.times_astropy) == 2


def test_observations_sliced_then_added_preserves_times_astropy() -> None:
    """Slicing then combining (the fit-seed pattern) must keep times_astropy."""
    obs = _make_obs_with_astropy_times(5)

    combined = obs[0] + obs[2] + obs[-1]
    assert (
        combined.times_astropy is not None
    ), "obs[0] + obs[2] + obs[-1] dropped times_astropy"
    assert len(combined.times_astropy) == 3


# ===========================================================================
# 1c.  System(state=SystemState) recovers the epoch from the self-describing state
# ===========================================================================


def test_system_from_systemstate_recovers_epoch() -> None:
    """System(state=SystemState) recovers the epoch from relative_time + time_reference.

    A SystemState is now self-describing (it carries the same
    ``(relative_time, time_reference)`` pair as CartesianState/KeplerianState), so
    ``System(state=p.cartesian_state.to_system())`` should reproduce the particle's
    epoch instead of raising.
    """
    p = Particle(x=_X0, v=_V0, time=_T0, gravity="newtonian planets")
    sys_state = p.cartesian_state.to_system()
    # sys_state carries relative_time=0.0, time_reference=p._t_ref_jd

    sys = System(state=sys_state, gravity="newtonian planets")

    assert abs(float(sys._t_ref_jd) - float(p._t_ref_jd)) < 1e-6, (
        f"System epoch ({float(sys._t_ref_jd)}) doesn't match Particle epoch "
        f"({float(p._t_ref_jd)})"
    )
    # The rebased state should carry relative_time=0.0 against the recovered anchor.
    assert abs(float(sys._state.relative_time)) < 1e-9
    assert abs(float(sys._state.time_reference) - float(p._t_ref_jd)) < 1e-6


def test_system_from_particles_has_correct_epoch() -> None:
    """The standard System(particles=[...]) path must preserve the Particle epoch."""
    p = Particle(x=_X0, v=_V0, time=_T0, gravity="newtonian planets")
    sys = System(particles=[p], gravity="newtonian planets")

    assert (
        abs(sys._t_ref_jd - p._t_ref_jd) < 1e-6
    ), f"System epoch ({sys._t_ref_jd}) doesn't match Particle epoch ({p._t_ref_jd})"


def test_system_from_particles_integrates_consistently() -> None:
    """System([p]) must give the same 1-day position as Particle.integrate()."""
    p = Particle(x=_X0, v=_V0, time=_T0, gravity="newtonian planets")
    sys = System(particles=[p], gravity="newtonian planets")

    target = _T0 + 1 * u.day
    pos_p, _ = p.integrate(target)  # shape (1, 3)
    pos_s, _ = sys.integrate(target)  # shape (1, 1, 3)

    diff_m = float(jnp.linalg.norm(pos_p[0] - pos_s[0, 0])) * u.au.to(u.m)
    assert diff_m < 1000.0, f"Position disagreement after 1 day: {diff_m:.0f} m"


# ===========================================================================
# 1d.  heliocentric_to_barycentric / barycentric_to_heliocentric roundtrip
# ===========================================================================


def test_heliocentric_barycentric_cartesian_roundtrip() -> None:
    """Bary -> helio -> bary must recover the original Cartesian state."""
    t = Time("2025-06-01")
    state = CartesianState(
        x=jnp.array([_X0]),
        v=jnp.array([_V0]),
        time_reference=t.tdb.jd,
        acceleration_func_kwargs={"c2": SPEED_OF_LIGHT**2},
    )

    helio = barycentric_to_heliocentric(state, t)
    recovered = heliocentric_to_barycentric(helio, t)

    assert jnp.allclose(
        recovered.x.flatten(), state.x.flatten(), atol=1e-10
    ), "Position roundtrip error exceeds 1e-10 AU"
    assert jnp.allclose(
        recovered.v.flatten(), state.v.flatten(), atol=1e-12
    ), "Velocity roundtrip error exceeds 1e-12 AU/day"


def test_heliocentric_barycentric_keplerian_roundtrip() -> None:
    """Bary -> helio (Keplerian form) -> bary must recover the original Cartesian state."""
    t = Time("2025-06-01")
    state = CartesianState(
        x=jnp.array([_X0]),
        v=jnp.array([_V0]),
        time_reference=t.tdb.jd,
        acceleration_func_kwargs={"c2": SPEED_OF_LIGHT**2},
    )
    kep = state.to_keplerian()

    helio = barycentric_to_heliocentric(kep, t)
    recovered = heliocentric_to_barycentric(helio, t)  # returns KeplerianState

    # Compare via Cartesian
    rec_cart = recovered.to_cartesian()
    assert jnp.allclose(
        rec_cart.x.flatten(), state.x.flatten(), atol=1e-9
    ), "Keplerian roundtrip position error exceeds 1e-9 AU"
    assert jnp.allclose(
        rec_cart.v.flatten(), state.v.flatten(), atol=1e-11
    ), "Keplerian roundtrip velocity error exceeds 1e-11 AU/day"


# ===========================================================================
# 1e.  Observations with pre-built 2x2 covariance matrices
# ===========================================================================


def test_observations_prebuilt_covariance_matrix() -> None:
    """Passing a (N, 2, 2) array as astrometric_uncertainties should work."""
    n = 4
    times = _T0 + np.arange(n) * u.day
    coords = SkyCoord(
        ra=np.linspace(0.10, 0.13, n) * u.rad,
        dec=np.zeros(n) * u.rad,
    )
    observer_positions = jnp.zeros((n, 3))

    sigma = 1.0  # arcsec
    rho = 0.3
    cov_1 = jnp.array([[sigma**2, rho * sigma**2], [rho * sigma**2, sigma**2]])
    cov_matrices = jnp.broadcast_to(cov_1, (n, 2, 2))

    obs = Observations(
        observed_coordinates=coords,
        times=times,
        observatories=observer_positions,
        astrometric_uncertainties=cov_matrices,
    )

    assert obs.cov_matrices.shape == (n, 2, 2)
    assert obs.inv_cov_matrices.shape == (n, 2, 2)
    assert obs.cov_log_dets.shape == (n,)

    for i in range(n):
        product = obs.inv_cov_matrices[i] @ obs.cov_matrices[i]
        assert jnp.allclose(product, jnp.eye(2), atol=1e-10), (
            f"inv_cov @ cov != I at index {i}: max off-diag = "
            f"{float(jnp.max(jnp.abs(product - jnp.eye(2)))):.2e}"
        )


# ===========================================================================
# 1f.  Particle initialised from state with non-zero relative_time
# ===========================================================================


def test_particle_from_state_with_nonzero_relative_time() -> None:
    """Particle from a state with nonzero relative_time must read epoch correctly.

    ``relative_time + time_reference`` is the absolute epoch. Verified by
    integrating back 30 days and comparing with the original position.
    """
    p_orig = Particle(x=_X0, v=_V0, time=_T0, gravity="newtonian planets")

    t_fwd = _T0 + 30 * u.day
    pos_fwd, vel_fwd = p_orig.integrate(t_fwd)  # shape (1, 3)

    offset_fwd = p_orig._times_to_offsets(t_fwd)  # should be +30.something days
    state_fwd = CartesianState(
        x=pos_fwd.reshape(1, 3),
        v=vel_fwd.reshape(1, 3),
        relative_time=offset_fwd,
        time_reference=p_orig._t_ref_jd,
        acceleration_func_kwargs={"c2": SPEED_OF_LIGHT**2},
    )

    # Particle should interpret relative_time + time_reference = t_fwd as epoch
    p_fwd = Particle(state=state_fwd, gravity="newtonian planets")

    expected_jd = t_fwd.tdb.jd
    assert abs(p_fwd._t_ref_jd - expected_jd) < 1e-6, (
        f"p_fwd epoch mismatch: got {p_fwd._t_ref_jd:.6f}, "
        f"expected {expected_jd:.6f}"
    )

    # Integrating back 30 days should recover the original position
    pos_back, _ = p_fwd.integrate(_T0)
    diff_m = float(jnp.linalg.norm(pos_back[0] - p_orig._x)) * u.au.to(u.m)
    assert (
        diff_m < 1000.0
    ), f"Roundtrip position error: {diff_m:.0f} m (expected < 1000 m)"


# ===========================================================================
# 1b + 1g.  Self-consistent synthetic pipeline: static == dynamic residuals
# ===========================================================================


def test_self_consistent_observations_static_residuals_finite() -> None:
    """Self-consistent observations must yield near-zero finite static residuals.

    A particle whose observations came from its own ephemeris should have
    residuals < 1 mas with no NaN or Inf values.
    """
    p_true = Particle(x=_X0, v=_V0, time=_T0)
    times = _T0 + np.array([1, 3, 7, 15, 30]) * u.day
    eph = p_true.ephemeris(times, "kitt peak")

    obs = Observations(
        observed_coordinates=eph,
        times=times,
        observatories="kitt peak",
        astrometric_uncertainties=1.0 * u.arcsec,
    )

    p = Particle(x=_X0, v=_V0, time=_T0, observations=obs)

    res = p.static_residuals(p.cartesian_state)
    assert jnp.all(jnp.isfinite(res)), "static_residuals returned NaN or Inf"
    assert (
        float(jnp.max(jnp.abs(res))) < 1e-3
    ), f"static_residuals not near-zero: max = {float(jnp.max(jnp.abs(res))):.2e} arcsec"


def test_static_and_dynamic_residuals_agree() -> None:
    """Static and dynamic residuals must agree to < 0.1 mas for a self-consistent orbit."""
    p_true = Particle(x=_X0, v=_V0, time=_T0)
    times = _T0 + np.array([2, 5, 10, 20, 40]) * u.day
    eph = p_true.ephemeris(times, "kitt peak")

    obs = Observations(
        observed_coordinates=eph,
        times=times,
        observatories="kitt peak",
        astrometric_uncertainties=1.0 * u.arcsec,
    )

    p = Particle(x=_X0, v=_V0, time=_T0, observations=obs)

    static_res = p.static_residuals(p.cartesian_state)
    dynamic_res = p.residuals(p.cartesian_state)

    assert jnp.all(jnp.isfinite(static_res)), "static_residuals has NaN/Inf"
    assert jnp.all(jnp.isfinite(dynamic_res)), "dynamic residuals has NaN/Inf"

    diff = jnp.abs(static_res - dynamic_res)
    max_diff_mas = float(jnp.max(diff)) * 1000.0
    assert (
        max_diff_mas < 0.1
    ), f"static vs dynamic residuals disagree by {max_diff_mas:.3f} mas (limit 0.1 mas)"

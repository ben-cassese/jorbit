"""Analytic two-body (Keplerian) integrate/ephemeris/likelihood helpers.

Also imported by the System subpackage (:mod:`jorbit.system.keplerian`), which vmaps
:func:`_keplerian_integrate` / :func:`_keplerian_on_sky` over multiple particles.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jorbit.astrometry.sky_projection import tangent_plane_projection
from jorbit.astrometry.transformations import (
    horizons_ecliptic_to_icrs,
    icrs_to_horizons_ecliptic,
)
from jorbit.data.constants import INV_SPEED_OF_LIGHT, TOTAL_SOLAR_SYSTEM_GM
from jorbit.particle.covariance import (
    _cov_from_jacobian,
    _state_to_vec,
    _state_vec_to_xv,
)
from jorbit.utils.kepler import keplerian_propagate
from jorbit.utils.states import CartesianState, KeplerianState


@jax.jit
def _keplerian_integrate(
    x: jnp.ndarray,
    v: jnp.ndarray,
    t0: float,
    times: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    x_ecl = icrs_to_horizons_ecliptic(x[None, :])
    v_ecl = icrs_to_horizons_ecliptic(v[None, :])

    positions_ecl, velocities_ecl = keplerian_propagate(
        x_ecl, v_ecl, t0, times, TOTAL_SOLAR_SYSTEM_GM
    )

    positions = horizons_ecliptic_to_icrs(positions_ecl)
    velocities = horizons_ecliptic_to_icrs(velocities_ecl)
    return positions, velocities


@jax.jit
def _keplerian_on_sky(
    x: jnp.ndarray,
    v: jnp.ndarray,
    time: float,
    observer_position: jnp.ndarray,
) -> tuple[float, float]:
    r = jnp.linalg.norm(x)
    a0 = -TOTAL_SOLAR_SYSTEM_GM * x / (r**3)

    xz = x
    for _ in range(3):
        earth_distance = jnp.linalg.norm(xz - observer_position)
        dt = -earth_distance * INV_SPEED_OF_LIGHT
        xz = x + v * dt + 0.5 * a0 * dt * dt

    X = xz - observer_position
    calc_ra = jnp.mod(jnp.arctan2(X[1], X[0]) + 2 * jnp.pi, 2 * jnp.pi)
    calc_dec = jnp.pi / 2 - jnp.arccos(X[-1] / jnp.linalg.norm(X))
    return calc_ra, calc_dec


@jax.jit
def _keplerian_ephem(
    x: jnp.ndarray,
    v: jnp.ndarray,
    t0: float,
    times: jnp.ndarray,
    observer_positions: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    positions, velocities = _keplerian_integrate(x, v, t0, times)

    def scan_func(carry: None, scan_over: tuple) -> tuple[None, tuple]:
        position, velocity, time, observer_position = scan_over
        ra, dec = _keplerian_on_sky(position, velocity, time, observer_position)
        return None, (ra, dec)

    _, (ras, decs) = jax.lax.scan(
        scan_func,
        None,
        (positions, velocities, times, observer_positions),
    )
    return ras, decs


@jax.jit
def _keplerian_ephem_with_cov(
    particle_state: CartesianState | KeplerianState,
    times: jnp.ndarray,
    observer_positions: jnp.ndarray,
    cov: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Keplerian-path ephemeris with sky-plane covariance via forward-mode AD.

    Supports both Keplerian and Cartesian input parameterizations; the covariance
    is propagated in whichever space the input state was supplied in.
    """
    is_keplerian_param = isinstance(particle_state, KeplerianState)
    t0 = particle_state.relative_time

    def radec_fn(state_vec: jnp.ndarray) -> jnp.ndarray:
        x, v = _state_vec_to_xv(state_vec, is_keplerian_param)
        ras, decs = _keplerian_ephem(
            x.flatten(), v.flatten(), t0, times, observer_positions
        )
        return jnp.stack([ras, decs], axis=1).flatten()

    nominal_vec = _state_to_vec(particle_state)
    return _cov_from_jacobian(radec_fn, nominal_vec, cov, times.shape[0])


@jax.jit
def _keplerian_residuals(
    times: jnp.ndarray,
    observer_positions: jnp.ndarray,
    ra: jnp.ndarray,
    dec: jnp.ndarray,
    particle_state: CartesianState | KeplerianState,
) -> jnp.ndarray:
    x = particle_state.to_cartesian().x.flatten()
    v = particle_state.to_cartesian().v.flatten()
    t0 = particle_state.relative_time

    ras, decs = _keplerian_ephem(x, v, t0, times, observer_positions)
    xis_etas = jax.vmap(tangent_plane_projection)(ra, dec, ras, decs)
    return xis_etas


@jax.jit
def _keplerian_loglike(
    times: jnp.ndarray,
    observer_positions: jnp.ndarray,
    ra: jnp.ndarray,
    dec: jnp.ndarray,
    inv_cov_matrices: jnp.ndarray,
    cov_log_dets: jnp.ndarray,
    particle_state: CartesianState | KeplerianState,
) -> float:
    xis_etas = _keplerian_residuals(times, observer_positions, ra, dec, particle_state)
    quad = jnp.einsum("bi,bij,bj->b", xis_etas, inv_cov_matrices, xis_etas)
    ll = jnp.sum(-0.5 * (2 * jnp.log(2 * jnp.pi) + cov_log_dets + quad))
    return ll

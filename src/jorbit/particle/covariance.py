"""Shared forward-mode-AD covariance helpers for the Particle ephemeris paths."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from jorbit.astrometry.transformations import (
    elements_to_cartesian,
    horizons_ecliptic_to_icrs,
)
from jorbit.data.constants import TOTAL_SOLAR_SYSTEM_GM
from jorbit.utils.states import CartesianState, KeplerianState

# Squared conversion factor from radians^2 to arcsec^2.
_RAD2ARCSEC_SQ = (180.0 * 3600.0 / jnp.pi) ** 2


def _state_vec_to_xv(
    state_vec: jnp.ndarray, is_keplerian_param: bool
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Unpack a (6,) parameter vector to ICRS Cartesian position/velocity.

    If ``is_keplerian_param``, the entries are (semi, ecc, inc, Omega, omega, nu)
    and we convert via :func:`elements_to_cartesian` followed by an ecliptic ->
    ICRS rotation. Otherwise the entries are interpreted directly as flat
    Cartesian (x, y, z, vx, vy, vz) in ICRS.

    Returns ``(x, v)`` each shaped ``(1, 3)``.
    """
    if is_keplerian_param:
        x_ecl, v_ecl = elements_to_cartesian(
            state_vec[0:1],
            state_vec[1:2],
            state_vec[5:6],
            state_vec[2:3],
            state_vec[3:4],
            state_vec[4:5],
            TOTAL_SOLAR_SYSTEM_GM,
        )
        x = horizons_ecliptic_to_icrs(x_ecl)
        v = horizons_ecliptic_to_icrs(v_ecl)
    else:
        x = state_vec[:3].reshape(1, 3)
        v = state_vec[3:].reshape(1, 3)
    return x, v


def _state_to_vec(state: CartesianState | KeplerianState) -> jnp.ndarray:
    """Flatten a CartesianState or KeplerianState to a (6,) parameter vector."""
    if isinstance(state, KeplerianState):
        return jnp.concatenate(
            [
                jnp.atleast_1d(state.semi),
                jnp.atleast_1d(state.ecc),
                jnp.atleast_1d(state.inc),
                jnp.atleast_1d(state.Omega),
                jnp.atleast_1d(state.omega),
                jnp.atleast_1d(state.nu),
            ]
        )
    return jnp.concatenate([state.x.flatten(), state.v.flatten()])


def _cov_from_jacobian(
    radec_fn: Callable,
    nominal_vec: jnp.ndarray,
    cov: jnp.ndarray,
    N: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Linear error propagation via forward-mode AD.

    ``radec_fn`` must accept a ``(6,)`` parameter vector and return a flat
    ``(2N,)`` interleaved ``[ra0, dec0, ra1, dec1, ...]`` array (radians).
    Returns ``(ra, dec, cov_radec)`` where ``cov_radec`` has shape ``(N, 2, 2)``
    in ``arcsec**2``.
    """
    radec_nominal = radec_fn(nominal_vec)
    ras = radec_nominal[0::2]
    decs = radec_nominal[1::2]
    J = jax.jacfwd(radec_fn)(nominal_vec)
    J_t = J.reshape(N, 2, 6)
    cov_radec = jnp.einsum("nij,jk,nlk->nil", J_t, cov, J_t) * _RAD2ARCSEC_SQ
    return ras, decs, cov_radec

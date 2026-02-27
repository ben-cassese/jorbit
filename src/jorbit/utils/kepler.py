"""Functions for solving Kepler's equation.

This is a fork of squishyplanet/engine/kepler.py, which itself is a fork of
jaxoplanet/src/jaxoplanet/core/kepler.py, many thanks to the original authors
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.interpreters import ad

from jorbit.astrometry.transformations import (
    cartesian_to_elements,
    elements_to_cartesian,
)


@jax.jit
def kepler(M: float, ecc: float) -> float:
    """Solve Kepler's equation to compute the true anomaly.

    This implementation is based on that within `jaxoplanet <https://github.com/exoplanet-dev/jaxoplanet/>`_, many thanks to the authors.

    Args:
        M (Array [Radian]): Mean anomaly
        ecc (Array [Dimensionless]): Eccentricity

    Returns:
        Array: True anomaly in radians
    """
    sinf, cosf = _kepler(M, ecc)
    # this is the only bit that's different from jaxoplanet-
    # puts true anomalies into the range [0, 2*pi)
    f = jnp.arctan2(sinf, cosf)
    return jnp.where(f < 0, f + 2 * jnp.pi, f)


@jax.custom_jvp
def _kepler(M: float, ecc: float) -> tuple[float, float]:
    # Wrap into the right range
    M = M % (2 * jnp.pi)

    # We can restrict to the range [0, pi)
    high = jnp.pi < M
    M = jnp.where(high, 2 * jnp.pi - M, M)

    # Solve
    ome = 1 - ecc
    E = _starter(M, ecc, ome)
    E = _refine(M, ecc, ome, E)

    # Re-wrap back into the full range
    E = jnp.where(high, 2 * jnp.pi - E, E)

    # Convert to true anomaly; tan(0.5 * f)
    tan_half_f = jnp.sqrt((1 + ecc) / (1 - ecc)) * jnp.tan(0.5 * E)
    tan2_half_f = jnp.square(tan_half_f)

    # Then we compute sin(f) and cos(f) using:
    #  sin(f) = 2*tan(0.5*f)/(1 + tan(0.5*f)^2), and
    #  cos(f) = (1 - tan(0.5*f)^2)/(1 + tan(0.5*f)^2)
    denom = 1 / (1 + tan2_half_f)
    sinf = 2 * tan_half_f * denom
    cosf = (1 - tan2_half_f) * denom

    return sinf, cosf


@_kepler.defjvp
def _(primals: tuple, tangents: tuple) -> tuple[tuple, tuple]:
    M, e = primals
    M_dot, e_dot = tangents
    sinf, cosf = _kepler(M, e)

    # Pre-compute some things
    ecosf = e * cosf
    ome2 = 1 - e**2

    def make_zero(tan: float) -> float:
        if type(tan) is ad.Zero:
            return ad.zeros_like_aval(tan.aval)
        else:
            return tan

    # Propagate the derivatives
    f_dot = make_zero(M_dot) * (1 + ecosf) ** 2 / ome2**1.5
    f_dot += make_zero(e_dot) * (2 + ecosf) * sinf / ome2

    return (sinf, cosf), (cosf * f_dot, -sinf * f_dot)


def _starter(M: float, ecc: float, ome: float) -> float:
    M2 = jnp.square(M)
    alpha = 3 * jnp.pi / (jnp.pi - 6 / jnp.pi)
    alpha += 1.6 / (jnp.pi - 6 / jnp.pi) * (jnp.pi - M) / (1 + ecc)
    d = 3 * ome + alpha * ecc
    alphad = alpha * d
    r = (3 * alphad * (d - ome) + M2) * M
    q = 2 * alphad * ome - M2
    q2 = jnp.square(q)
    w = jnp.square(jnp.cbrt(jnp.abs(r) + jnp.sqrt(q2 * q + r * r)))
    return (2 * r * w / (jnp.square(w) + w * q + q2) + M) / d


def _refine(M: float, ecc: float, ome: float, E: float) -> float:
    sE = E - jnp.sin(E)
    cE = 1 - jnp.cos(E)

    f_0 = ecc * sE + E * ome - M
    f_1 = ecc * cE + ome
    f_2 = ecc * (E - sE)
    f_3 = 1 - f_1
    d_3 = -f_0 / (f_1 - 0.5 * f_0 * f_2 / f_1)
    d_4 = -f_0 / (f_1 + 0.5 * d_3 * f_2 + (d_3 * d_3) * f_3 / 6)
    d_42 = d_4 * d_4
    dE = -f_0 / (f_1 + 0.5 * d_4 * f_2 + d_4 * d_4 * f_3 / 6 - d_42 * d_4 * f_2 / 24)

    return E + dE


@jax.jit
def M_from_f(f: float, ecc: float) -> float:
    """Compute the mean anomaly from the true anomaly and eccentricity.

    Args:
        f (float):
            True anomaly in radians.
        ecc (float):
            Eccentricity.

    Returns:
        float:
            Mean anomaly in radians.
    """
    E = jnp.arctan2(jnp.sqrt(1 - ecc**2) * jnp.sin(f), ecc + jnp.cos(f))
    M = E - ecc * jnp.sin(E)
    return jnp.where(M < 0, M + 2 * jnp.pi, M)


@jax.jit
def keplerian_propagate(
    x0: jnp.ndarray,
    v0: jnp.ndarray,
    t0: float,
    times: jnp.ndarray,
    gm: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Propagate a Keplerian orbit to new times.

    Args:
        x0: Initial position (1, 3), ecliptic frame.
        v0: Initial velocity (1, 3), ecliptic frame.
        t0: Initial time (scalar, JD TDB).
        times: Target times (N,), JD TDB.
        gm: Gravitational parameter (GM) in AU^3/day^2.

    Returns:
        tuple: Positions (N, 3) and velocities (N, 3) in ecliptic frame.
    """
    a, ecc, nu, inc, Omega, omega = cartesian_to_elements(x0, v0, gm)

    nu_rad = nu[0] * jnp.pi / 180
    M0 = M_from_f(nu_rad, ecc[0])
    n = jnp.sqrt(gm / a[0] ** 3)

    M_new = M0 + n * (times - t0)
    nu_new_rad = jax.vmap(kepler, in_axes=(0, None))(M_new, ecc[0])
    nu_new_deg = nu_new_rad * 180 / jnp.pi

    a_arr = jnp.full_like(times, a[0])
    ecc_arr = jnp.full_like(times, ecc[0])
    inc_arr = jnp.full_like(times, inc[0])
    Omega_arr = jnp.full_like(times, Omega[0])
    omega_arr = jnp.full_like(times, omega[0])

    positions, velocities = elements_to_cartesian(
        a_arr, ecc_arr, nu_new_deg, inc_arr, Omega_arr, omega_arr, gm
    )
    return positions, velocities

"""Various tools for projecting positions onto the sky."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.data.constants import INV_SPEED_OF_LIGHT
from jorbit.utils.states import SystemState


@jax.jit
def sky_sep(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Compute the angular separation between two points on the sky.

    Following Astropy's SkyCoord.separation, this uses the Vincenty formula.

    Args:
        ra1 (float): Right ascension of the first position in radians.
        dec1 (float): Declination of the first position in radians.
        ra2 (float): Right ascension of the second position in radians.
        dec2 (float): Declination of the second position in radians.

    Returns:
        float: The angular separation in arcseconds.
    """
    # all inputs are floats, ICRS positions in radians
    # output is in arcsec

    # following the astropy source on .separation, using Vincenty
    delta = ra2 - ra1
    numerator = jnp.sqrt(
        (jnp.cos(dec2) * jnp.sin(delta)) ** 2
        + (
            jnp.cos(dec1) * jnp.sin(dec2)
            - jnp.sin(dec1) * jnp.cos(dec2) * jnp.cos(delta)
        )
        ** 2
    )

    denominator = jnp.sin(dec1) * jnp.sin(dec2) + jnp.cos(dec1) * jnp.cos(
        dec2
    ) * jnp.cos(delta)

    return jnp.arctan2(numerator, denominator) * 206264.80624709636


@jax.jit
def tangent_plane_projection(
    ra_ref: float, dec_ref: float, ra: float, dec: float
) -> jnp.ndarray:
    """Project a point on the sky onto a tangent plane at a reference point.

    Somewhat overkill, rotates the positions to the equator to avoid any potential
    issues near the poles.

    Args:
        ra_ref (float): Right ascension of the reference point in radians.
        dec_ref (float): Declination of the reference point in radians.
        ra (float): Right ascension of the point to project in radians.
        dec (float): Declination of the point to project in radians.

    Returns:
        jnp.ndarray: The projected coordinates in arcseconds.
    """
    # Convert to unit vectors
    cos_dec = jnp.cos(dec)
    sin_dec = jnp.sin(dec)
    cos_ra = jnp.cos(ra)
    sin_ra = jnp.sin(ra)

    # Initial cartesian coordinates
    x = cos_dec * cos_ra
    y = cos_dec * sin_ra
    z = sin_dec

    # Rotation matrices (combined into single operation)
    cos_ra_ref = jnp.cos(ra_ref)
    sin_ra_ref = jnp.sin(ra_ref)
    cos_dec_ref = jnp.cos(dec_ref)
    sin_dec_ref = jnp.sin(dec_ref)

    # Apply rotations (optimized matrix multiplication)
    x_rot = (x * cos_ra_ref + y * sin_ra_ref) * cos_dec_ref + z * sin_dec_ref
    y_rot = -x * sin_ra_ref + y * cos_ra_ref
    z_rot = -(x * cos_ra_ref + y * sin_ra_ref) * sin_dec_ref + z * cos_dec_ref

    # Project to plane
    xi = y_rot / x_rot
    eta = z_rot / x_rot

    return jnp.array([xi, eta]) * 206264.80624709636  # rad -> arcsec


@jax.jit
def on_sky(
    x: jnp.ndarray,
    v: jnp.ndarray,
    time: float,
    observer_position: jnp.ndarray,
    acc_func: jax.tree_util.Partial,
    acc_func_kwargs: dict = {},
    ltt_position_fn: jax.tree_util.Partial | None = None,
) -> tuple[float, float]:
    """Compute the on-sky position of a particle from a given observer position.

    This function computes the on-sky position of a particle at a given time, correcting
    for light travel time. By default it uses a 2nd-order Taylor expansion (position,
    velocity, acceleration) to propagate backward by the light travel time, where the
    acceleration is evaluated once at the observation time using ``acc_func``. Three
    iterations of the light travel time correction are applied, which is sufficient for
    most cases.

    For richer cases (e.g. a distant observer watching a close flyby, where higher-order
    terms in the polynomial expansion of position around the observation time matter),
    pass an explicit ``ltt_position_fn`` closure. When provided, this replaces
    both the on-the-fly acceleration evaluation and the constant-acceleration Taylor
    formula with a user-supplied propagator (typically built from IAS15 b-coefficients).

    Note: you can vmap this function, but don't pass multiple particles at once: each
    one needs its own light travel time correction.

    Args:
        x (jnp.ndarray): Position of the particle, shape (3,).
        v (jnp.ndarray): Velocity of the particle, shape (3,).
        time (float): Time at which to compute the on-sky position, JD, tdb.
        observer_position (jnp.ndarray): Position of the observer, shape (3,).
        acc_func (jax.tree_util.Partial):
            Acceleration function to use during light travel time correction. Must be a
            continuous function that can evaluate the positions of any fixed perturbers
            at arbitrary times. Ignored when ``ltt_position_fn`` is provided.
        acc_func_kwargs (dict, optional): Additional arguments for the acceleration
            function.
        ltt_position_fn (jax.tree_util.Partial | None, optional):
            Optional callable mapping a (negative) time offset ``dt`` to the particle's
            position at ``time + dt``. When provided, this is used inside the LTT
            iteration in place of the constant-acceleration Taylor expansion, and
            ``acc_func`` is not called. Default ``None`` preserves the original
            Taylor-based behavior.

    Returns:
        tuple[float, float]:
            The right ascension and declination of the particle in radians, ICRS.
    """
    if ltt_position_fn is None:
        # Default: evaluate acceleration once and Taylor-expand backward by LTT
        state = SystemState(
            massive_positions=jnp.empty((0, 3)),
            massive_velocities=jnp.empty((0, 3)),
            tracer_positions=jnp.array([x]),
            tracer_velocities=jnp.array([v]),
            log_gms=jnp.empty(0),
            time=time,
            fixed_perturber_positions=jnp.empty((0, 3)),
            fixed_perturber_velocities=jnp.empty((0, 3)),
            fixed_perturber_log_gms=jnp.empty(0),
            acceleration_func_kwargs=acc_func_kwargs,
        )
        a0 = acc_func(state)[0]  # shape (3,), acceleration of the single tracer

        def propagate(dt: jnp.ndarray) -> jnp.ndarray:
            return x + v * dt + 0.5 * a0 * dt * dt

    else:
        propagate = ltt_position_fn

    xz = x
    for _ in range(3):
        earth_distance = jnp.linalg.norm(xz - observer_position)
        dt = -earth_distance * INV_SPEED_OF_LIGHT
        xz = propagate(dt)

    X = xz - observer_position
    calc_ra = jnp.mod(jnp.arctan2(X[1], X[0]) + 2 * jnp.pi, 2 * jnp.pi)
    calc_dec = jnp.pi / 2 - jnp.arccos(X[-1] / jnp.linalg.norm(X))
    return calc_ra, calc_dec

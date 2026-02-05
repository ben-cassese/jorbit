"""Various tools for projecting positions onto the sky."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.data.constants import INV_SPEED_OF_LIGHT
from jorbit.integrators import ias15_evolve, initialize_ias15_integrator_state
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
    perturber_state_func: jax.tree_util.Partial,
    acc_func_kwargs: dict = {},
    perturber_x_coeffs: jnp.ndarray = None,
    perturber_v_coeffs: jnp.ndarray = None,
) -> tuple[float, float]:
    """Compute the on-sky position of a particle from a given observer position.

    This function computes the on-sky position of a particle at a given time, correcting
    for light travel time. It uses the IAS15 integrator and the provided acceleration
    function to evolve the particle's position and velocity as needed. There's a
    hard-coded three iteration limit to the light travel time correction, which is
    sufficient for most cases but may need to be adjusted for extreme scenarios.

    Note: you can vmap this function, but don't pass multiple particles at once: each
    one needs its own light travel time correction, and the IAS15 integrator needs to
    move a system of particles all to the same times.

    Args:
        x (jnp.ndarray): Position of the particle, shape (3,).
        v (jnp.ndarray): Velocity of the particle, shape (3,).
        time (float): Time at which to compute the on-sky position, JD, tdb.
        observer_position (jnp.ndarray): Position of the observer, shape (3,).
        acc_func (jax.tree_util.Partial): Acceleration function to use during light
            travel time correction
        perturber_state_func (jax.tree_util.Partial):
            Function to compute the position/velocity of any fixed perturbers.
        acc_func_kwargs (dict, optional): Additional arguments for the acceleration
            function.
        perturber_x_coeffs (jnp.ndarray, optional):
            Chebyshev coefficients for the perturber positions. Optional, only needed if
            the acceleration function requires fixed perturbers.
        perturber_v_coeffs (jnp.ndarray, optional):
            Chebyshev coefficients for the perturber velocities. Optional, only needed
            if the acceleration function requires fixed perturbers.

    Returns:
        tuple[float, float]:
            The right ascension and declination of the particle in radians, ICRS.
    """
    # has to be one particle at one time to get the light travel time right

    # get initial perturber positions/velocities via a dummy call w/ 0.1 day dt
    fixed_perturber_x0, fixed_perturber_v0, fixed_perturber_log_gms = (
        perturber_state_func(time, 0.1, perturber_x_coeffs, perturber_v_coeffs)
    )
    state = SystemState(
        massive_positions=jnp.empty((0, 3)),
        massive_velocities=jnp.empty((0, 3)),
        tracer_positions=jnp.array([x]),
        tracer_velocities=jnp.array([v]),
        log_gms=jnp.empty(0),
        time=time,
        fixed_perturber_positions=fixed_perturber_x0[0],  # only taking the first time
        fixed_perturber_velocities=fixed_perturber_v0[0],
        fixed_perturber_log_gms=fixed_perturber_log_gms,
        acceleration_func_kwargs=acc_func_kwargs,
    )
    a0 = acc_func(state)
    initial_integrator_state = initialize_ias15_integrator_state(a0)

    xz = state.tracer_positions[0]
    for _ in range(3):
        earth_distance = jnp.linalg.norm(xz - observer_position)
        light_travel_time = earth_distance * INV_SPEED_OF_LIGHT
        # dt = -light_travel_time

        # fixed_perturber_xs, fixed_perturber_vs, fixed_perturber_log_gms = (
        #     perturber_state_func(time, dt, perturber_x_coeffs, perturber_v_coeffs)
        # )

        _positions, _velocities, final_system_state, _final_integrator_state = (
            ias15_evolve(
                state,
                acc_func,
                jnp.array([state.time - light_travel_time]),
                initial_integrator_state,
            )
        )
        xz = final_system_state.tracer_positions[0]

    X = xz - observer_position
    calc_ra = jnp.mod(jnp.arctan2(X[1], X[0]) + 2 * jnp.pi, 2 * jnp.pi)
    calc_dec = jnp.pi / 2 - jnp.arccos(X[-1] / jnp.linalg.norm(X))
    return calc_ra, calc_dec

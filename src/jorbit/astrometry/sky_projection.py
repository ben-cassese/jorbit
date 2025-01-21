import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.data.constants import INV_SPEED_OF_LIGHT
from jorbit.integrators import ias15_evolve, initialize_ias15_integrator_state
from jorbit.utils.states import SystemState


@jax.jit
def sky_sep(ra1, dec1, ra2, dec2):
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
def tangent_plane_projection(ra_ref, dec_ref, ra, dec):
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
    x,
    v,
    time,
    observer_position,
    acc_func,
    acc_func_kwargs={},
):
    # has to be one particle at one time to get the light travel time right
    state = SystemState(
        massive_positions=jnp.empty((0, 3)),
        massive_velocities=jnp.empty((0, 3)),
        tracer_positions=jnp.array([x]),
        tracer_velocities=jnp.array([v]),
        log_gms=jnp.empty(0),
        time=time,
        acceleration_func_kwargs=acc_func_kwargs,
    )
    a0 = acc_func(state)
    initial_integrator_state = initialize_ias15_integrator_state(a0)

    def scan_func(carry, scan_over):
        xz = carry
        earth_distance = jnp.linalg.norm(xz - observer_position)
        light_travel_time = earth_distance * INV_SPEED_OF_LIGHT

        positions, velocities, final_system_state, final_integrator_state = (
            ias15_evolve(
                state,
                acc_func,
                jnp.array([state.time - light_travel_time]),
                initial_integrator_state,
            )
        )

        return final_system_state.tracer_positions[0], None

    xz, _ = jax.lax.scan(
        scan_func,
        state.tracer_positions[0],
        None,
        length=3,
    )

    X = xz - observer_position
    calc_ra = jnp.mod(jnp.arctan2(X[1], X[0]) + 2 * jnp.pi, 2 * jnp.pi)
    calc_dec = jnp.pi / 2 - jnp.arccos(X[-1] / jnp.linalg.norm(X))
    return calc_ra, calc_dec

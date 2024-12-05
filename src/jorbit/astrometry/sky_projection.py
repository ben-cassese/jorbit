import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


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
def on_sky(
    state: SystemState,
    acc_func,
    observer_position,
):
    a0 = acc_func(state)
    initial_integrator_state = initialize_ias15_integrator_state(a0)

    def scan_func(carry, scan_over):
        xz = carry
        earth_distance = jnp.linalg.norm(xz - observer_position)
        light_travel_time = earth_distance * INV_SPEED_OF_LIGHT

        positions, final_system_state, final_integrator_state = ias15_evolve(
            state,
            acc_func,
            jnp.array([state.time - light_travel_time]),
            initial_integrator_state,
            n_steps=5,
        )

        return final_system_state.positions, None

    xz = jax.lax.scan(scan_func, state.positions, None, length=2)[0]

    X = xz - observer_position
    calc_ra = jnp.mod(jnp.arctan2(X[:, 1], X[:, 0]) + 2 * jnp.pi, 2 * jnp.pi)
    calc_dec = jnp.pi / 2 - jnp.arccos(X[:, -1] / jnp.linalg.norm(X, axis=1))
    return calc_ra, calc_dec

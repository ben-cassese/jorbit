import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.data.constants import (
    ICRS_TO_HORIZONS_ECLIPTIC_ROT_MAT,
    HORIZONS_ECLIPTIC_TO_ICRS_ROT_MAT,
    TOTAL_SOLAR_SYSTEM_GM,
)


def icrs_to_horizons_ecliptic(xs):
    rotated_xs = jnp.dot(xs, ICRS_TO_HORIZONS_ECLIPTIC_ROT_MAT.T)
    return rotated_xs


def horizons_ecliptic_to_icrs(xs):
    rotated_xs = jnp.dot(xs, HORIZONS_ECLIPTIC_TO_ICRS_ROT_MAT.T)
    return rotated_xs


def elements_to_cartesian(a, ecc, nu, inc, Omega, omega):
    # # Each of the elements are (n_particles, )
    # # The angles are in *degrees*. Always assuming orbital element angles are in degrees

    nu *= jnp.pi / 180
    inc *= jnp.pi / 180
    Omega *= jnp.pi / 180
    omega *= jnp.pi / 180

    t = (a * (1 - ecc**2))[:, None]
    r_w = (
        t
        / (1 + ecc[:, None] * jnp.cos(nu[:, None]))
        * jnp.column_stack((jnp.cos(nu), jnp.sin(nu), nu * 0.0))
    )
    v_w = (
        jnp.sqrt(TOTAL_SOLAR_SYSTEM_GM)
        / jnp.sqrt(t)
        * jnp.column_stack((-jnp.sin(nu), ecc + jnp.cos(nu), nu * 0))
    )

    zeros = jnp.zeros_like(omega, dtype=jnp.float64)
    ones = jnp.ones_like(omega, dtype=jnp.float64)
    Rot1 = jnp.array(
        [
            [jnp.cos(-omega), -jnp.sin(-omega), zeros],
            [jnp.sin(-omega), jnp.cos(-omega), zeros],
            [zeros, zeros, ones],
        ]
    )

    Rot2 = jnp.array(
        [
            [ones, zeros, zeros],
            [zeros, jnp.cos(-inc), -jnp.sin(-inc)],
            [zeros, jnp.sin(-inc), jnp.cos(-inc)],
        ]
    )

    Rot3 = jnp.array(
        [
            [jnp.cos(-Omega), -jnp.sin(-Omega), zeros],
            [jnp.sin(-Omega), jnp.cos(-Omega), zeros],
            [zeros, zeros, ones],
        ]
    )

    rot = jax.vmap(
        lambda r1, r2, r3: jnp.matmul(jnp.matmul(r1, r2), r3), in_axes=(2, 2, 2)
    )(Rot1, Rot2, Rot3)

    x = jax.vmap(lambda x, y: jnp.matmul(x, y))(r_w, rot)
    v = jax.vmap(lambda x, y: jnp.matmul(x, y))(v_w, rot)

    return x, v


def cartesian_to_elements(x, v):
    r_mag = jnp.linalg.norm(x, axis=1)
    v_mag = jnp.linalg.norm(v, axis=1)

    # Specific angular momentum
    h = jnp.cross(x, v)
    h_mag = jnp.linalg.norm(h, axis=1)

    # Eccentricity vector
    e_vec = jnp.cross(v, h) / TOTAL_SOLAR_SYSTEM_GM - x / r_mag[:, jnp.newaxis]
    ecc = jnp.linalg.norm(e_vec, axis=1)

    # Specific orbital energy
    specific_energy = v_mag**2 / 2 - GM / r_mag

    a = -TOTAL_SOLAR_SYSTEM_GM / (2 * specific_energy)

    inc = jnp.arccos(h[:, 2] / h_mag) * 180 / jnp.pi

    n = jnp.cross(jnp.array([0, 0, 1]), h)
    n_mag = jnp.linalg.norm(n, axis=1)

    Omega = jnp.where(
        n[:, 1] >= 0,
        jnp.arccos(n[:, 0] / n_mag) * 180 / jnp.pi,
        360.0 - jnp.arccos(n[:, 0] / n_mag) * 180 / jnp.pi,
    )
    Omega = jnp.where(n_mag == 0, 0, Omega)

    omega = jnp.where(
        n_mag > 0,
        jnp.where(
            e_vec[:, 2] >= 0,
            jnp.arccos(
                jnp.clip(
                    jnp.sum(n * e_vec, axis=1)
                    / (n_mag * jnp.linalg.norm(e_vec, axis=1)),
                    -1,
                    1,
                )
            )
            * 180
            / jnp.pi,
            360
            - jnp.arccos(
                jnp.clip(
                    jnp.sum(n * e_vec, axis=1)
                    / (n_mag * jnp.linalg.norm(e_vec, axis=1)),
                    -1,
                    1,
                )
            )
            * 180
            / jnp.pi,
        ),
        0,
    )

    nu = jnp.where(
        jnp.sum(x * v, axis=1) >= 0,
        jnp.arccos(jnp.clip(jnp.sum(e_vec * x, axis=1) / (ecc * r_mag), -1, 1))
        * 180
        / jnp.pi,
        360
        - jnp.arccos(jnp.clip(jnp.sum(e_vec * x, axis=1) / (ecc * r_mag), -1, 1))
        * 180
        / jnp.pi,
    )

    return {"a": a, "ecc": ecc, "nu": nu, "inc": inc, "Omega": Omega, "omega": omega}

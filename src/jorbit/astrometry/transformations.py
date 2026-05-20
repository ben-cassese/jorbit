"""Transformations between coordinate systems or representations of a particle's state."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.data.constants import (
    HORIZONS_ECLIPTIC_TO_ICRS_ROT_MAT,
    ICRS_TO_HORIZONS_ECLIPTIC_ROT_MAT,
)


@jax.jit
def icrs_to_horizons_ecliptic(xs: jnp.ndarray) -> jnp.ndarray:
    """Transform an ICRS 3D cartesian position to a Horizons ecliptic 3D cartesian position.

    Args:
        xs (jnp.ndarray): ICRS 3D cartesian position.

    Returns:
        jnp.ndarray: Horizons ecliptic 3D cartesian position.
    """
    rotated_xs = jnp.dot(xs, ICRS_TO_HORIZONS_ECLIPTIC_ROT_MAT.T)
    return rotated_xs


@jax.jit
def horizons_ecliptic_to_icrs(xs: jnp.ndarray) -> jnp.ndarray:
    """Transform a Horizons ecliptic 3D cartesian position to an ICRS 3D cartesian position.

    Args:
        xs (jnp.ndarray): Horizons ecliptic 3D cartesian position.

    Returns:
        jnp.ndarray: ICRS 3D cartesian position.
    """
    rotated_xs = jnp.dot(xs, HORIZONS_ECLIPTIC_TO_ICRS_ROT_MAT.T)
    return rotated_xs


@jax.jit
def elements_to_cartesian(
    a: float, ecc: float, nu: float, inc: float, Omega: float, omega: float, mass: float
) -> tuple:
    """Convert orbital elements to cartesian coordinates.

    Relies on the total mass of the solar system, which is assumed to be the sum of all
    GM values of the sun, planets, and 16 most massive asteroids as assumed by DE440.

    This is the inverse of cartesian_to_elements.

    NOTE All orbital element angles are assumed to be *degrees*, in contrast to
    sky coordinate angles, which are usually assumed to be in radians when not in their
    Astropy SkyCoord form, or angular separations, which are usually assumed to be in
    arcsec.

    Args:
        a (float): Semi-major axis in AU.
        ecc (float): Eccentricity.
        nu (float): True anomaly in degrees.
        inc (float): Inclination in degrees.
        Omega (float): Longitude of the ascending node in degrees.
        omega (float): Argument of periapsis in degrees.
        mass (float): Total mass (GM) of the central object with G in AU^3 / day^2.

    Returns:
        tuple: (x, v) where x is the position in AU and v is the velocity in AU/day.
    """
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
        jnp.sqrt(mass)
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


@jax.jit
def cartesian_to_elements(x: jnp.ndarray, v: jnp.ndarray, mass: float) -> tuple:
    """Convert cartesian coordinates to orbital elements.

    Relies on the total mass of the solar system, which is assumed to be the sum of all
    GM values of the sun, planets, and 16 most massive asteroids as assumed by DE440.

    This is the inverse of elements_to_cartesian. Two degenerate cases are handled:
    - Circular orbits (ecc < 1e-10): omega is set to zero and nu becomes the argument
      of latitude (or position angle from x-axis for in-plane orbits).
    - In-plane orbits (inc = 0, n_mag = 0): Omega is set to zero; omega is set to the
      ecliptic longitude of periapsis (atan2(e_y, e_x)); nu for circular in-plane orbits
      is the position angle from the x-axis.

    Args:
        x (jnp.ndarray): Position in AU.
        v (jnp.ndarray): Velocity in AU/day.
        mass (float): Total mass (GM) of the central object with G in AU^3 / day^2.


    Returns:
        tuple:
            (a, ecc, nu, inc, Omega, omega) where a is the semi-major axis in AU,
            ecc is the eccentricity, nu is the true anomaly in degrees, inc is the
            inclination in degrees, Omega is the longitude of the ascending node in
            degrees, and omega is the argument of periapsis in degrees.
    """
    r_mag = jnp.linalg.norm(x, axis=1)
    v_mag = jnp.linalg.norm(v, axis=1)

    # Specific angular momentum
    h = jnp.cross(x, v)
    h_mag = jnp.linalg.norm(h, axis=1)

    # Eccentricity vector
    e_vec = jnp.cross(v, h) / mass - x / r_mag[:, jnp.newaxis]
    ecc = jnp.linalg.norm(e_vec, axis=1)

    # Specific orbital energy
    specific_energy = v_mag**2 / 2 - mass / r_mag

    a = -mass / (2 * specific_energy)

    inc = jnp.arccos(h[:, 2] / h_mag) * 180 / jnp.pi

    n = jnp.cross(jnp.array([0, 0, 1]), h)
    n_mag = jnp.linalg.norm(n, axis=1)
    # Prevents 0/0 NaN in jnp.where branches when inc=0 (n_mag=0)
    safe_n_mag = jnp.where(n_mag == 0, 1.0, n_mag)

    Omega = jnp.where(
        n[:, 1] >= 0,
        jnp.arccos(jnp.clip(n[:, 0] / safe_n_mag, -1, 1)) * 180 / jnp.pi,
        360.0 - jnp.arccos(jnp.clip(n[:, 0] / safe_n_mag, -1, 1)) * 180 / jnp.pi,
    )
    Omega = jnp.where(n_mag == 0, 0, Omega)

    # ── omega (argument of periapsis) ──
    # For in-plane orbits (n_mag=0), omega = ecliptic longitude of periapsis = atan2(e_y, e_x).
    # safe_e_mag guards ecc=0; the is_circular branch below overrides omega=0 in that case.
    safe_e_mag = jnp.where(ecc == 0, 1.0, ecc)
    omega_in_plane = jnp.where(
        e_vec[:, 1] >= 0,
        jnp.arccos(jnp.clip(e_vec[:, 0] / safe_e_mag, -1, 1)) * 180 / jnp.pi,
        360.0 - jnp.arccos(jnp.clip(e_vec[:, 0] / safe_e_mag, -1, 1)) * 180 / jnp.pi,
    )
    omega_standard = jnp.where(
        n_mag > 0,
        jnp.where(
            e_vec[:, 2] >= 0,
            jnp.arccos(
                jnp.clip(
                    jnp.sum(n * e_vec, axis=1)
                    / (safe_n_mag * jnp.linalg.norm(e_vec, axis=1)),
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
                    / (safe_n_mag * jnp.linalg.norm(e_vec, axis=1)),
                    -1,
                    1,
                )
            )
            * 180
            / jnp.pi,
        ),
        omega_in_plane,
    )

    # ── nu (true anomaly) ──
    # Standard computation from eccentricity vector
    nu_standard = jnp.where(
        jnp.sum(x * v, axis=1) >= 0,
        jnp.arccos(jnp.clip(jnp.sum(e_vec * x, axis=1) / (ecc * r_mag), -1, 1))
        * 180
        / jnp.pi,
        360
        - jnp.arccos(jnp.clip(jnp.sum(e_vec * x, axis=1) / (ecc * r_mag), -1, 1))
        * 180
        / jnp.pi,
    )

    # ── Circular orbit fallback: omega=0, nu=argument of latitude ──
    # u = angle from ascending node to position, measured in the orbital plane.
    # For in-plane orbits (n_mag=0), use the position angle from the x-axis instead.
    cos_u = jnp.sum(n * x, axis=1) / (safe_n_mag * r_mag)
    u_standard = jnp.where(
        x[:, 2] >= 0,
        jnp.arccos(jnp.clip(cos_u, -1, 1)) * 180 / jnp.pi,
        360 - jnp.arccos(jnp.clip(cos_u, -1, 1)) * 180 / jnp.pi,
    )
    nu_in_plane = jnp.where(
        x[:, 1] >= 0,
        jnp.arccos(jnp.clip(x[:, 0] / r_mag, -1, 1)) * 180 / jnp.pi,
        360 - jnp.arccos(jnp.clip(x[:, 0] / r_mag, -1, 1)) * 180 / jnp.pi,
    )
    u_deg = jnp.where(n_mag == 0, nu_in_plane, u_standard)

    is_circular = ecc < 1e-10  # Threshold for circular orbits, can be tuned
    omega = jnp.where(is_circular, 0.0, omega_standard)
    nu = jnp.where(is_circular, u_deg, nu_standard)
    ecc = jnp.where(is_circular, 0.0, ecc)

    return a, ecc, nu, inc, Omega, omega

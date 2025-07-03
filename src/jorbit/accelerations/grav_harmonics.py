"""Accelerations due to gravitational harmonics of a single extended body."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def grav_harmonics(
    body_gm: float,
    body_req: float,
    body_pos: jnp.ndarray,
    pole_ra: float,
    pole_dec: float,
    jns: jnp.ndarray,
    particle_xs: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the acceleration felt by each particle due to J distortion of one body.

    This computes Newtonian gravitational contributions from a single body's J2, J3, etc
    harmonics. Can handle arbitrarily many particles and J's, but is only for a single
    extended body (e.g. Earth, Sun). Makes no use of an Ephemeris, so can also be used
    for free particles in a system. Does not handle relativistic effects. Also does not
    self-consistently handle movement of the pole of the body, so the rotation axis
    remains fixed in the J2000 frame.

    Args:
        body_gm (float): The gravitational parameter of the body.
        body_req (float): The equatorial radius of the body.
        body_pos (jnp.ndarray): The position of the body's center of mass in 3D space,
            shape (3,), barycentric, equatorial coordinates.
        pole_ra (float): The right ascension of the pole in radians.
        pole_dec (float): The declination of the pole in radians.
        jns (jnp.ndarray): The spherical harmonic coefficients for the body, shape
            (N,), where N is the number of J's starting from J2.
        particle_xs (jnp.ndarray): The positions of the particles in 3D space, shape
            (P, 3), where P is the number of particles.

    Returns:
        jnp.ndarray:
            The 3D acceleration felt by each particle, shape (P, 3).
    """
    # center the particles on the body
    x = particle_xs - body_pos

    # rotate into the body's frame (which in this case will be no change, since we're
    # fixing the pole to the z-axis and pretending we're at the J200 epoch)
    sin_a = jnp.sin(pole_ra)
    cos_a = jnp.cos(pole_ra)
    sin_d = jnp.sin(pole_dec)
    cos_d = jnp.cos(pole_dec)
    rot_matrix = jnp.array(
        [
            [-sin_a, -cos_a * sin_d, cos_a * cos_d],
            [cos_a, -sin_a * sin_d, sin_a * cos_d],
            [0.0, cos_d, sin_d],
        ]
    )
    x = jnp.dot(x, rot_matrix)

    def _vn(x: jnp.ndarray) -> jnp.ndarray:
        # for a single particle, what is the potential for all Jn's?
        r = jnp.linalg.norm(x)
        theta = jnp.arccos(x[-1] / r)
        cos_theta = jnp.array([jnp.cos(theta)])
        tmp = jns.shape[0] + 1
        p = jax.scipy.special.lpmn_values(tmp, tmp, cos_theta, is_normalized=False)
        return (
            -(body_gm / r)
            * (body_req / r) ** jnp.arange(2, tmp + 1)
            * jns
            * p[0, 2:, 0]
        )

    # the acceleration is just the gradient of that, then vmapped over all particles
    accs = jax.vmap(lambda p: jnp.sum(jax.jacrev(_vn)(p), axis=0))(x)

    # rotate them back to the original frame
    accs = jnp.dot(accs, rot_matrix.T)

    return accs

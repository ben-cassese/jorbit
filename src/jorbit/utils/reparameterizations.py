import jax

jax.config.update("jax_enable_x64", True)

from functools import partial

import jax.numpy as jnp

from jorbit.utils.kepler import M_from_f, kepler


@jax.jit
def square_to_unit_disk(a, b):
    # https://doi.org/10.1080/10867651.1997.10487479
    a = 2 * a - 1
    b = 2 * b - 1

    flag1 = a > -b
    flag2 = a > b
    flag3 = a < b

    r = (
        (a * flag1 * flag2)
        + (b * flag1 * ~flag2)
        + (-a * ~flag1 * flag3)
        + (-b * ~flag1 * ~flag3)
    )
    phi = (
        (((jnp.pi / 4) * (b / a)) * flag1 * flag2)
        + ((jnp.pi / 4) * (2 - (a / b)) * flag1 * ~flag2)
        + ((jnp.pi / 4) * (4 + (b / a)) * ~flag1 * flag3)
        + ((jnp.pi / 4) * (6 - (a / b)) * ~flag1 * ~flag3)
    )

    return r, phi


@jax.jit
def unit_disk_to_square(r, phi):
    # inverse of square_to_unit_disk
    cond1 = (phi <= jnp.pi / 4) | (phi > 7 * jnp.pi / 4)
    cond2 = (phi > jnp.pi / 4) & (phi <= 3 * jnp.pi / 4)
    cond3 = (phi > 3 * jnp.pi / 4) & (phi <= 5 * jnp.pi / 4)
    cond4 = (phi > 5 * jnp.pi / 4) & (phi <= 7 * jnp.pi / 4)

    A = jnp.zeros_like(r)
    B = jnp.zeros_like(r)

    phi1 = jnp.where(phi > 7 * jnp.pi / 4, phi - 2 * jnp.pi, phi)
    A = jnp.where(cond1, r, A)
    B = jnp.where(cond1, (4 * r / jnp.pi) * phi1, B)

    A = jnp.where(cond2, r * (2 - (4 / jnp.pi) * phi), A)
    B = jnp.where(cond2, r, B)

    A = jnp.where(cond3, -r, A)
    B = jnp.where(cond3, r * (4 - (4 / jnp.pi) * phi), B)

    A = jnp.where(cond4, r * ((4 / jnp.pi) * phi - 6), A)
    B = jnp.where(cond4, -r, B)

    a = (A + 1) / 2
    b = (B + 1) / 2

    return a, b


@partial(jax.jit, static_argnums=(3))
def unit_cube_to_orbital_elements(u, a_low, a_high, uniform_inc):

    _r, _theta = square_to_unit_disk(u[0], u[1])
    _r = _r**2  # this gives us uniform e
    h = _r * jnp.cos(_theta)
    k = _r * jnp.sin(_theta)
    e = _r
    omega = jnp.arctan2(h, k) + jnp.pi

    _r, _theta = square_to_unit_disk(u[2], u[3])
    if uniform_inc:
        _r = jnp.sin(jnp.pi * _r**2 / 2)  # this gives us uniform i
    p = _r * jnp.cos(_theta)
    q = _r * jnp.sin(_theta)
    i = 2 * jnp.arcsin(_r)
    Omega = jnp.arctan2(q, p) + jnp.pi

    a = u[4] * (a_high - a_low) + a_low

    _r, _theta = square_to_unit_disk(u[5], u[6])
    lamb = _theta
    lamb = jnp.where(lamb < 0, lamb + 2 * jnp.pi, lamb)
    M = lamb - omega - Omega
    M = jnp.where(M < 0, M + 2 * jnp.pi, M)
    f = kepler(M, e)

    return jnp.array([a, e, i, Omega, omega, f, _r])


@partial(jax.jit, static_argnums=(3))
def orbital_elements_to_unit_cube(orb, a_low, a_high, uniform_inc):
    a, e, i, Omega, omega, f, r3 = orb

    theta1 = 3 * jnp.pi / 2 - omega
    theta1 = jnp.where(theta1 < 0, theta1 + 2 * jnp.pi, theta1)
    r1 = jnp.sqrt(e)
    u0, u1 = unit_disk_to_square(r1, theta1)

    r2 = jnp.sin(i / 2)
    if uniform_inc:
        r2 = jnp.sqrt(2 / jnp.pi) * jnp.sqrt(jnp.arcsin(r2))
    theta2 = Omega - jnp.pi
    theta2 = jnp.where(theta2 < 0, theta2 + 2 * jnp.pi, theta2)
    u2, u3 = unit_disk_to_square(r2, theta2)

    u4 = (a - a_low) / (a_high - a_low)

    M = M_from_f(f, e)  # This function must be provided.
    lamb = M + omega + Omega
    lamb = jnp.where(lamb < 0, lamb + 2 * jnp.pi, lamb)
    lamb = jnp.mod(lamb, 2 * jnp.pi)
    u5, u6 = unit_disk_to_square(r3, lamb)

    return jnp.array([u0, u1, u2, u3, u4, u5, u6])

# These are pythonized/jaxified versions of acceleration models within REBOUNDx,
# Tamayo et al. (2020) (DOI: 10.1093/mnras/stz2870). The gr_full function is the
# equivalent of rebx_calculate_gr_full in REBOUNDx, which is itself based on
# Newhall et al. (1984) (bibcode: 1983A&A...125..150N)
# The original code is available at https://github.com/dtamayo/reboundx/blob/502abf3066d9bae174cb20538294c916e73391cd/src/gr_full.c
# Accessed Fall 2024.

# Many thanks to the REBOUNDx developers for their work, and for making it open source!

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from functools import partial

from jorbit.data.constants import SPEED_OF_LIGHT


# equivalent of rebx_calculate_gr_full in reboundx
@partial(jax.jit, static_argnames=["max_iterations"])
def gr_full(
    x: jnp.ndarray,  # positions (N,3)
    v: jnp.ndarray,  # velocities (N,3)
    gms: jnp.ndarray,  # masses (N,)
    max_iterations: int = 10,
    c2: float = SPEED_OF_LIGHT**2,
) -> jnp.ndarray:

    N = x.shape[0]

    # Calculate pairwise differences
    dx = x[:, None, :] - x[None, :, :]  # (N,N,3)
    dv = v[:, None, :] - v[None, :, :]  # (N,N,3)
    r2 = jnp.sum(dx * dx, axis=-1)  # (N,N)
    r = jnp.sqrt(r2)  # (N,N)
    r3 = r2 * r  # (N,N)

    # Mask for i!=j calculations
    mask = ~jnp.eye(N, dtype=bool)  # (N,N)

    # Compute initial Newtonian accelerations
    prefac = 1.0 / r3
    prefac = jnp.where(mask, prefac, 0.0)
    a_newt = -jnp.sum(prefac[:, :, None] * dx * gms[None, :, None], axis=1)  # (N,3)

    # Move to barycentric frame
    x_com = jnp.sum(x * gms[:, None], axis=0) / jnp.sum(gms)
    v_com = jnp.sum(v * gms[:, None], axis=0) / jnp.sum(gms)
    x = x - x_com
    v = v - v_com
    v2 = jnp.sum(v * v, axis=-1)
    vdot = jnp.sum(v[:, None, :] * v[None, :, :], axis=-1)

    # Compute constant acceleration terms
    # first part of the constant term
    a1_arr = jnp.sum((4.0 / c2) * gms / r, axis=1, where=mask)
    a1_arr = jnp.broadcast_to(a1_arr, (N, N)).T

    a2_arr = jnp.sum((1.0 / c2) * gms / r, axis=1, where=mask)
    a2_arr = jnp.broadcast_to(a2_arr, (N, N))

    a3_arr = jnp.broadcast_to(-v2 / c2, (N, N)).T

    a4_arr = -2.0 * jnp.broadcast_to(v2, (N, N)) / c2

    a5_arr = (4.0 / c2) * vdot

    a6_0 = jnp.sum(dx * v[None, :, :], axis=-1)
    a6_arr = (3.0 / (2 * c2)) * (a6_0**2) / r2

    a7_arr = jnp.sum(dx * a_newt[None, :, :], axis=-1) / (2 * c2)

    factor1_arr = a1_arr + a2_arr + a3_arr + a4_arr + a5_arr + a6_arr + a7_arr
    part1_arr = (
        jnp.broadcast_to(gms, (N, N))[:, :, None]
        * dx
        * factor1_arr[:, :, None]
        / r3[:, :, None]
    )

    # second part of the constant term
    factor2_arr = jnp.sum(dx * (4 * v[:, None, :] - 3 * v[None, :, :]), axis=-1)
    part2_arr = (
        jnp.broadcast_to(gms, (N, N))[:, :, None]
        * (
            (
                factor2_arr[:, :, None] * dv / r3[:, :, None]
                + 7 / 2 * a_newt[None, :, :] / r[:, :, None]
            )
        )
        / c2
    )

    a_const_arr = part1_arr + part2_arr
    a_const = jnp.sum(a_const_arr, axis=1, where=mask[:, :, None])

    def iteration_step(a_curr):
        rdota = jnp.sum(dx * a_curr[None, :, :], axis=-1)  # (N,N)
        non_const = jnp.sum(
            (gms[None, :, None] / (2.0 * c2))
            * (
                (dx * rdota[:, :, None] / r3[:, :, None])
                + (7.0 * a_curr[None, :, :] / r[:, :, None])
            ),
            axis=1,
            where=mask[:, :, None],
        )
        return a_const + non_const

    def do_nothing(carry):
        return carry

    def do_iteration(carry):
        a_prev, a_curr, _ = carry
        a_next = iteration_step(a_curr)
        ratio = jnp.max(jnp.abs((a_next - a_curr) / a_next))
        return (a_curr, a_next, ratio)

    def body_fn(carry, _):
        a_prev, a_curr, ratio = carry
        should_continue = ratio > 2.220446049250313e-16
        new_carry = jax.lax.cond(should_continue, do_iteration, do_nothing, carry)
        return new_carry, None

    # Initialize with constant terms
    init_a = jnp.zeros_like(a_const)
    init_carry = (init_a, a_const, 1.0)

    # Run fixed number of iterations using scan
    final_carry, _ = jax.lax.scan(body_fn, init_carry, None, length=max_iterations)

    # Extract final acceleration
    _, a_final, _ = final_carry

    return a_newt + a_final

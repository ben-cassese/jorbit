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

from jorbit.data.constants import SPEED_OF_LIGHT


# def _constant_acceleration_terms(
#     x: jnp.ndarray,  # positions (N,3)
#     v: jnp.ndarray,  # velocities (N,3)
#     gms: jnp.ndarray,  # masses (N,)):
# ):
#     N = x.shape[0]
#     C2 = SPEED_OF_LIGHT ** 2
#     C2 = 10000.0

#     # Calculate pairwise differences
#     dx = x[:, None, :] - x[None, :, :]  # (N,N,3)
#     r2 = jnp.sum(dx**2, axis=-1)  # (N,N)
#     r = jnp.sqrt(r2)  # (N,N)
#     r3 = r2 * r  # (N,N)

#     # Mask for i!=j calculations
#     mask = ~jnp.eye(N, dtype=bool)  # (N,N)

#     # Compute initial Newtonian accelerations
#     prefac = 1 / r3
#     prefac = jnp.where(mask, prefac, 0.0)
#     a_newt = -jnp.sum(prefac[:, :, None] * dx * gms[None, :, None], axis=1)  # (N,3)

#     # Move to barycentric frame
#     x_com = jnp.sum(x * gms[:, None], axis=0) / jnp.sum(gms)
#     v_com = jnp.sum(v * gms[:, None], axis=0) / jnp.sum(gms)
#     x = x - x_com
#     v = v - v_com

#     # Compute constant acceleration terms
#     v2 = jnp.sum(v**2, axis=1)  # (N,)
#     vdotv = jnp.dot(v, v.T)  # (N,N)
#     dv = v[:, None, :] - v[None, :, :]  # (N,N,3)
#     rdotv = jnp.sum(dx * v[None, :, :], axis=-1)  # (N,N)

#     # First constant part calculations
#     a1 = (4.0 / C2) * jnp.sum(gms[None, :] / r, axis=1, where=mask)  # (N,)
#     a2 = (1.0 / C2) * jnp.sum(gms[None, :] / r, axis=1, where=mask)  # (N,)
#     a3 = -v2 / C2  # (N,)
#     a4 = -2 * v2[None, :] / C2  # (N,N)
#     a5 = (4 / C2) * vdotv  # (N,N)
#     a6 = (3 / (2 * C2)) * (rdotv**2 / r2)  # (N,N)
#     a7 = jnp.sum(dx * a_newt[None, :, :], axis=-1) / (2 * C2)  # (N,N)

#     # Combine all factors
#     factor1 = (a1 + a2 + a3)[:, None] + jnp.where(mask, a4 + a5 + a6 + a7, 0.0)  # (N,N)

#     # Calculate first part of a_const
#     a_const = jnp.sum(
#         gms[None, :, None] * dx * (factor1[:, :, None]) / r3[:, :, None],
#         axis=1,
#         where=mask[:, :, None],
#     )  # (N,3)

#     # Second constant part
#     factor2 = jnp.sum(dx * (4 * v[:, None, :] - 3 * v[None, :, :]), axis=-1)  # (N,N)

#     # Add second part to a_const
#     a_const += jnp.sum(
#         (gms[None, :, None] / C2)
#         * (
#             (factor2[:, :, None] * dv / r3[:, :, None])
#             + (7 / 2 * a_newt[None, :, :] / r[:, :, None])
#         ),
#         axis=1,
#         where=mask[:, :, None],
#     )

#     return dx, r, r3, mask, a_newt, a_const


# equivalent of rebx_calculate_gr_full in reboundx
def gr_full(
    x: jnp.ndarray,  # positions (N,3)
    v: jnp.ndarray,  # velocities (N,3)
    gms: jnp.ndarray,  # masses (N,)
    max_iterations: int = 10,
) -> jnp.ndarray:

    # C2 = SPEED_OF_LIGHT ** 2
    # C2 = 100.0
    # dx, r, r3, mask, a_newt, a_const = _constant_acceleration_terms(x, v, gms)

    N = x.shape[0]
    C2 = SPEED_OF_LIGHT**2
    C2 = 100.0

    # Calculate pairwise differences
    dx = x[:, None, :] - x[None, :, :]  # (N,N,3)
    r2 = jnp.sum(dx**2, axis=-1)  # (N,N)
    r = jnp.sqrt(r2)  # (N,N)
    r3 = r2 * r  # (N,N)

    # Mask for i!=j calculations
    mask = ~jnp.eye(N, dtype=bool)  # (N,N)

    # Compute initial Newtonian accelerations
    prefac = 1 / r3
    prefac = jnp.where(mask, prefac, 0.0)
    a_newt = -jnp.sum(prefac[:, :, None] * dx * gms[None, :, None], axis=1)  # (N,3)

    # Move to barycentric frame
    x_com = jnp.sum(x * gms[:, None], axis=0) / jnp.sum(gms)
    v_com = jnp.sum(v * gms[:, None], axis=0) / jnp.sum(gms)
    x = x - x_com
    v = v - v_com

    # Compute constant acceleration terms
    v2 = jnp.sum(v**2, axis=1)  # (N,)
    vdotv = jnp.dot(v, v.T)  # (N,N)
    dv = v[:, None, :] - v[None, :, :]  # (N,N,3)
    rdotv = jnp.sum(dx * v[None, :, :], axis=-1)  # (N,N)

    # First constant part calculations
    a1 = (4.0 / C2) * jnp.sum(gms[None, :] / r, axis=1, where=mask)  # (N,)
    a2 = (1.0 / C2) * jnp.sum(gms[None, :] / r, axis=1, where=mask)  # (N,)
    a3 = -v2 / C2  # (N,)
    a4 = -2.0 * v2[None, :] / C2  # (N,N)
    a5 = (4.0 / C2) * vdotv  # (N,N)
    a6 = (3.0 / (2.0 * C2)) * (rdotv**2 / r2)  # (N,N)
    a7 = jnp.sum(dx * a_newt[None, :, :], axis=-1) / (2 * C2)  # (N,N)

    # Combine all factors
    factor1 = (a1 + a2 + a3)[:, None] + jnp.where(mask, a4 + a5 + a6 + a7, 0.0)  # (N,N)

    # Calculate first part of a_const
    a_const = jnp.sum(
        gms[None, :, None] * dx * (factor1[:, :, None]) / r3[:, :, None],
        axis=1,
        where=mask[:, :, None],
    )  # (N,3)

    # Second constant part
    factor2 = jnp.sum(dx * (4 * v[:, None, :] - 3 * v[None, :, :]), axis=-1)  # (N,N)

    # Add second part to a_const
    # a_const *= 0.0
    a_const += jnp.sum(
        (gms[None, :, None] / C2)
        * (
            (factor2[:, :, None] * dv / r3[:, :, None])
            + (7 / 2 * a_newt[None, :, :] / r[:, :, None])
        ),
        axis=1,
        where=mask[:, :, None],
    )

    def iteration_step(a_curr):
        rdota = jnp.sum(dx * a_curr[None, :, :], axis=-1)  # (N,N)
        non_const = jnp.sum(
            (gms[None, :, None] / (2 * C2))
            * (
                (dx * rdota[:, :, None] / r3[:, :, None])
                + (7 * a_curr[None, :, :] / r[:, :, None])
            ),
            axis=1,
            where=mask[:, :, None],
        )
        return a_const + non_const

    def do_nothing(carry):
        return carry

    def do_iteration(carry):
        jax.debug.print("doing iteration")
        a_prev, a_curr, _ = carry
        a_next = iteration_step(a_curr)
        ratio = jnp.max(jnp.abs((a_next - a_curr) / a_next))
        return (a_curr, a_next, ratio)

    def body_fn(carry, _):
        a_prev, a_curr, ratio = carry

        # Use cond to either continue iteration or return current state
        should_continue = ratio > jnp.finfo(jnp.float64).eps
        new_carry = jax.lax.cond(should_continue, do_iteration, do_nothing, carry)

        return new_carry, None

    # Initialize with constant terms
    init_a = jnp.zeros_like(a_const)
    init_carry = (init_a, a_const, 1.0)

    # Run fixed number of iterations using scan
    final_carry, _ = jax.lax.scan(body_fn, init_carry, None, length=max_iterations)

    # Extract final acceleration
    _, a_final, _ = final_carry

    return a_newt + a_final, a_newt


# def gr_fixed_perturber(
#     particle_x,
#     particle_v,
#     perturbers_x,
#     perturbers_v,
#     perturbers_gms,
# ):
#     pass

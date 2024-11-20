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


# equivalent of  in reboundx
def gr_full2(
    x: jnp.ndarray,  # positions (N,3)
    v: jnp.ndarray,  # velocities (N,3)
    a0: jnp.ndarray,  # initial accelerations (N,3)
    m: jnp.ndarray,  # masses (N,)
    C2: float,
    G: float,
    max_iterations: int = 10,
) -> jnp.ndarray:
    N = x.shape[0]
    # Calculate pairwise differences
    dx = x[:, None, :] - x[None, :, :]  # (N,N,3)
    r2 = jnp.sum(dx**2, axis=-1)  # (N,N)
    r = jnp.sqrt(r2)  # (N,N)
    r3 = r2 * r  # (N,N)

    # Mask for i!=j calculations
    mask = ~jnp.eye(N, dtype=bool)  # (N,N)

    # Compute initial Newtonian accelerations
    prefac = G / r3
    prefac = jnp.where(mask, prefac, 0.0)
    a_newt = -jnp.sum(prefac[:, :, None] * dx * m[None, :, None], axis=1)  # (N,3)

    # Move to barycentric frame
    x_com = jnp.sum(x * m[:, None], axis=0) / jnp.sum(m)
    v_com = jnp.sum(v * m[:, None], axis=0) / jnp.sum(m)
    x = x - x_com
    v = v - v_com

    # Compute constant acceleration terms
    v2 = jnp.sum(v**2, axis=1)  # (N,)
    vdotv = jnp.dot(v, v.T)  # (N,N)
    dv = v[:, None, :] - v[None, :, :]  # (N,N,3)
    rdotv = jnp.sum(dx * v[None, :, :], axis=-1)  # (N,N)

    # First constant part calculations
    a1 = (4.0 / C2) * G * jnp.sum(m[None, :] / r, axis=1, where=mask)  # (N,)
    a2 = (1.0 / C2) * G * jnp.sum(m[None, :] / r, axis=1, where=mask)  # (N,)
    a3 = -v2 / C2  # (N,)
    a4 = -2 * v2[None, :] / C2  # (N,N)
    a5 = (4 / C2) * vdotv  # (N,N)
    a6 = (3 / (2 * C2)) * (rdotv**2 / r2)  # (N,N)
    a7 = jnp.sum(dx * a0[None, :, :], axis=-1) / (2 * C2)  # (N,N)

    # Combine all factors
    factor1 = (a1 + a2 + a3)[:, None] + jnp.where(mask, a4 + a5 + a6 + a7, 0.0)  # (N,N)

    # Calculate first part of a_const
    a_const = jnp.sum(
        G * m[None, :, None] * dx * (factor1[:, :, None]) / r3[:, :, None],
        axis=1,
        where=mask[:, :, None],
    )  # (N,3)

    # Second constant part
    factor2 = jnp.sum(dx * (4 * v[:, None, :] - 3 * v[None, :, :]), axis=-1)  # (N,N)

    # Add second part to a_const
    a_const += jnp.sum(
        (G * m[None, :, None] / C2)
        * (
            (factor2[:, :, None] * dv / r3[:, :, None])
            + (7 / 2 * a0[None, :, :] / r[:, :, None])
        ),
        axis=1,
        where=mask[:, :, None],
    )

    def iteration_step(a_curr):
        rdota = jnp.sum(dx * a_curr[None, :, :], axis=-1)  # (N,N)
        non_const = jnp.sum(
            (G * m[None, :, None] / (2 * C2))
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

    return a_newt + a_final


# def gr_full(
#     x: jnp.ndarray,  # positions (N,3)
#     v: jnp.ndarray,  # velocities (N,3)
#     a0: jnp.ndarray,  # initial accelerations (N,3)
#     m: jnp.ndarray,  # masses (N,)
#     C2: float,
#     G: float,
#     max_iterations: int = 10,
# ) -> jnp.ndarray:
#     N = x.shape[0]

#     # Calculate pairwise differences
#     dx = x[:, None, :] - x[None, :, :]  # (N,N,3)
#     r2 = jnp.sum(dx**2, axis=-1)  # (N,N)
#     r = jnp.sqrt(r2)  # (N,N)
#     r3 = r2 * r  # (N,N)

#     # Mask for i!=j calculations
#     mask = ~jnp.eye(N, dtype=bool)  # (N,N)

#     # Compute initial Newtonian accelerations (if not provided)
#     prefac = G / r3
#     prefac = jnp.where(mask, prefac, 0.0)

#     a_newt = -jnp.sum(prefac[:, :, None] * dx * m[None, :, None], axis=1)  # (N,3)

#     # Move to barycentric frame (simplified - you might need to adjust this)
#     x_com = jnp.sum(x * m[:, None], axis=0) / jnp.sum(m)
#     v_com = jnp.sum(v * m[:, None], axis=0) / jnp.sum(m)
#     x = x - x_com
#     v = v - v_com

#     # Compute constant acceleration terms
#     v2 = jnp.sum(v**2, axis=1)  # (N,)
#     vdotv = jnp.dot(v, v.T)  # (N,N)

#     # Compute r_ij dot v_j
#     dv = v[:, None, :] - v[None, :, :]  # (N,N,3)
#     rdotv = jnp.sum(dx * v[None, :, :], axis=-1)  # (N,N)

#     # First constant part calculations
#     a1 = (4.0 / C2) * G * jnp.sum(m[None, :] / r, axis=1, where=mask)  # (N,)
#     a2 = (1.0 / C2) * G * jnp.sum(m[None, :] / r, axis=1, where=mask)  # (N,)
#     a3 = -v2 / C2  # (N,)
#     a4 = -2 * v2[None, :] / C2  # (N,N)
#     a5 = (4 / C2) * vdotv  # (N,N)
#     a6 = (3 / (2 * C2)) * (rdotv**2 / r2)  # (N,N)
#     a7 = jnp.sum(dx * a0[None, :, :], axis=-1) / (2 * C2)  # (N,N)

#     # Combine all factors with proper broadcasting
#     factor1 = (a1 + a2 + a3)[:, None] + jnp.where(mask, a4 + a5 + a6 + a7, 0.0)  # (N,N)

#     # Calculate first part of a_const with proper broadcasting
#     a_const = jnp.sum(
#         G * m[None, :, None] * dx * (factor1[:, :, None]) / r3[:, :, None],
#         axis=1,
#         where=mask[:, :, None],
#     )  # (N,3)

#     # Second constant part
#     factor2 = jnp.sum(dx * (4 * v[:, None, :] - 3 * v[None, :, :]), axis=-1)  # (N,N)

#     # Add second part to a_const with proper broadcasting
#     a_const += jnp.sum(
#         (G * m[None, :, None] / C2)
#         * (
#             (factor2[:, :, None] * dv / r3[:, :, None])
#             + (7 / 2 * a0[None, :, :] / r[:, :, None])
#         ),
#         axis=1,
#         where=mask[:, :, None],
#     )

#     # Initialize acceleration with constant terms
#     a = a_const

#     # Iterative refinement
#     def iteration_step(i, a):
#         rdota = jnp.sum(dx * a[None, :, :], axis=-1)  # (N,N)

#         non_const = jnp.sum(
#             (G * m[None, :, None] / (2 * C2))
#             * (
#                 (dx * rdota[:, :, None] / r3[:, :, None])
#                 + (7 * a[None, :, :] / r[:, :, None])
#             ),
#             axis=1,
#             where=mask[:, :, None],
#         )

#         return a_const + non_const

#     # Run iterations
#     def cond_fn(state):
#         i, a_prev, a_curr, ratio = state
#         rel_diff = jnp.where(ratio > jnp.finfo(jnp.float64).eps, ratio, 0.0)
#         return (i < max_iterations) & (ratio > jnp.finfo(jnp.float64).eps)

#     def body_fn(state):
#         i, _, a_curr, _ = state
#         a_next = iteration_step(i, a_curr)
#         return i + 1, a_curr, a_next, jnp.max(jnp.abs((a_next - a_curr) / a_next))

#     _, _, a_final, max_dev = lax.while_loop(
#         cond_fn, body_fn, (0, jnp.zeros_like(a), a, 1.0)
#     )

#     return a_newt + a_final

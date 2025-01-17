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

from jorbit.utils.states import SystemState
from jorbit.data.constants import SPEED_OF_LIGHT


# equivalent of rebx_calculate_gr_full in reboundx
@partial(jax.jit, static_argnames=["max_iterations"])
def ppn_gravity(
    inputs: SystemState,
    max_iterations: int = 10,
) -> jnp.ndarray:

    c2 = inputs.acceleration_func_kwargs.get("c2", SPEED_OF_LIGHT**2)

    # surrendering on the efficient handling of tracers vs. massive particles-
    # lots of unnecessary computation here if T > 0, but ah well for now
    M = inputs.massive_positions.shape[0]
    T = inputs.tracer_positions.shape[0]
    N = M + T
    positions = jnp.concatenate(
        [inputs.massive_positions, inputs.tracer_positions], axis=0
    )
    velocities = jnp.concatenate(
        [inputs.massive_velocities, inputs.tracer_velocities], axis=0
    )
    gms = jnp.concatenate([jnp.exp(inputs.log_gms), jnp.zeros(T)])

    dx = positions[:, None, :] - positions[None, :, :]
    r2 = jnp.sum(dx * dx, axis=-1)
    r = jnp.sqrt(r2)
    r3 = r2 * jnp.sqrt(r2)

    # Mask for i!=j calculations
    mask = ~jnp.eye(N, dtype=bool)
    prefac = jnp.where(mask, 1.0 / r3, 0.0)

    # Newtonian acceleration
    a_newt = -jnp.sum(
        prefac[:, :, None] * dx * gms[None, :, None],
        axis=1,
    )  # (N,3)

    dv = velocities[:, None, :] - velocities[None, :, :]  # (N,N,3)

    x_com = jnp.sum(positions * gms[:, None], axis=0) / jnp.sum(gms)
    v_com = jnp.sum(velocities * gms[:, None], axis=0) / jnp.sum(gms)

    positions = positions - x_com
    velocities = velocities - v_com

    # Compute velocity-dependent terms
    v2 = jnp.sum(velocities * velocities, axis=-1)  # (N,)
    vdot = jnp.sum(velocities[:, None, :] * velocities[None, :, :], axis=-1)  # (N,N)

    a1 = jnp.sum((4.0 / c2) * gms / r, axis=1, where=mask)
    a1 = jnp.broadcast_to(a1, (N, N)).T

    a2 = jnp.sum((1.0 / c2) * gms / r, axis=1, where=mask)
    a2 = jnp.broadcast_to(a2, (N, N))

    a3 = jnp.broadcast_to(-v2 / c2, (N, N)).T
    a4 = -2.0 * jnp.broadcast_to(v2, (N, N)) / c2
    a5 = (4.0 / c2) * vdot

    a6_0 = jnp.sum(dx * velocities[None, :, :], axis=-1)
    a6 = (3.0 / (2 * c2)) * (a6_0**2) / r2

    a7 = jnp.sum(dx * a_newt[None, :, :], axis=-1) / (2 * c2)

    factor1 = a1 + a2 + a3 + a4 + a5 + a6 + a7
    part1 = (
        jnp.broadcast_to(gms, (N, N))[:, :, None]
        * dx
        * factor1[:, :, None]
        / r3[:, :, None]
    )

    factor2 = jnp.sum(
        dx * (4 * velocities[:, None, :] - 3 * velocities[None, :, :]), axis=-1
    )
    part2 = (
        jnp.broadcast_to(gms, (N, N))[:, :, None]
        * (
            factor2[:, :, None] * dv / r3[:, :, None]
            + 7.0 / 2.0 * a_newt[None, :, :] / r[:, :, None]
        )
        / c2
    )

    a_const = jnp.sum(part1 + part2, axis=1, where=mask[:, :, None])

    def iteration_step(a_curr):
        rdota = jnp.sum(dx * a_curr[None, :, :], axis=-1)  # (N, N)
        non_const = jnp.sum(
            (gms[None, :, None] / (2.0 * c2))
            * (
                (dx * rdota[:, :, None] / r3[:, :, None])
                + (7.0 * a_curr[None, :, :] / r[:, :, None])
            ),
            axis=1,
            where=mask[:, :, None],
        )

        return non_const

    def do_nothing(carry):
        return carry

    def do_iteration(carry):
        a_prev, a_curr, _ = carry
        non_const = iteration_step(a_curr)
        a_next = a_const + non_const

        ratio = jnp.max(jnp.abs((a_next - a_curr) / a_next), initial=0.0)

        return (a_curr, a_next, ratio)

    def body_fn(carry, _):
        a_prev, a_curr, ratio = carry
        should_continue = ratio > jnp.finfo(jnp.float64).eps
        new_carry = jax.lax.cond(should_continue, do_iteration, do_nothing, carry)
        return new_carry, None

    # Initialize with constant terms
    init_a = jnp.zeros_like(a_const)
    init_carry = (init_a, a_const, 1.0)

    # Run fixed number of iterations using scan
    final_carry, _ = jax.lax.scan(body_fn, init_carry, None, length=max_iterations)

    _, a_final, _ = final_carry

    # Combine Newtonian and GR terms
    a = a_newt + a_final

    return a

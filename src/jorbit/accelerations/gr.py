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
from jorbit.accelerations.newtonian import newtonian_gravity
from jorbit.data.constants import SPEED_OF_LIGHT


# equivalent of rebx_calculate_gr_full in reboundx
@partial(jax.jit, static_argnames=["max_iterations"])
def ppn_gravity(
    inputs: SystemState,
    max_iterations: int = 10,
) -> jnp.ndarray:

    c2 = SystemState.acceleration_func_kwargs.get("c2", SPEED_OF_LIGHT**2)

    # same as newtonian_gravity:
    M = inputs.massive_positions.shape[0]  # number of massive particles
    T = inputs.tracer_positions.shape[0]  # number of tracer particles

    # 1. Compute accelerations on massive particles due to other massive particles
    dx_massive = (
        inputs.massive_positions[:, None, :] - inputs.massive_positions[None, :, :]
    )  # (M,M,3)
    r2_massive = jnp.sum(dx_massive * dx_massive, axis=-1)  # (M,M)
    r_massive = jnp.sqrt(r2_massive)  # (M,M)
    r3_massive = r2_massive * jnp.sqrt(r2_massive)  # (M,M)

    # Mask for i!=j calculations among massive particles
    mask_massive = ~jnp.eye(M, dtype=bool)  # (M,M)
    prefac_massive = jnp.where(mask_massive, 1.0 / r3_massive, 0.0)

    # Accelerations on massive particles from massive particles
    a_newt_massive = -jnp.sum(
        prefac_massive[:, :, None]
        * dx_massive
        * jnp.exp(inputs.log_gms[None, :, None]),
        axis=1,
    )  # (M,3)

    # 2. Compute accelerations on tracer particles due to massive particles
    # (This will work even when T=0 due to JAX's shape polymorphism)
    dx_tracer = (
        inputs.tracer_positions[:, None, :] - inputs.massive_positions[None, :, :]
    )  # (T,M,3)
    r2_tracer = jnp.sum(dx_tracer * dx_tracer, axis=-1)  # (T,M)
    r_tracer = jnp.sqrt(r2_tracer)  # (T,M)
    r3_tracer = r2_tracer * jnp.sqrt(r2_tracer)  # (T,M)

    # Accelerations on tracer particles from massive particles
    a_newt_tracer = -jnp.sum(
        (1.0 / r3_tracer)[:, :, None]
        * dx_tracer
        * jnp.exp(inputs.log_gms[None, :, None]),
        axis=1,
    )  # (T,3)

    massive_x = inputs.massive_positions
    massive_v = inputs.massive_velocities
    dv_massive = massive_v[:, None, :] - massive_v[None, :, :]  # (M,M,3)
    tracer_x = inputs.tracer_positions
    tracer_v = inputs.tracer_velocities
    dv_tracer = tracer_v[:, None, :] - massive_v[None, :, :]  # (T,M,3)
    gms = jnp.exp(inputs.log_gms)

    x_com = jnp.sum(massive_x * gms[:, None], axis=0) / jnp.sum(gms)
    v_com = jnp.sum(massive_v * gms[:, None], axis=0) / jnp.sum(gms)

    massive_x = massive_x - x_com
    massive_v = massive_v - v_com
    tracer_x = tracer_x - x_com
    tracer_v = tracer_v - v_com

    # Compute velocity-dependent terms
    v2_massive = jnp.sum(massive_v * massive_v, axis=-1)  # (M,)
    v2_tracer = jnp.sum(tracer_v * tracer_v, axis=-1)  # (T,)
    vdot_mm = jnp.sum(massive_v[:, None, :] * massive_v[None, :, :], axis=-1)  # (M,M)
    vdot_tm = jnp.sum(tracer_v[:, None, :] * massive_v[None, :, :], axis=-1)  # (T,M)

    # Compute GR correction terms for massive particles
    a1_mm = jnp.sum((4.0 / c2) * gms / r_massive, axis=1, where=mask_massive)
    a1_mm = jnp.broadcast_to(a1_mm, (M, M)).T

    a2_mm = jnp.sum((1.0 / c2) * gms / r_massive, axis=1, where=mask_massive)
    a2_mm = jnp.broadcast_to(a2_mm, (M, M))

    a3_mm = jnp.broadcast_to(-v2_massive / c2, (M, M)).T
    a4_mm = -2.0 * jnp.broadcast_to(v2_massive, (M, M)) / c2
    a5_mm = (4.0 / c2) * vdot_mm

    a6_0_mm = jnp.sum(dx_massive * massive_v[None, :, :], axis=-1)
    a6_mm = (3.0 / (2 * c2)) * (a6_0_mm**2) / r2_massive

    a7_mm = jnp.sum(dx_massive * a_newt_massive[None, :, :], axis=-1) / (2 * c2)

    factor1_mm = a1_mm + a2_mm + a3_mm + a4_mm + a5_mm + a6_mm + a7_mm
    part1_mm = (
        jnp.broadcast_to(gms, (M, M))[:, :, None]
        * dx_massive
        * factor1_mm[:, :, None]
        / r3_massive[:, :, None]
    )

    factor2_massive = jnp.sum(
        dx_massive * (4 * massive_v[:, None, :] - 3 * massive_v[None, :, :]), axis=-1
    )
    part2_mm = (
        jnp.broadcast_to(gms, (M, M))[:, :, None]
        * (
            factor2_massive[:, :, None] * dv_massive / r3_massive[:, :, None]
            + 7.0 / 2.0 * a_newt_massive[None, :, :] / r_massive[:, :, None]
        )
        / c2
    )

    a_const_massive = jnp.sum(
        part1_mm + part2_mm, axis=1, where=mask_massive[:, :, None]
    )

    # Compute GR correction terms for tracer particles
    a1_tm = (4.0 / c2) * gms[None, :] / r_tracer
    a2_tm = (1.0 / c2) * gms[None, :] / r_tracer
    a3_tm = -v2_tracer[:, None] / c2
    a4_tm = -2.0 * v2_massive[None, :] / c2
    a5_tm = (4.0 / c2) * vdot_tm

    a6_0_tm = jnp.sum(dx_tracer * massive_v[None, :, :], axis=-1)
    a6_tm = (3.0 / (2 * c2)) * (a6_0_tm**2) / r2_tracer

    a7_tm = jnp.sum(dx_tracer * a_newt_tracer[:, None, :], axis=-1) / (2 * c2)

    factor1_tm = a1_tm + a2_tm + a3_tm + a4_tm + a5_tm + a6_tm + a7_tm
    part1_tm = (
        gms[None, :, None] * dx_tracer * factor1_tm[:, :, None] / r3_tracer[:, :, None]
    )

    factor2_tracer = jnp.sum(
        dx_tracer * (4 * tracer_v[:, None, :] - 3 * massive_v[None, :, :]), axis=-1
    )
    part2_tm = (
        gms[None, :, None]
        * (
            factor2_tracer[:, :, None] * dv_tracer / r3_tracer[:, :, None]
            + 7.0 / 2.0 * a_newt_tracer[:, None, :] / r_tracer[:, :, None]
        )
        / c2
    )

    a_const_tracer = jnp.sum(part1_tm + part2_tm, axis=1)

    def iteration_step_massive(a_curr_massive, a_curr_tracer):
        rdota_mm = jnp.sum(dx_massive * a_curr_massive[None, :, :], axis=-1)  # (M,M)
        non_const_massive = jnp.sum(
            (gms[None, :, None] / (2.0 * c2))
            * (
                (dx_massive * rdota_mm[:, :, None] / r3_massive[:, :, None])
                + (7.0 * a_curr_massive[None, :, :] / r_massive[:, :, None])
            ),
            axis=1,
            where=mask_massive[:, :, None],
        )

        rdota_tm = jnp.sum(dx_tracer * a_curr_tracer[:, None, :], axis=-1)  # (T,M)
        non_const_tracer = jnp.sum(
            (gms[None, :, None] / (2.0 * c2))
            * (
                (dx_tracer * rdota_tm[:, :, None] / r3_tracer[:, :, None])
                + (7.0 * a_curr_tracer[:, None, :] / r_tracer[:, :, None])
            ),
            axis=1,
        )

        return non_const_massive, non_const_tracer

    def do_nothing(carry):
        return carry

    def do_iteration(carry):
        a_prev_massive, a_curr_massive, a_prev_tracer, a_curr_tracer, _ = carry
        non_const_massive, non_const_tracer = iteration_step_massive(
            a_curr_massive, a_curr_tracer
        )
        a_next_massive = a_const_massive + non_const_massive
        a_next_tracer = a_const_tracer + non_const_tracer

        ratio_massive = jnp.max(
            jnp.abs((a_next_massive - a_curr_massive) / a_next_massive), initial=0.0
        )
        ratio_tracers = jnp.max(
            jnp.abs((a_next_tracer - a_curr_tracer) / a_next_tracer), initial=0.0
        )
        ratio = jnp.maximum(ratio_massive, ratio_tracers)

        return (a_curr_massive, a_next_massive, a_curr_tracer, a_next_tracer, ratio)

    def body_fn(carry, _):
        a_prev_massive, a_curr_massive, a_prev_tracer, a_curr_tracer, ratio = carry
        should_continue = ratio > 2.220446049250313e-16
        new_carry = jax.lax.cond(should_continue, do_iteration, do_nothing, carry)
        return new_carry, None

    # Initialize with constant terms
    init_a_massive = jnp.zeros_like(a_const_massive)
    init_a_tracer = jnp.zeros_like(a_const_tracer)
    init_carry = (init_a_massive, a_const_massive, init_a_tracer, a_const_tracer, 1.0)

    # Run fixed number of iterations using scan
    final_carry, _ = jax.lax.scan(body_fn, init_carry, None, length=max_iterations)

    # Extract final accelerations
    _, a_final_massive, _, a_final_tracer, _ = final_carry

    # Combine Newtonian and GR terms
    a_massive = a_newt_massive + a_final_massive
    a_tracer = a_newt_tracer + a_final_tracer

    return jnp.concatenate([a_massive, a_tracer], axis=0)

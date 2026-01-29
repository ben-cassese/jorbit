"""General refactor of ias15.py to avoid dataclasses make it more JAX friendly."""

import jax

jax.config.update("jax_enable_x64", True)
# from collections.abc import Callable

import jax.numpy as jnp

from jorbit.data.constants import (
    IAS15_BV_DENOMS,
    IAS15_BX_DENOMS,
    # IAS15_D_MATRIX,
    # IAS15_H,
    # IAS15_c1,
    # IAS15_c2,
    # IAS15_c3,
    # IAS15_c4,
    # IAS15_c5,
    # IAS15_c6,
    # IAS15_c7,
    # IAS15_r1,
    # IAS15_r2,
    # IAS15_r3,
    # IAS15_r4,
    # IAS15_r5,
    # IAS15_r6,
    # IAS15_r7,
)

# from jorbit.utils.states import IAS15IntegratorState, SystemState


@jax.jit
def add_cs(p: jnp.ndarray, csp: jnp.ndarray, inp: jnp.ndarray) -> tuple:
    """Compensated summation.

    Args:
        p (jnp.ndarray):
            The current sum.
        csp (jnp.ndarray):
            The current compensation.
        inp (jnp.ndarray):
            The input to add.

    Returns:
        tuple:
            The new sum and compensation.
    """
    y = inp - csp
    t = p + y
    csp = (t - p) - y
    p = t
    return p, csp


def _estimate_x_v_from_b(
    a0: jnp.ndarray,
    v0: jnp.ndarray,
    x0: jnp.ndarray,
    h: jnp.ndarray,
    dt: jnp.ndarray,
    bp: jnp.ndarray,  # remember to flip it!
) -> tuple[jnp.ndarray, jnp.ndarray]:
    xcoeffs = bp * dt * dt / IAS15_BX_DENOMS
    x, _ = jax.lax.scan(lambda y, _p: (y * h + _p, None), jnp.zeros_like(x0), xcoeffs)
    x *= h * h * h
    x += (v0 * dt) * h + (a0 * dt * dt / 2.0) * h * h + x0

    vcoeffs = bp * dt / IAS15_BV_DENOMS
    v, _ = jax.lax.scan(lambda y, _p: (y * h + _p, None), jnp.zeros_like(x0), vcoeffs)
    v *= h * h
    v += v0 + (a0 * dt) * h
    return x, v


def _refine_sub_g(
    at: jnp.ndarray, a0: jnp.ndarray, previous_gs: jnp.ndarray, r: jnp.ndarray
) -> jnp.ndarray:

    def scan_body(carry: tuple, scan_over: tuple) -> tuple:
        result = carry
        g, r_sub = scan_over
        result = (result - g) * r_sub
        return result, None

    initial_result = (at - a0) * r[0]
    new_g, _ = jax.lax.scan(scan_body, initial_result, (previous_gs, r[1:]))
    return new_g


def _update_bs(
    current_bs: jnp.ndarray,
    current_csbs: jnp.ndarray,
    g_diff: jnp.ndarray,
    c: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return add_cs(current_bs, current_csbs, (g_diff[None, :] * c[:, None, None]))


# beginnings, just scratch
# def _step(
#     initial_system_state: SystemState,
#     acceleration_func: Callable[[SystemState], jnp.ndarray],
#     initial_integrator_state: IAS15IntegratorState,
# ) -> SystemState:
#     t_beginning = initial_system_state.time
#     M = initial_system_state.massive_positions.shape[0]
#     x0 = jnp.concatenate(
#         (initial_system_state.massive_positions, initial_system_state.tracer_positions)
#     )
#     v0 = jnp.concatenate(
#         (
#             initial_system_state.massive_velocities,
#             initial_system_state.tracer_velocities,
#         )
#     )
#     dt = initial_integrator_state.dt
#     a0 = initial_integrator_state.a0
#     b = initial_integrator_state.b

#     b = jnp.stack([b.p0, b.p1, b.p2, b.p3, b.p4, b.p5, b.p6], axis=0)
#     csb = jnp.zeros_like(b)
#     g = jnp.einsum("ij,jnk->ink", IAS15_D_MATRIX, b)

#     def _predictor_corrector_iteration(
#         b: jnp.ndarray,
#         csb: jnp.ndarray,
#         g: jnp.ndarray,
#         predictor_corrector_error: float,
#     ) -> tuple:
#         n = 1
#         step_time = t_beginning + dt * IAS15_H[n]
#         x, v = _estimate_x_v_from_b(
#             a0=a0,
#             v0=v0,
#             x0=x0,
#             h=IAS15_H[n],
#             dt=dt,
#             bp=b,
#         )
#         acc_state = SystemState(
#             massive_positions=x[:M],
#             massive_velocities=v[:M],
#             tracer_positions=x[M:],
#             tracer_velocities=v[M:],
#             log_gms=initial_system_state.log_gms,
#             time=step_time,
#             acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
#         )
#         at = acceleration_func(acc_state)
#         g_old = g[n - 1]
#         g_new = _refine_sub_g(at, a0, g[: n - 1], IAS15_r1)
#         g_diff = g_new - g_old
#         new_bs, new_csbs = _update_bs(b[:n], csb[:n], g_diff, IAS15_c1)
#         g = g.at[n - 1].set(g_new)
#         b = b.at[:n].set(new_bs)
#         csb = csb.at[:n].set(new_csbs)

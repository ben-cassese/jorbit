"""General refactor of ias15.py to avoid dataclasses make it more JAX friendly."""

import jax

jax.config.update("jax_enable_x64", True)
from collections.abc import Callable

import jax.numpy as jnp

from jorbit.data.constants import (
    EPSILON,
    IAS15_BV_DENOMS,
    IAS15_BX_DENOMS,
    IAS15_D_MATRIX,
    IAS15_H,
    IAS15_sub_cs,
    IAS15_sub_rs,
)
from jorbit.utils.states import IAS15IntegratorState, SystemState


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


def _step(
    initial_system_state: SystemState,
    acceleration_func: Callable[[SystemState], jnp.ndarray],
    initial_integrator_state: IAS15IntegratorState,
) -> SystemState:
    t_beginning = initial_system_state.time
    M = initial_system_state.massive_positions.shape[0]
    x0 = jnp.concatenate(
        (initial_system_state.massive_positions, initial_system_state.tracer_positions)
    )
    v0 = jnp.concatenate(
        (
            initial_system_state.massive_velocities,
            initial_system_state.tracer_velocities,
        )
    )
    dt = initial_integrator_state.dt
    a0 = initial_integrator_state.a0
    b = initial_integrator_state.b

    b = jnp.stack([b.p0, b.p1, b.p2, b.p3, b.p4, b.p5, b.p6], axis=0)
    csb = jnp.zeros_like(b)
    g = jnp.einsum("ij,jnk->ink", IAS15_D_MATRIX, b)

    def _predictor_corrector_iteration(
        b: jnp.ndarray,
        csb: jnp.ndarray,
        g: jnp.ndarray,
        predictor_corrector_error: float,
    ) -> tuple:
        predictor_corrector_error_last = predictor_corrector_error
        predictor_corrector_error = 0.0
        for n, h, c, r in zip(range(1, 8), IAS15_H[1:], IAS15_sub_cs, IAS15_sub_rs):
            # jax.debug.print("Substep number: {n}", n=n)
            step_time = t_beginning + dt * h
            x, v = _estimate_x_v_from_b(
                a0=a0,
                v0=v0,
                x0=x0,
                h=h,
                dt=dt,
                bp=b[::-1],
            )
            # jax.debug.print("x[0]: {x0}", x0=x[0])
            acc_state = SystemState(
                massive_positions=x[:M],
                massive_velocities=v[:M],
                tracer_positions=x[M:],
                tracer_velocities=v[M:],
                log_gms=initial_system_state.log_gms,
                time=step_time,
                acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
            )
            at = acceleration_func(acc_state)
            g_old = g[n - 1]
            # jax.debug.print("g_old[0]: {g0}", g0=g_old[0])
            g_new = _refine_sub_g(at, a0, g[: n - 1], r)
            # jax.debug.print("g_new[0]: {g0}", g0=g_new[0])
            g_diff = g_new - g_old
            new_bs, new_csbs = _update_bs(b[:n], csb[:n], g_diff, c)
            g = g.at[n - 1].set(g_new)
            b = b.at[:n].set(new_bs)
            csb = csb.at[:n].set(new_csbs)
            # jax.debug.print("b[0]: {b0}\n", b0=b[0])

        maxa = jnp.max(jnp.abs(at))
        maxb6tmp = jnp.max(jnp.abs(g_diff))

        predictor_corrector_error = jnp.abs(maxb6tmp / maxa)
        # jax.debug.print("---")

        return b, csb, g, predictor_corrector_error, predictor_corrector_error_last

    def _scan_func(carry: tuple, scan_over: None) -> tuple:
        b, csb, g, predictor_corrector_error = carry
        b, csb, g, predictor_corrector_error, predictor_corrector_error_last = (
            _predictor_corrector_iteration(b, csb, g, predictor_corrector_error)
        )
        _condition = (predictor_corrector_error < EPSILON) | (
            (scan_over > 2)
            & (predictor_corrector_error > predictor_corrector_error_last)
        )
        return (b, csb, g, predictor_corrector_error), None

    initial_carry = (b, csb, g, 1e300)
    (b, csb, g, _predictor_corrector_error), _ = jax.lax.scan(
        _scan_func, initial_carry, jnp.arange(3)
    )

    dt_done = dt
    csx = jnp.zeros_like(x0)
    csv = jnp.zeros_like(v0)
    # jax.debug.print("\nFinal b[0]: {b0}", b0=b[0])
    x0, csx = add_cs(x0, csx, b[6] / 72.0 * dt_done * dt_done)
    x0, csx = add_cs(x0, csx, b[5] / 56.0 * dt_done * dt_done)
    x0, csx = add_cs(x0, csx, b[4] / 42.0 * dt_done * dt_done)
    x0, csx = add_cs(x0, csx, b[3] / 30.0 * dt_done * dt_done)
    x0, csx = add_cs(x0, csx, b[2] / 20.0 * dt_done * dt_done)
    x0, csx = add_cs(x0, csx, b[1] / 12.0 * dt_done * dt_done)
    x0, csx = add_cs(x0, csx, b[0] / 6.0 * dt_done * dt_done)
    x0, csx = add_cs(x0, csx, a0 / 2.0 * dt_done * dt_done)
    x0, csx = add_cs(x0, csx, v0 * dt_done)
    v0, csv = add_cs(v0, csv, b[6] / 8.0 * dt_done)
    v0, csv = add_cs(v0, csv, b[5] / 7.0 * dt_done)
    v0, csv = add_cs(v0, csv, b[4] / 6.0 * dt_done)
    v0, csv = add_cs(v0, csv, b[3] / 5.0 * dt_done)
    v0, csv = add_cs(v0, csv, b[2] / 4.0 * dt_done)
    v0, csv = add_cs(v0, csv, b[1] / 3.0 * dt_done)
    v0, csv = add_cs(v0, csv, b[0] / 2.0 * dt_done)
    v0, csv = add_cs(v0, csv, a0 * dt_done)
    new_system_state = SystemState(
        massive_positions=x0[:M],
        massive_velocities=v0[:M],
        tracer_positions=x0[M:],
        tracer_velocities=v0[M:],
        log_gms=initial_system_state.log_gms,
        time=t_beginning + dt_done,
        acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
    )

    return new_system_state

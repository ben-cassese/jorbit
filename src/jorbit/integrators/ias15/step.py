"""The single-step IAS15 predictor-corrector and its helpers."""

import jax

jax.config.update("jax_enable_x64", True)

from collections.abc import Callable

import jax.numpy as jnp

from jorbit.data.constants import (
    EPSILON,
    IAS15_BEZIER_COEFFS,
    IAS15_D_MATRIX,
    IAS15_H,
    IAS15_SAFETY_FACTOR,
    IAS15_sub_cs,
    IAS15_sub_rs,
)
from jorbit.integrators.ias15.helpers import _estimate_x_v_from_b, add_cs
from jorbit.utils.states import IAS15IntegratorState, SystemState


@jax.jit
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


@jax.jit
def _update_bs(
    current_bs: jnp.ndarray,
    current_csbs: jnp.ndarray,
    g_diff: jnp.ndarray,
    c: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return add_cs(current_bs, current_csbs, (g_diff[None, :] * c[:, None, None]))


@jax.jit
def _predict_next_step(ratio: float, e: jnp.ndarray, b: jnp.ndarray) -> tuple:

    def large_ratio(ratio: float, e: jnp.ndarray, b: jnp.ndarray) -> tuple:
        # On saturated growth, zero only `e` and keep `b` as the starting point
        # for the next step's PC iteration.
        e_new = jnp.zeros_like(e)
        return e_new, b

    def reasonable_ratio(ratio: float, e: jnp.ndarray, b: jnp.ndarray) -> tuple:
        qs = ratio ** jnp.arange(1, 8)
        diff = b - e
        e = jnp.einsum("i,ij,j...->i...", qs, IAS15_BEZIER_COEFFS, b)
        b = e + diff
        return e, b

    e, b = jax.lax.cond(
        ratio >= 1 / IAS15_SAFETY_FACTOR, large_ratio, reasonable_ratio, ratio, e, b
    )

    return e, b


@jax.jit
def ias15_step(
    initial_system_state: SystemState,
    acceleration_func: Callable[[SystemState], jnp.ndarray],
    initial_integrator_state: IAS15IntegratorState,
    step_scheduler: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, float, jnp.ndarray, jnp.ndarray], float
    ],
) -> SystemState:
    """Take a single step using the IAS15 integrator.

    Contains all of the predictor/corrector logic and step validity checks. Does not
    accept any pre-computed perturber information, since we don't know the times this
    will be needed until runtime. For a static version that accepts pre-computed
    perturber data, see ias15_static_step.

    Args:
        initial_system_state (SystemState):
            The initial system state.
        acceleration_func (Callable[[SystemState], jnp.ndarray]):
            The acceleration function.
        initial_integrator_state (IAS15IntegratorState):
            The initial integrator state.
        step_scheduler (Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, float, jnp.ndarray, jnp.ndarray], float]):
            The step scheduler function, which is either going to be
            next_proposed_dt_PRS23 or next_proposed_dt_global

    Returns:
        tuple[SystemState, IAS15IntegratorState, jnp.ndarray]:
            The new system state, the new integrator state (with the *predicted*
            next-step b coefficients), and the *converged* b coefficients for the
            step just completed, shape (7, n_particles, 3). The converged b is
            what should be stored when building dense output for interpolation.
    """
    t_beginning = initial_system_state.relative_time
    # The absolute-JD anchor is constant across the step; the integrator marches
    # relative_time and carries time_reference through unchanged.
    t_ref = initial_system_state.time_reference

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
    csx = initial_integrator_state.csx
    csv = initial_integrator_state.csv
    e = initial_integrator_state.e
    b = initial_integrator_state.b

    csb = jnp.zeros_like(b)
    g = jnp.einsum("ij,jnk->ink", IAS15_D_MATRIX, b)

    def _do_nothing(
        b: jnp.ndarray,
        csb: jnp.ndarray,
        g: jnp.ndarray,
        predictor_corrector_error: jnp.ndarray,
        at_last: jnp.ndarray,
        x_end: jnp.ndarray,
        v_end: jnp.ndarray,
    ) -> tuple:
        # jax.debug.print("just chillin")
        return (
            b,
            csb,
            g,
            predictor_corrector_error,
            predictor_corrector_error,
            at_last,
            x_end,
            v_end,
        )

    def _predictor_corrector_iteration(
        b: jnp.ndarray,
        csb: jnp.ndarray,
        g: jnp.ndarray,
        predictor_corrector_error: float,
        at_last: jnp.ndarray,
        x_end: jnp.ndarray,
        v_end: jnp.ndarray,
    ) -> tuple:
        # jax.debug.print("PC iteration starting")
        del at_last, x_end, v_end
        predictor_corrector_error_last = predictor_corrector_error
        predictor_corrector_error = 0.0
        for n, h, c, r in zip(
            range(1, 8), IAS15_H[1:], IAS15_sub_cs, IAS15_sub_rs, strict=True
        ):
            # jax.debug.print("   pc iter {n}: g={g}", n=n, g=g)
            step_time = t_beginning + dt * h
            x, v = _estimate_x_v_from_b(
                a0=a0,
                v0=v0,
                x0=x0,
                h=h,
                dt=dt,
                bp=b[::-1],
            )
            # note that the fixed perturber bits likely can/will be overwritten by the
            # acceleration function- see ias15_static_step + create_static_default_acceleration_func
            acc_state = SystemState(
                massive_positions=x[:M],
                massive_velocities=v[:M],
                tracer_positions=x[M:],
                tracer_velocities=v[M:],
                log_gms=initial_system_state.log_gms,
                time_reference=t_ref,
                relative_time=step_time,
                fixed_perturber_positions=jnp.empty(
                    (0, 3),
                ),
                fixed_perturber_velocities=jnp.empty(
                    (0, 3),
                ),
                fixed_perturber_log_gms=jnp.empty((0,)),
                acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
            )
            at = acceleration_func(acc_state)
            g_old = g[n - 1]
            g_new = _refine_sub_g(at, a0, g[: n - 1], r)
            g_diff = g_new - g_old
            # jax.debug.print("   min/max g_diff: {x}, {y}", x=jnp.max(g_diff), y=jnp.min(g_diff))
            new_bs, new_csbs = _update_bs(b[:n], csb[:n], g_diff, c)
            g = g.at[n - 1].set(g_new)
            b = b.at[:n].set(new_bs)
            csb = csb.at[:n].set(new_csbs)

        maxa = jnp.max(jnp.abs(at))
        maxb6tmp = jnp.max(jnp.abs(g_diff))
        # jax.debug.print("maxa: {maxa}, maxb6tmp: {maxb6tmp}", maxa=maxa, maxb6tmp=maxb6tmp)

        predictor_corrector_error = jnp.abs(maxb6tmp / maxa)
        # jax.debug.print("PC iteration error: {error}\n\n", error=predictor_corrector_error)

        # `at`, `x`, `v` here are from the last sub-step (n=7, h=IAS15_H[7]=0.977),
        # i.e. the freshly-evaluated end-of-step acceleration and predictor state.
        # REBOUND's GLOBAL controller uses these (integrator_ias15.c:382-385, 547).
        return (
            b,
            csb,
            g,
            predictor_corrector_error,
            predictor_corrector_error_last,
            at,
            x,
            v,
        )

    def scan_func(carry: tuple, scan_over: int) -> tuple:
        (
            b,
            csb,
            g,
            predictor_corrector_error,
            predictor_corrector_error_last,
            at_last,
            x_end,
            v_end,
        ) = carry

        condition = (predictor_corrector_error < EPSILON) | (
            (scan_over > 2)
            & (predictor_corrector_error > predictor_corrector_error_last)
        )

        carry = jax.lax.cond(
            condition,
            _do_nothing,
            _predictor_corrector_iteration,
            b,
            csb,
            g,
            predictor_corrector_error,
            at_last,
            x_end,
            v_end,
        )
        return carry, None

    initial_carry = (b, csb, g, 1e300, 2.0, a0, x0, v0)
    (b, csb, g, _pc_error, _pc_error_last, at_final, x_end, v_end), _ = jax.lax.scan(
        scan_func, initial_carry, jnp.arange(12)
    )

    dt_done = dt
    next_dt = step_scheduler(a0, at_final, b, dt, x_end, v_end)

    def step_too_ambitious(
        x0: jnp.ndarray,
        v0: jnp.ndarray,
        csx: jnp.ndarray,
        csv: jnp.ndarray,
        dt_done: float,
        next_dt: float,
    ) -> tuple:
        dt_done = 0.0
        return x0, v0, dt_done, next_dt

    def step_was_good(
        x0: jnp.ndarray,
        v0: jnp.ndarray,
        csx: jnp.ndarray,
        csv: jnp.ndarray,
        dt_done: float,
        next_dt: float,
    ) -> tuple:
        safe_next_dt = jnp.where(
            next_dt / dt_done > 1 / IAS15_SAFETY_FACTOR,
            dt_done / IAS15_SAFETY_FACTOR,
            next_dt,
        )

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

        return x0, v0, dt_done, safe_next_dt

    x0, v0, dt_done, next_dt = jax.lax.cond(
        jnp.abs(next_dt / dt_done) < IAS15_SAFETY_FACTOR,
        step_too_ambitious,
        step_was_good,
        x0,
        v0,
        csx,
        csv,
        dt_done,
        next_dt,
    )

    new_system_state = SystemState(
        massive_positions=x0[:M],
        massive_velocities=v0[:M],
        tracer_positions=x0[M:],
        tracer_velocities=v0[M:],
        log_gms=initial_system_state.log_gms,
        time_reference=t_ref,
        relative_time=t_beginning + dt_done,
        fixed_perturber_positions=initial_system_state.fixed_perturber_positions * 0,
        fixed_perturber_velocities=initial_system_state.fixed_perturber_velocities * 0,
        fixed_perturber_log_gms=initial_system_state.fixed_perturber_log_gms * 0,
        acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
    )

    # On rejection (dt_done == 0), force ratio into the large_ratio no-op
    # branch of _predict_next_step (zeros e, keeps b).
    ratio = jnp.where(dt_done == 0.0, 100.0, next_dt / dt_done)
    predicted_next_e, predicted_next_b = _predict_next_step(ratio, e, b)

    new_integrator_state = IAS15IntegratorState(
        g=g,
        b=predicted_next_b,
        e=predicted_next_e,
        csx=csx,
        csv=csv,
        a0=acceleration_func(new_system_state),
        dt=next_dt,
        dt_last_done=dt_done,
    )

    return new_system_state, new_integrator_state, b

"""A JAX implementation of the IAS15 integrator.

This is a pythonized/jaxified version of the IAS15 integrator from Rein & Spiegel (2015)
(DOI: 10.1093/mnras/stu2164), currently implemented in REBOUND. It used to follow the
implementation found in the REBOUND source as closely as possible; see < v1.2 for that.

The original code is available on `github <https://github.com/hannorein/rebound/blob/0b5c85d836fec20bc284d1f1bb326f418e11f591/src/integrator_ias15.c>`_.
Accessed Summer 2023, re-visited Fall 2024. Refactored early 2026.

Many thanks to the REBOUND developers for their work on this integrator, and for making it open source!
"""

# This is a pythonized/jaxified version of the IAS15 integrator from
# Rein & Spiegel (2015) (DOI: 10.1093/mnras/stu2164), currently implemented in REBOUND.
# The original code is available at https://github.com/hannorein/rebound/blob/0b5c85d836fec20bc284d1f1bb326f418e11f591/src/integrator_ias15.c
# Accessed Summer 2023, re-visited Fall 2024. Refactored early 2026.

# Many thanks to the REBOUND developers for their work on this integrator,
# and for making it open source!

import jax

jax.config.update("jax_enable_x64", True)
from collections.abc import Callable

import jax.numpy as jnp

from jorbit.data.constants import (
    EPSILON,
    IAS15_BEZIER_COEFFS,
    IAS15_BV_DENOMS,
    IAS15_BX_DENOMS,
    IAS15_D_MATRIX,
    IAS15_H,
    IAS15_SAFETY_FACTOR,
    IAS15_EPS_Modified,
    IAS15_sub_cs,
    IAS15_sub_rs,
)
from jorbit.utils.states import IAS15IntegratorState, SystemState


def initialize_ias15_integrator_state(a0: jnp.ndarray) -> IAS15IntegratorState:
    """Initializes the IAS15IntegratorState dataclass with zeros.

    Args:
        a0 (jnp.ndarray):
            The initial acceleration.

    Returns:
        IAS15IntegratorState:
            An instance of the IAS15IntegratorState dataclass with zeros.
    """
    n_particles = a0.shape[0]
    return IAS15IntegratorState(
        g=jnp.zeros((7, n_particles, 3), dtype=jnp.float64),
        b=jnp.zeros((7, n_particles, 3), dtype=jnp.float64),
        e=jnp.zeros((7, n_particles, 3), dtype=jnp.float64),
        csx=jnp.zeros((n_particles, 3), dtype=jnp.float64),
        csv=jnp.zeros((n_particles, 3), dtype=jnp.float64),
        a0=a0,
        dt=10.0,  # 10 days
        dt_last_done=0.0,
    )


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


@jax.jit
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
def _next_proposed_dt(a0: jnp.ndarray, b: jnp.ndarray, dt_done: float) -> jnp.ndarray:
    tmp = a0 + jnp.sum(b, axis=0)
    y2 = jnp.sum(tmp * tmp, axis=1)

    coeffs_1 = jnp.arange(1, 8)
    tmp = jnp.sum(coeffs_1[:, None, None] * b, axis=0)
    y3 = jnp.sum(tmp * tmp, axis=1)

    coeffs_2 = jnp.arange(2, 8) * jnp.arange(1, 7)
    tmp = jnp.sum(coeffs_2[:, None, None] * b[1:], axis=0)
    y4 = jnp.sum(tmp * tmp, axis=1)

    timescale2 = 2.0 * y2 / (y3 + jnp.sqrt(y4 * y2))  # PRS23
    min_timescale2 = jnp.nanmin(timescale2)
    dt_new = jnp.sqrt(min_timescale2) * dt_done * IAS15_EPS_Modified
    return dt_new


@jax.jit
def _predict_next_step(ratio: float, e: jnp.ndarray, b: jnp.ndarray) -> tuple:

    def large_ratio(ratio: float, e: jnp.ndarray, b: jnp.ndarray) -> tuple:
        e_new = jnp.zeros_like(e)
        return e_new, b

    def reasonable_ratio(ratio: float, e: jnp.ndarray, b: jnp.ndarray) -> tuple:
        qs = ratio ** jnp.arange(1, 8)
        diff = b - e
        e = jnp.einsum("i,ij,j...->i...", qs, IAS15_BEZIER_COEFFS, b)
        b = e - diff
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

    Returns:
        tuple[SystemState, IAS15IntegratorState]:
            The new system state and new integrator state.
    """
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
    ) -> tuple:
        # jax.debug.print("just chillin")
        return b, csb, g, predictor_corrector_error, predictor_corrector_error

    def _predictor_corrector_iteration(
        b: jnp.ndarray,
        csb: jnp.ndarray,
        g: jnp.ndarray,
        predictor_corrector_error: float,
    ) -> tuple:
        predictor_corrector_error_last = predictor_corrector_error
        predictor_corrector_error = 0.0
        for n, h, c, r in zip(
            range(1, 8), IAS15_H[1:], IAS15_sub_cs, IAS15_sub_rs, strict=True
        ):
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
                time=step_time,
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
            new_bs, new_csbs = _update_bs(b[:n], csb[:n], g_diff, c)
            g = g.at[n - 1].set(g_new)
            b = b.at[:n].set(new_bs)
            csb = csb.at[:n].set(new_csbs)

        maxa = jnp.max(jnp.abs(at))
        maxb6tmp = jnp.max(jnp.abs(g_diff))

        predictor_corrector_error = jnp.abs(maxb6tmp / maxa)

        return b, csb, g, predictor_corrector_error, predictor_corrector_error_last

    def scan_func(carry: tuple, scan_over: int) -> tuple:
        b, csb, g, predictor_corrector_error, predictor_corrector_error_last = carry

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
        )
        return carry, None

    initial_carry = (b, csb, g, 1e300, 2.0)
    (b, csb, g, _pc_error, _pc_error_last), _ = jax.lax.scan(
        scan_func, initial_carry, jnp.arange(10)
    )

    dt_done = dt
    next_dt = _next_proposed_dt(a0, b, dt)

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
        time=t_beginning + dt_done,
        fixed_perturber_positions=initial_system_state.fixed_perturber_positions * 0,
        fixed_perturber_velocities=initial_system_state.fixed_perturber_velocities * 0,
        fixed_perturber_log_gms=initial_system_state.fixed_perturber_log_gms * 0,
        acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
    )

    ratio = next_dt / dt_done
    # ratio = 100 # temporarily disable prediction
    # if we're rejecting the step, trick predict_next_step into not predicting
    ratio = jnp.where(dt_done == 0.0, 100.0, ratio)
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

    return new_system_state, new_integrator_state


@jax.jit
def ias15_evolve(
    initial_system_state: SystemState,
    acceleration_func: Callable[[SystemState], jnp.ndarray],
    times: jnp.ndarray,
    initial_integrator_state: IAS15IntegratorState,
) -> tuple[jnp.ndarray, jnp.ndarray, SystemState, IAS15IntegratorState]:
    """Evolve a system to multiple different timesteps using the IAS15 integrator.

    Chains multiple ias15_step calls together until each timestep is reached. Keeps
    track of the second to last step before each arrival time to avoid setting dt to
    small values representing the final jumps.

    .. warning::
       To avoid potential infinite hangs or osciallating behavior, this function caps
       the maximum number of steps taken between requested times at 10,000. For a
       particle on a radius=1 circular orbit around an m=1 central object, that
       corresponds to about 280 orbits. It will *not* error if the final time isn't
       reached due to the step limit interruption, so keep the jump between times to be
       less than ~200 dynamical times.

    Args:
        initial_system_state (SystemState):
            The initial state of the system.
        acceleration_func (Callable[[SystemState], jnp.ndarray]):
            The acceleration function to use.
        times (jnp.ndarray):
            The times to evolve the system to.
        initial_integrator_state (IAS15IntegratorState):
            The initial state of the integrator.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, SystemState, IAS15IntegratorState]:
            The positions and velocities of the system at each timestep,
            the final state of the system, and the final state of the integrator.
    """

    def evolve(
        initial_system_state: IAS15IntegratorState,
        acceleration_func: Callable,
        final_time: float,
        initial_integrator_state: IAS15IntegratorState,
    ) -> tuple[SystemState, IAS15IntegratorState]:
        def step_needed(args: tuple) -> tuple:
            system_state, integrator_state, last_meaningful_dt, iter_num = args

            t = system_state.time
            # integrator_state.dt = 0.0001

            diff = final_time - t
            step_length = jnp.sign(diff) * jnp.min(
                jnp.array([jnp.abs(diff), jnp.abs(integrator_state.dt)])
            )

            # jax.debug.print(
            #     "another step is needed. the current time is {x}, the final time is {y}, the diff is {q},  \nintegrator_dt is {w}, step_length being set to {z}",
            #     x=t,
            #     y=final_time,
            #     q=diff,
            #     z=step_length,
            #     w=integrator_state.dt,
            # )
            integrator_state.dt = step_length
            # system_state, integrator_state = ias15_step_dynamic_predictor(
            #     system_state, acceleration_func, integrator_state
            # )
            system_state, integrator_state = ias15_step(
                system_state, acceleration_func, integrator_state
            )
            return system_state, integrator_state, last_meaningful_dt, iter_num + 1

        def cond_func(args: tuple) -> bool:
            system_state, integrator_state, _last_meaningful_dt, iter_num = args
            t = system_state.time

            step_length = jnp.sign(final_time - t) * jnp.min(
                jnp.array([jnp.abs(final_time - t), jnp.abs(integrator_state.dt)])
            )
            return (step_length != 0) & (iter_num < 10_000)

        final_system_state, final_integrator_state, _last_meaningful_dt, _iter_num = (
            jax.lax.while_loop(
                cond_func,
                step_needed,
                (
                    initial_system_state,
                    initial_integrator_state,
                    initial_integrator_state.dt,
                    0,
                ),
            )
        )
        # jax.debug.print(
        #     "finished taking steps to goal time in {x} iterations", x=_iter_num
        # )

        return (final_system_state, final_integrator_state)

    def scan_func(carry: tuple, scan_over: float) -> tuple:
        # jax.debug.print(
        #     "\nattempting jump to next time: {x}. the current time is: {y}",
        #     x=scan_over,
        #     y=carry[0].time,
        # )
        system_state, integrator_state = carry
        final_time = scan_over
        system_state, integrator_state = evolve(
            system_state, acceleration_func, final_time, integrator_state
        )
        return (system_state, integrator_state), (
            jnp.concatenate(
                (
                    system_state.massive_positions,
                    system_state.tracer_positions,
                )
            ),
            jnp.concatenate(
                (
                    system_state.massive_velocities,
                    system_state.tracer_velocities,
                )
            ),
        )

    (final_system_state, final_integrator_state), (positions, velocities) = (
        jax.lax.scan(scan_func, (initial_system_state, initial_integrator_state), times)
    )
    return positions, velocities, final_system_state, final_integrator_state

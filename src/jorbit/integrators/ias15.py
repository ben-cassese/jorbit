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
    IAS15_EPSILON,
    IAS15_H,
    IAS15_MIN_DT,
    IAS15_SAFETY_FACTOR,
    IAS15_EPS_Modified,
    IAS15_sub_cs,
    IAS15_sub_rs,
)
from jorbit.utils.states import IAS15IntegratorState, SystemState

# Maximum number of accepted adaptive steps captured by ias15_evolve's dense-output
# buffer. ASSIST GLOBAL+min_dt=0.001 takes ~2.1k steps for the 2029 Apophis flyby
# year; jorbit's port currently takes ~13k under the same recipe (sits at the floor
# longer than ASSIST due to PC/numerical b6 differences out of scope). 15000 leaves
# headroom for slightly tighter encounters and matches that envelope.
IAS15_MAX_DYNAMIC_STEPS = 15_000


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
    zeros_b = jnp.zeros((7, n_particles, 3), dtype=jnp.float64)
    return IAS15IntegratorState(
        g=zeros_b,
        b=zeros_b,
        e=zeros_b,
        csx=jnp.zeros((n_particles, 3), dtype=jnp.float64),
        csv=jnp.zeros((n_particles, 3), dtype=jnp.float64),
        a0=a0,
        dt=0.001,
        dt_last_done=0.0,
        b_last=zeros_b,
        e_last=zeros_b,
        dt_last_accepted=0.0,
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


def precompute_interpolation_indices(
    t_step_starts: jnp.ndarray,
    dts: jnp.ndarray,
    query_times: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Precompute the step indices and fractional times for interpolation.

    Call this once during setup, then pass the results into
    interpolate_from_dense_output to avoid redundant searchsorted calls
    inside the JIT'd residuals function.

    Args:
        t_step_starts (jnp.ndarray):
            Start time of each step, shape (n_steps,).
        dts (jnp.ndarray):
            Per-step time step sizes, shape (n_steps,).
        query_times (jnp.ndarray):
            Times at which to interpolate, shape (n_queries,).

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]:
            step_indices: Integer index of the containing step for each query time,
                shape (n_queries,).
            h_values: Fractional time within each step (0 to 1),
                shape (n_queries,).
    """
    step_indices = jnp.searchsorted(t_step_starts, query_times, side="right") - 1
    h_values = (query_times - t_step_starts[step_indices]) / dts[step_indices]
    return step_indices, h_values


def make_ltt_propagator(
    b_step: jnp.ndarray,
    a0_step: jnp.ndarray,
    x0_step: jnp.ndarray,
    v0_step: jnp.ndarray,
    dt_step: jnp.ndarray,
    h_obs: jnp.ndarray,
) -> jax.tree_util.Partial:
    """Build a closure that evaluates the IAS15 polynomial at a light-travel-delayed time.

    Used inside ``on_sky`` to propagate a particle backward by the light travel time
    using the converged 7th-order Hermite polynomial for the step containing the
    observation time, instead of a constant-acceleration Taylor expansion.

    The returned closure maps a (negative) time offset ``dt`` to the particle's
    position at fractional position ``h_obs + dt / dt_step`` within the step. It
    accepts ``h`` slightly outside ``[0, 1]`` (i.e. it will extrapolate within the
    same step's polynomial) — typically only by a small amount, since the LTT is
    much shorter than ``dt_step`` for normal solar-system geometries. For close
    flybys with very small steps where LTT may exceed dt_step, this still gives a
    much higher-order correction than the constant-acceleration Taylor.

    Args:
        b_step (jnp.ndarray): Converged b coefficients for this step (single
            particle slice), shape (7, 3).
        a0_step (jnp.ndarray): Acceleration at the start of this step, shape (3,).
        x0_step (jnp.ndarray): Position at the start of this step, shape (3,).
        v0_step (jnp.ndarray): Velocity at the start of this step, shape (3,).
        dt_step (jnp.ndarray): Length of this step (scalar).
        h_obs (jnp.ndarray): Fractional position of the observation time within
            this step, in ``[0, 1]`` (scalar).

    Returns:
        jax.tree_util.Partial:
            A pytree-friendly callable ``f(dt) -> x_at_delayed_time`` of shape (3,).
    """
    # _estimate_x_v_from_b assumes a per-particle axis (IAS15_BX_DENOMS broadcasts
    # against shape (7, n_particles, 3)). Add a singleton particle axis here and
    # strip it in the output so callers can work with plain (3,) / (7, 3) shapes.
    bp = b_step[::-1][:, None, :]
    a0 = a0_step[None, :]
    v0 = v0_step[None, :]
    x0 = x0_step[None, :]

    def f(dt: jnp.ndarray) -> jnp.ndarray:
        h = h_obs + dt / dt_step
        x_at_delayed_time, _ = _estimate_x_v_from_b(a0, v0, x0, h, dt_step, bp)
        return x_at_delayed_time[0]

    return jax.tree_util.Partial(f)


@jax.jit
def interpolate_from_dense_output(
    b_all: jnp.ndarray,
    a0_all: jnp.ndarray,
    x0_all: jnp.ndarray,
    v0_all: jnp.ndarray,
    dts: jnp.ndarray,
    step_indices: jnp.ndarray,
    h_values: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Interpolate positions and velocities at arbitrary times from stored IAS15 polynomial data.

    Uses the b coefficients from completed IAS15 steps to evaluate the 7th-order
    polynomial at fractional times within each step, without re-integrating.

    The step_indices and h_values should be precomputed via
    precompute_interpolation_indices. Since they depend only on the fixed step
    structure and observation times (not the particle state), precomputing them
    keeps searchsorted out of the JIT graph and avoids redundant work on every
    forward and backward pass.

    Args:
        b_all (jnp.ndarray):
            Per-step b coefficients, shape (n_steps, 7, n_particles, 3).
        a0_all (jnp.ndarray):
            Per-step initial accelerations, shape (n_steps, n_particles, 3).
        x0_all (jnp.ndarray):
            Per-step initial positions, shape (n_steps, n_particles, 3).
        v0_all (jnp.ndarray):
            Per-step initial velocities, shape (n_steps, n_particles, 3).
        dts (jnp.ndarray):
            Per-step time step sizes, shape (n_steps,).
        step_indices (jnp.ndarray):
            Index of the containing step for each query time, shape (n_queries,).
            From precompute_interpolation_indices.
        h_values (jnp.ndarray):
            Fractional time within each step (0 to 1), shape (n_queries,).
            From precompute_interpolation_indices.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]:
            Interpolated positions and velocities, each shape (n_queries, n_particles, 3).
    """
    b = b_all[step_indices]
    a0 = a0_all[step_indices]
    x0 = x0_all[step_indices]
    v0 = v0_all[step_indices]
    dt = dts[step_indices]

    positions, velocities = jax.vmap(
        lambda a, v, x, _h, _dt, _b: _estimate_x_v_from_b(a, v, x, _h, _dt, _b[::-1])
    )(a0, v0, x0, h_values, dt, b)

    return positions, velocities


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
def next_proposed_dt_PRS23(
    a0: jnp.ndarray,
    at_fresh: jnp.ndarray,
    b: jnp.ndarray,
    dt_done: float,
    x_end: jnp.ndarray,
    v_end: jnp.ndarray,
) -> jnp.ndarray:
    """The PRS23 step controller."""
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
def next_proposed_dt_global(
    a0: jnp.ndarray,
    at_fresh: jnp.ndarray,
    b: jnp.ndarray,
    dt_done: float,
    x_end: jnp.ndarray,
    v_end: jnp.ndarray,
) -> jnp.ndarray:
    """REBOUND's GLOBAL step controller (legacy, used by ASSIST).

    Compares the magnitude of the highest-order polynomial coefficient (`b[6]`)
    to the freshly-evaluated end-of-step acceleration (`at_fresh`, taken from
    the last predictor-corrector sub-step at h = IAS15_H[7] = 0.977). Includes
    REBOUND's "slow-acceleration" filter that skips particles with
    `v²·dt²/x² < 1e-16`, evaluated on the END-of-step predictor state
    (`x_end, v_end`) to match REBOUND's `particles[mi]` semantics
    (`integrator_ias15.c:543-558`). Falls back to `dt/safety_factor` growth
    when no particle contributes. Finally clamps the proposed step to
    `IAS15_MIN_DT`. See REBOUND `integrator_ias15.c:534-619`. ASSIST forces
    this mode at `assist.c:446`.
    """
    del a0
    v2 = jnp.sum(v_end * v_end, axis=1)
    x2 = jnp.sum(x_end * x_end, axis=1)
    keep = (v2 * dt_done * dt_done / x2) >= 1e-16
    at_masked = jnp.where(keep[:, None], at_fresh, 0.0)
    b6_masked = jnp.where(keep[:, None], b[6], 0.0)
    maxa = jnp.max(jnp.abs(at_masked))
    maxj = jnp.max(jnp.abs(b6_masked))
    integrator_error = maxj / maxa
    dt_new = jnp.where(
        jnp.isfinite(integrator_error) & (integrator_error > 0),
        dt_done * jnp.power(IAS15_EPSILON / integrator_error, 1.0 / 7.0),
        dt_done / IAS15_SAFETY_FACTOR,
    )
    dt_new = jnp.where(
        jnp.abs(dt_new) < IAS15_MIN_DT,
        jnp.sign(dt_new) * IAS15_MIN_DT,
        dt_new,
    )
    return dt_new


@jax.jit
def _predict_next_step(ratio: float, e: jnp.ndarray, b: jnp.ndarray) -> tuple:

    def large_ratio(ratio: float, e: jnp.ndarray, b: jnp.ndarray) -> tuple:
        # probably delete this comment
        # When the dt ratio is large (saturated growth or pathological rejection),
        # zero only `e` and keep `b`. This is the heuristic from before the REBOUND
        # parity attempt; tested empirically to give sub-km PRS23 accuracy on the
        # Apophis flyby year. REBOUND zeros both at ratio>20, but that interaction
        # with jorbit's PC degrades accuracy here, so we keep the older behavior.
        e_new = jnp.zeros_like(e)
        b_new = jnp.zeros_like(b)
        return e_new, b_new

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
    # REBOUND-style "last accepted" snapshots; used as the source for predict_next_step
    # when the current step is rejected (integrator_ias15.c:637-640).
    b_last_in = initial_integrator_state.b_last
    e_last_in = initial_integrator_state.e_last
    dt_last_accepted_in = initial_integrator_state.dt_last_accepted

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
        time=t_beginning + dt_done,
        fixed_perturber_positions=initial_system_state.fixed_perturber_positions * 0,
        fixed_perturber_velocities=initial_system_state.fixed_perturber_velocities * 0,
        fixed_perturber_log_gms=initial_system_state.fixed_perturber_log_gms * 0,
        acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
    )

    # Match REBOUND's predict_next_step semantics:
    #   accepted (integrator_ias15.c:696-697):
    #       ratio = next_dt / dt_done                           # this step's dt
    #       predict in-place from current (e, b)
    #   rejected (integrator_ias15.c:637-640):
    #       ratio = next_dt / dt_last_accepted                  # last successful dt
    #       predict from (e_last, b_last) into current (e, b)
    #       (and skip the predict entirely on the very first step, when
    #        dt_last_accepted is still 0).
    accepted = dt_done != 0.0
    first_step_reject = (~accepted) & (dt_last_accepted_in == 0.0)
    denom = jnp.where(accepted, dt_done, dt_last_accepted_in)
    ratio = jnp.where(first_step_reject, 1.0, next_dt / denom)
    e_source = jnp.where(accepted, e, e_last_in)
    b_source = jnp.where(accepted, b, b_last_in)
    predicted_next_e, predicted_next_b = _predict_next_step(ratio, e_source, b_source)
    # On the first-step rejection there is no last-accepted state to extrapolate
    # from, so leave e, b as the converged-but-rejected values; PC will refine
    # them on the retry from those starting points (matches REBOUND's behavior
    # of skipping the predict_next_step call when dt_last_done == 0).
    predicted_next_e = jnp.where(first_step_reject, e, predicted_next_e)
    predicted_next_b = jnp.where(first_step_reject, b, predicted_next_b)

    # Update "last accepted" snapshots only on a successful step.
    new_b_last = jnp.where(accepted, b, b_last_in)
    new_e_last = jnp.where(accepted, e, e_last_in)
    new_dt_last_accepted = jnp.where(accepted, dt_done, dt_last_accepted_in)

    new_integrator_state = IAS15IntegratorState(
        g=g,
        b=predicted_next_b,
        e=predicted_next_e,
        csx=csx,
        csv=csv,
        a0=acceleration_func(new_system_state),
        dt=next_dt,
        dt_last_done=dt_done,
        b_last=new_b_last,
        e_last=new_e_last,
        dt_last_accepted=new_dt_last_accepted,
    )

    return new_system_state, new_integrator_state, b


@jax.jit
def ias15_evolve_forced_landing(
    initial_system_state: SystemState,
    acceleration_func: Callable[[SystemState], jnp.ndarray],
    times: jnp.ndarray,
    initial_integrator_state: IAS15IntegratorState,
    step_scheduler: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, float, jnp.ndarray, jnp.ndarray], float
    ],
) -> tuple[jnp.ndarray, jnp.ndarray, SystemState, IAS15IntegratorState]:
    """Forced-landing IAS15 evolve (internal testing reference only).

    Clamps the adaptive step size so that a step always lands exactly on the next
    entry of ``times``. Kept private because the public ``ias15_evolve`` (below)
    uses dense-output polynomial interpolation instead, which avoids the small
    final jumps that the forced-landing scheme is prone to. This function is
    retained as an independent reference path for tests and benchmarks.

    .. warning::
       Caps the number of steps between requested times at 10,000.

    Args:
        initial_system_state (SystemState):
            The initial state of the system.
        acceleration_func (Callable[[SystemState], jnp.ndarray]):
            The acceleration function to use.
        times (jnp.ndarray):
            The times to evolve the system to.
        initial_integrator_state (IAS15IntegratorState):
            The initial state of the integrator.
        step_scheduler (Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, float, jnp.ndarray, jnp.ndarray], float]):
            The step scheduler function to use for determining the next proposed
            step size.

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

            diff = final_time - t
            step_length = jnp.sign(diff) * jnp.min(
                jnp.array([jnp.abs(diff), jnp.abs(integrator_state.dt)])
            )

            integrator_state.dt = step_length
            system_state, integrator_state, _ = ias15_step(
                system_state, acceleration_func, integrator_state, step_scheduler
            )
            return system_state, integrator_state, last_meaningful_dt, iter_num + 1

        def cond_func(args: tuple) -> bool:
            system_state, integrator_state, _last_meaningful_dt, iter_num = args
            t = system_state.time

            step_length = jnp.sign(final_time - t) * jnp.min(
                jnp.array([jnp.abs(final_time - t), jnp.abs(integrator_state.dt)])
            )
            return (step_length != 0) & (iter_num < 10_000)

        final_system_state, final_integrator_state, _last_meaningful_dt, iter_num = (
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

        return (final_system_state, final_integrator_state, iter_num)

    def scan_func(carry: tuple, scan_over: float) -> tuple:
        # jax.debug.print(
        #     "\nattempting jump to next time: {x}. the current time is: {y}",
        #     x=scan_over,
        #     y=carry[0].time,
        # )
        system_state, integrator_state, steps_so_far = carry
        final_time = scan_over
        system_state, integrator_state, new_steps = evolve(
            system_state, acceleration_func, final_time, integrator_state
        )
        return (system_state, integrator_state, steps_so_far + new_steps), (
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

    (final_system_state, final_integrator_state, tot_steps), (positions, velocities) = (
        jax.lax.scan(
            scan_func, (initial_system_state, initial_integrator_state, 0), times
        )
    )
    return positions, velocities, final_system_state, final_integrator_state, tot_steps


@jax.jit
def ias15_evolve_with_dense_output(
    initial_system_state: SystemState,
    acceleration_func: Callable[[SystemState], jnp.ndarray],
    times: jnp.ndarray,
    initial_integrator_state: IAS15IntegratorState,
    step_scheduler: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, float, jnp.ndarray, jnp.ndarray], float
    ],
) -> tuple:
    """Evolve a system, returning interpolated states plus the underlying dense-output buffers.

    Same integration logic as :func:`ias15_evolve`, but in addition to the interpolated
    positions and velocities at ``times`` it returns the converged 7th-order b coefficients
    plus the start-of-step state for every step. Callers that want to do their own
    polynomial evaluation (e.g. richer light-travel-time correction in :func:`on_sky` via
    :func:`make_ltt_propagator`) should use this function instead of
    :func:`ias15_evolve`.

    Returns:
        tuple:
            ``(positions, velocities, final_system_state, final_integrator_state,
            iter_num, b_buf, a0_buf, x0_buf, v0_buf, dts_buf, t_step_starts,
            step_indices, h_values)``.
            ``b_buf`` has shape ``(IAS15_MAX_DYNAMIC_STEPS, 7, n_particles, 3)``;
            ``a0_buf, x0_buf, v0_buf`` have shape
            ``(IAS15_MAX_DYNAMIC_STEPS, n_particles, 3)``; ``dts_buf`` and
            ``t_step_starts`` have shape ``(IAS15_MAX_DYNAMIC_STEPS,)``;
            ``step_indices`` and ``h_values`` have shape ``(len(times),)``.
    """
    # Body shared with ias15_evolve.
    return _ias15_evolve_core(
        initial_system_state,
        acceleration_func,
        times,
        initial_integrator_state,
        step_scheduler,
    )


def _ias15_evolve_core(
    initial_system_state: SystemState,
    acceleration_func: Callable[[SystemState], jnp.ndarray],
    times: jnp.ndarray,
    initial_integrator_state: IAS15IntegratorState,
    step_scheduler: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, float, jnp.ndarray, jnp.ndarray], float
    ],
) -> tuple:
    """Internal: full ias15_evolve implementation, returning interpolated states *and* dense output.

    Drives the adaptive IAS15 loop, populates the per-step dense-output buffers, and
    interpolates positions/velocities at ``times``. Public callers should use
    :func:`ias15_evolve` (compact return) or :func:`ias15_evolve_with_dense_output`
    (full return).

    .. warning::
       The dense-output buffer is sized by ``IAS15_MAX_DYNAMIC_STEPS`` (15000 by
       default). Integrations requiring more accepted steps will silently truncate,
       with all query times beyond the truncation returning the last captured step's
       polynomial value. For safety the loop also caps total iterations (including
       rejected steps) at ``4 * IAS15_MAX_DYNAMIC_STEPS``.

    Args:
        initial_system_state (SystemState):
            The initial state of the system.
        acceleration_func (Callable[[SystemState], jnp.ndarray]):
            The acceleration function to use.
        times (jnp.ndarray):
            Times at which to return interpolated positions and velocities. Must be
            within [initial_system_state.time, t_end_of_last_natural_step].
        initial_integrator_state (IAS15IntegratorState):
            The initial state of the integrator.
        step_scheduler (Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, float, jnp.ndarray, jnp.ndarray], float]):
            The step scheduler function to use for determining the next proposed
            step size.

    Returns:
        tuple:
            ``(positions, velocities, final_system_state, final_integrator_state,
            iter_num, b_buf, a0_buf, x0_buf, v0_buf, dts_buf, t_step_starts,
            step_indices, h_values)``.
    """
    n_particles = initial_integrator_state.a0.shape[0]

    # Seed buffer index 0 with the initial state so that a zero-length integration
    # (final_time == t0) still yields a valid interpolation: with dts_buf[0]=1e30
    # the only reachable query is t == t0, which lands at h=0 where the
    # polynomial collapses to (x0, v0) regardless of b. For non-degenerate
    # integrations the first accepted step overwrites index 0 with the same
    # x0/v0/a0 it sees as the start-of-step state, so this seeding is invisible.
    x0_initial = jnp.concatenate(
        (initial_system_state.massive_positions, initial_system_state.tracer_positions)
    )
    v0_initial = jnp.concatenate(
        (
            initial_system_state.massive_velocities,
            initial_system_state.tracer_velocities,
        )
    )

    b_buf = jnp.zeros((IAS15_MAX_DYNAMIC_STEPS, 7, n_particles, 3))
    a0_buf = (
        jnp.zeros((IAS15_MAX_DYNAMIC_STEPS, n_particles, 3))
        .at[0]
        .set(initial_integrator_state.a0)
    )
    x0_buf = jnp.zeros((IAS15_MAX_DYNAMIC_STEPS, n_particles, 3)).at[0].set(x0_initial)
    v0_buf = jnp.zeros((IAS15_MAX_DYNAMIC_STEPS, n_particles, 3)).at[0].set(v0_initial)
    # Trailing (unfilled) dts are a huge sentinel so cumulative t_step_starts past
    # the valid prefix exceed any query time; searchsorted then safely routes all
    # valid queries into the accepted-step prefix.
    dts_buf = jnp.full((IAS15_MAX_DYNAMIC_STEPS,), 1e30)

    t0 = initial_system_state.time
    final_time = jnp.max(times)
    direction = jnp.sign(final_time - t0)

    def cond_fn(carry: tuple) -> bool:
        system_state, _ig, _b, _a0, _x0, _v0, _dts, n_accepted, iter_num = carry
        t = system_state.time
        # Non-strict on `direction` so that direction == 0 (final_time == t0)
        # short-circuits past_final to True at iter 0, skipping the loop body
        # entirely. For direction != 0 only one disjunct is active and the
        # (t >= final_time) / (t <= final_time) checks are unchanged.
        past_final = ((direction >= 0) & (t >= final_time)) | (
            (direction <= 0) & (t <= final_time)
        )
        return (
            (~past_final)
            & (n_accepted < IAS15_MAX_DYNAMIC_STEPS)
            & (iter_num < 4 * IAS15_MAX_DYNAMIC_STEPS)
        )

    def body_fn(carry: tuple) -> tuple:
        (
            system_state,
            integrator_state,
            b_buf,
            a0_buf,
            x0_buf,
            v0_buf,
            dts_buf,
            n_accepted,
            iter_num,
        ) = carry

        x0_start = jnp.concatenate(
            (system_state.massive_positions, system_state.tracer_positions)
        )
        v0_start = jnp.concatenate(
            (system_state.massive_velocities, system_state.tracer_velocities)
        )
        a0_start = integrator_state.a0

        integrator_state.dt = direction * jnp.abs(integrator_state.dt)
        new_system_state, new_integrator_state, converged_b = ias15_step(
            system_state, acceleration_func, integrator_state, step_scheduler
        )

        accepted = new_integrator_state.dt_last_done != 0.0

        def write(buf_state: tuple) -> tuple:
            b_buf, a0_buf, x0_buf, v0_buf, dts_buf = buf_state
            b_buf = b_buf.at[n_accepted].set(converged_b)
            a0_buf = a0_buf.at[n_accepted].set(a0_start)
            x0_buf = x0_buf.at[n_accepted].set(x0_start)
            v0_buf = v0_buf.at[n_accepted].set(v0_start)
            dts_buf = dts_buf.at[n_accepted].set(new_integrator_state.dt_last_done)
            return (b_buf, a0_buf, x0_buf, v0_buf, dts_buf)

        def skip(buf_state: tuple) -> tuple:
            return buf_state

        b_buf, a0_buf, x0_buf, v0_buf, dts_buf = jax.lax.cond(
            accepted, write, skip, (b_buf, a0_buf, x0_buf, v0_buf, dts_buf)
        )

        n_accepted = n_accepted + jnp.where(accepted, 1, 0)
        return (
            new_system_state,
            new_integrator_state,
            b_buf,
            a0_buf,
            x0_buf,
            v0_buf,
            dts_buf,
            n_accepted,
            iter_num + 1,
        )

    init_carry = (
        initial_system_state,
        initial_integrator_state,
        b_buf,
        a0_buf,
        x0_buf,
        v0_buf,
        dts_buf,
        0,
        0,
    )
    (
        final_system_state,
        final_integrator_state,
        b_buf,
        a0_buf,
        x0_buf,
        v0_buf,
        dts_buf,
        _n_accepted,
        iter_num,
    ) = jax.lax.while_loop(cond_fn, body_fn, init_carry)

    t_step_starts = t0 + jnp.concatenate([jnp.array([0.0]), jnp.cumsum(dts_buf[:-1])])
    step_indices, h_values = precompute_interpolation_indices(
        t_step_starts, dts_buf, times
    )
    # Safety rail: in case of floating-point drift at the boundary.
    h_values = jnp.clip(h_values, 0.0, 1.0)
    positions, velocities = interpolate_from_dense_output(
        b_buf, a0_buf, x0_buf, v0_buf, dts_buf, step_indices, h_values
    )

    return (
        positions,
        velocities,
        final_system_state,
        final_integrator_state,
        iter_num,
        b_buf,
        a0_buf,
        x0_buf,
        v0_buf,
        dts_buf,
        t_step_starts,
        step_indices,
        h_values,
    )


@jax.jit
def ias15_evolve(
    initial_system_state: SystemState,
    acceleration_func: Callable[[SystemState], jnp.ndarray],
    times: jnp.ndarray,
    initial_integrator_state: IAS15IntegratorState,
    step_scheduler: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, float, jnp.ndarray, jnp.ndarray], float
    ],
) -> tuple[jnp.ndarray, jnp.ndarray, SystemState, IAS15IntegratorState, jnp.ndarray]:
    """Evolve a system and recover positions/velocities at ``times`` via interpolation.

    Takes natural adaptive IAS15 steps from the initial time past ``jnp.max(times)``,
    stores the per-step dense output (converged 7th-order b coefficients plus start-
    of-step acceleration/position/velocity) in a pre-allocated buffer, then evaluates
    the polynomial at each entry of ``times``. This matches the approach used by
    ASSIST/REBOUND and avoids the small final jumps that forced-landing integration
    is prone to.

    Supports forward-mode AD only (``jax.lax.while_loop`` has no reverse-mode rule).

    Args:
        initial_system_state (SystemState):
            The initial state of the system.
        acceleration_func (Callable[[SystemState], jnp.ndarray]):
            The acceleration function to use.
        times (jnp.ndarray):
            Times at which to return interpolated positions and velocities. Must be
            within ``[initial_system_state.time, t_end_of_last_natural_step]``.
        initial_integrator_state (IAS15IntegratorState):
            The initial state of the integrator.
        step_scheduler (Callable):
            The step scheduler function to use for determining the next proposed
            step size.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, SystemState, IAS15IntegratorState, jnp.ndarray]:
            Interpolated positions and velocities at ``times``, the final system
            state, the final integrator state, and the iteration count.
    """
    (
        positions,
        velocities,
        final_system_state,
        final_integrator_state,
        iter_num,
        *_dense,
    ) = _ias15_evolve_core(
        initial_system_state,
        acceleration_func,
        times,
        initial_integrator_state,
        step_scheduler,
    )
    return positions, velocities, final_system_state, final_integrator_state, iter_num

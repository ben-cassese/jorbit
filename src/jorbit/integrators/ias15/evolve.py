"""The driving loops that march the IAS15 integrator across requested times."""

import jax

jax.config.update("jax_enable_x64", True)

from collections.abc import Callable
from functools import partial

import jax.numpy as jnp

from jorbit.integrators.ias15.interpolation import (
    interpolate_from_dense_output,
    precompute_interpolation_indices,
)
from jorbit.integrators.ias15.step import ias15_step
from jorbit.utils.states import IAS15IntegratorState, SystemState

# Maximum number of accepted adaptive steps captured by ias15_evolve's dense-output
# buffer. ASSIST GLOBAL+min_dt=0.001 takes ~2.1k steps for the 2029 Apophis flyby
# year; jorbit's port currently takes ~13k under the same recipe (sits at the floor
# longer than ASSIST due to PC/numerical b6 differences out of scope). 15000 leaves
# headroom for slightly tighter encounters and matches that envelope.
IAS15_MAX_DYNAMIC_STEPS = 15_000


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

            t = system_state.relative_time

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
            t = system_state.relative_time

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


@partial(jax.jit, static_argnames=["max_steps"])
def ias15_evolve_with_dense_output(
    initial_system_state: SystemState,
    acceleration_func: Callable[[SystemState], jnp.ndarray],
    times: jnp.ndarray,
    initial_integrator_state: IAS15IntegratorState,
    step_scheduler: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, float, jnp.ndarray, jnp.ndarray], float
    ],
    max_steps: int | None = None,
) -> tuple:
    """Evolve a system, returning interpolated states plus the underlying dense-output buffers.

    Same integration logic as :func:`ias15_evolve`, but in addition to the interpolated
    positions and velocities at ``times`` it returns the converged 7th-order b coefficients
    plus the start-of-step state for every step. Callers that want to do their own
    polynomial evaluation (e.g. richer light-travel-time correction in :func:`on_sky` via
    :func:`make_ltt_propagator`) should use this function instead of
    :func:`ias15_evolve`.

    Args:
        initial_system_state (SystemState):
            The initial state of the system.
        acceleration_func (Callable[[SystemState], jnp.ndarray]):
            The acceleration function to use.
        times (jnp.ndarray):
            Times at which to return interpolated positions and velocities. Must be
            within [initial_system_state.relative_time, t_end_of_last_natural_step].
        initial_integrator_state (IAS15IntegratorState):
            The initial state of the integrator.
        step_scheduler (Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, float, jnp.ndarray, jnp.ndarray], float]):
            The step scheduler function to use for determining the next proposed
            step size.
        max_steps (int | None):
            Depth of the dense-output buffer, i.e. the maximum number of accepted
            adaptive steps this call can capture (a static argument; changing it
            triggers recompilation). ``None`` (default) uses
            ``IAS15_MAX_DYNAMIC_STEPS`` (15000). Arcs needing only a few steps can
            pass a much smaller value to avoid allocating and zeroing the full-size
            buffers. Integrations needing more accepted steps than ``max_steps``
            truncate; the final system state reports how far the integration
            actually reached.

    Returns:
        tuple:
            ``(positions, velocities, final_system_state, final_integrator_state,
            iter_num, b_buf, a0_buf, x0_buf, v0_buf, dts_buf, t_step_starts,
            step_indices, h_values)``.
            ``b_buf`` has shape ``(max_steps, 7, n_particles, 3)``;
            ``a0_buf, x0_buf, v0_buf`` have shape
            ``(max_steps, n_particles, 3)``; ``dts_buf`` and
            ``t_step_starts`` have shape ``(max_steps,)``;
            ``step_indices`` and ``h_values`` have shape ``(len(times),)``.
    """
    # Body shared with ias15_evolve.
    return _ias15_evolve_core(
        initial_system_state,
        acceleration_func,
        times,
        initial_integrator_state,
        step_scheduler,
        max_steps,
    )


def _ias15_evolve_core(
    initial_system_state: SystemState,
    acceleration_func: Callable[[SystemState], jnp.ndarray],
    times: jnp.ndarray,
    initial_integrator_state: IAS15IntegratorState,
    step_scheduler: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, float, jnp.ndarray, jnp.ndarray], float
    ],
    max_steps: int | None = None,
) -> tuple:
    """Internal: full ias15_evolve implementation, returning interpolated states *and* dense output.

    Drives the adaptive IAS15 loop, populates the per-step dense-output buffers, and
    interpolates positions/velocities at ``times``. Public callers should use
    :func:`ias15_evolve` (compact return) or :func:`ias15_evolve_with_dense_output`
    (full return).

    .. warning::
       The dense-output buffer is sized by ``max_steps`` (``IAS15_MAX_DYNAMIC_STEPS``,
       15000, when None). Integrations requiring more accepted steps will silently
       truncate, with all query times beyond the truncation returning the last
       captured step's polynomial value. For safety the loop also caps total
       iterations (including rejected steps) at ``4 * max_steps``.

    Args:
        initial_system_state (SystemState):
            The initial state of the system.
        acceleration_func (Callable[[SystemState], jnp.ndarray]):
            The acceleration function to use.
        times (jnp.ndarray):
            Times at which to return interpolated positions and velocities. Must be
            within [initial_system_state.relative_time, t_end_of_last_natural_step].
        initial_integrator_state (IAS15IntegratorState):
            The initial state of the integrator.
        step_scheduler (Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, float, jnp.ndarray, jnp.ndarray], float]):
            The step scheduler function to use for determining the next proposed
            step size.
        max_steps (int | None):
            Dense-output buffer depth (see :func:`ias15_evolve_with_dense_output`).
            Resolved to the module-level ``IAS15_MAX_DYNAMIC_STEPS`` at trace time
            when None.

    Returns:
        tuple:
            ``(positions, velocities, final_system_state, final_integrator_state,
            iter_num, b_buf, a0_buf, x0_buf, v0_buf, dts_buf, t_step_starts,
            step_indices, h_values)``.
    """
    if max_steps is None:
        max_steps = IAS15_MAX_DYNAMIC_STEPS
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

    b_buf = jnp.zeros((max_steps, 7, n_particles, 3))
    a0_buf = (
        jnp.zeros((max_steps, n_particles, 3)).at[0].set(initial_integrator_state.a0)
    )
    x0_buf = jnp.zeros((max_steps, n_particles, 3)).at[0].set(x0_initial)
    v0_buf = jnp.zeros((max_steps, n_particles, 3)).at[0].set(v0_initial)
    # Trailing (unfilled) dts are a huge sentinel so cumulative t_step_starts past
    # the valid prefix exceed any query time; searchsorted then safely routes all
    # valid queries into the accepted-step prefix.
    dts_buf = jnp.full((max_steps,), 1e30)

    t0 = initial_system_state.relative_time
    max_t = jnp.max(times)
    min_t = jnp.min(times)
    final_time = jnp.where(jnp.abs(max_t - t0) >= jnp.abs(min_t - t0), max_t, min_t)

    direction = jnp.sign(final_time - t0)

    def cond_fn(carry: tuple) -> bool:
        system_state, _ig, _b, _a0, _x0, _v0, _dts, n_accepted, iter_num = carry
        t = system_state.relative_time
        # Non-strict on `direction` so that direction == 0 (final_time == t0)
        # short-circuits past_final to True at iter 0, skipping the loop body
        # entirely. For direction != 0 only one disjunct is active and the
        # (t >= final_time) / (t <= final_time) checks are unchanged.
        past_final = ((direction >= 0) & (t >= final_time)) | (
            (direction <= 0) & (t <= final_time)
        )
        return (~past_final) & (n_accepted < max_steps) & (iter_num < 4 * max_steps)

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


@partial(jax.jit, static_argnames=["max_steps"])
def ias15_evolve(
    initial_system_state: SystemState,
    acceleration_func: Callable[[SystemState], jnp.ndarray],
    times: jnp.ndarray,
    initial_integrator_state: IAS15IntegratorState,
    step_scheduler: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, float, jnp.ndarray, jnp.ndarray], float
    ],
    max_steps: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, SystemState, IAS15IntegratorState, jnp.ndarray]:
    """Evolve a system and recover positions/velocities at ``times`` via interpolation.

    Takes natural adaptive IAS15 steps from the initial time past ``jnp.max(times)``,
    stores the per-step dense output (converged 7th-order b coefficients plus start-
    of-step acceleration/position/velocity) in a pre-allocated buffer, then evaluates
    the polynomial at each entry of ``times``. This matches the approach used by
    ASSIST/REBOUND and avoids the small final jumps that forced-landing integration
    is prone to.

    Supports forward-mode AD only (``jax.lax.while_loop`` has no reverse-mode rule).
    Also, because the step sequence is chosen adaptively from the state, an
    infinitesimal change in the initial state can change the realized number and
    placement of steps: the outputs (and any scalar built from them, e.g. a
    likelihood) are only piecewise-smooth in the initial state, with
    discontinuities at roughly the integrator tolerance. Use forward-mode AD for
    derivatives — it differentiates the realized step sequence and is exact for
    the function actually evaluated. Finite differences straddle the
    discontinuities and silently return garbage whenever the true derivative is
    small (weakly-constrained directions).

    Args:
        initial_system_state (SystemState):
            The initial state of the system.
        acceleration_func (Callable[[SystemState], jnp.ndarray]):
            The acceleration function to use.
        times (jnp.ndarray):
            Times at which to return interpolated positions and velocities. Must be
            within ``[initial_system_state.relative_time, t_end_of_last_natural_step]``.
        initial_integrator_state (IAS15IntegratorState):
            The initial state of the integrator.
        step_scheduler (Callable):
            The step scheduler function to use for determining the next proposed
            step size.
        max_steps (int | None):
            Dense-output buffer depth, i.e. the maximum number of accepted adaptive
            steps (a static argument; changing it triggers recompilation). ``None``
            (default) uses ``IAS15_MAX_DYNAMIC_STEPS`` (15000). See
            :func:`ias15_evolve_with_dense_output`.

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
        max_steps,
    )
    return positions, velocities, final_system_state, final_integrator_state, iter_num

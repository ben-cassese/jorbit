"""Front-end orchestration that keeps IAS15 integrations off the backend buffers.

The IAS15 backends in :mod:`jorbit.integrators.ias15` are deliberately hard-capped so
that their buffers stay bounded and JIT-friendly:

- the interpolation path (:func:`ias15_evolve` / :func:`ias15_evolve_with_dense_output`)
  stores per-step dense output in a buffer of ``IAS15_MAX_DYNAMIC_STEPS`` (15000) steps,
  and
- the forced-landing path (:func:`ias15_evolve_forced_landing`) caps the number of
  steps *between* consecutive requested times at 10000.

Past those caps the backends silently truncate. Keeping the caps is the right call for
the bounded JIT kernels, but the public ``Particle``/``System`` methods should never
hand a user a silently-truncated answer. This module sits on top of the (unchanged)
backends and does extra, data-dependent work on the host so that the nominal public
methods are truncation-proof:

- :func:`stitched_per_query_gather` stitches successive interpolation chunks together,
  carrying the integrator state forward so the result is bit-identical to a single run
  with a larger buffer.
- :func:`budgeted_forced_landing` detects a truncating forced-landing run and inserts
  "dummy" landing times (dropped from the output afterward) so no single interval
  exceeds the backend's per-interval cap. This mirrors the existing
  :func:`jorbit.integrators.create_leapfrog_times` "expand then select" pattern.

None of these helpers is JIT'd: the public methods that call them already produce host
objects (e.g. ``SkyCoord``), so a short host-side loop over JIT'd backend calls is fine.
The common case is a single backend call plus one cheap scalar check.
"""

from collections.abc import Callable, Iterator

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.integrators.ias15 import (
    ias15_evolve,
    ias15_evolve_forced_landing,
    ias15_evolve_with_dense_output,
    interpolate_from_dense_output,
)
from jorbit.utils.states import IAS15IntegratorState, SystemState

# Tolerance (days) for "did the integrator reach this time" comparisons. ~0.1 ms, far
# below any meaningful step size, so it only absorbs floating-point round-trips.
_TIME_TOL = 1e-9

# Natural-step budget per forced-landing interval. The backend caps a single interval at
# 10000 *iterations* (accepted + rejected); we budget on accepted natural steps and keep
# generous headroom for the occasional rejected step plus the clamp step each dummy adds.
FORCED_LANDING_STEP_BUDGET = 8000

# Sentinel value used by the backend to fill unused dense-output slots (see ias15.py).
_DTS_SENTINEL = 1e29


def _iterate_evolve_chunks(
    initial_system_state: SystemState,
    acceleration_func: Callable,
    times: jnp.ndarray,
    initial_integrator_state: IAS15IntegratorState,
    step_scheduler: Callable,
) -> Iterator[tuple]:
    """Yield successive dense-output chunks until the integration reaches ``max(times)``.

    Each chunk is a full :func:`ias15_evolve_with_dense_output` run (the full ``times``
    array is passed every chunk, so the kernel compiles once and is reused). The final
    state/integrator state of each chunk seed the next, which continues the adaptive
    sequence bit-identically. Raises if a chunk fails to make forward progress (e.g. a
    NaN acceleration), so the loop can never spin forever or silently truncate.

    Yields:
        ``(chunk_output, chunk_start, t_reached, direction)`` per chunk, where
        ``chunk_output`` is the full 13-tuple from
        :func:`ias15_evolve_with_dense_output`.
    """
    state = initial_system_state
    integrator_state = initial_integrator_state
    target = float(jnp.max(times))
    t0 = float(state.relative_time)
    direction = 0.0
    chunk_start = t0

    while True:
        out = ias15_evolve_with_dense_output(
            state, acceleration_func, times, integrator_state, step_scheduler
        )
        final_system_state = out[2]
        t_reached = float(final_system_state.relative_time)

        if direction == 0.0:
            direction = 1.0 if (t_reached - t0) >= 0.0 else -1.0

        yield out, chunk_start, t_reached, direction

        # Reached the farthest requested time -> done.
        if direction * (t_reached - target) >= -_TIME_TOL:
            return

        # No forward progress despite not having reached the target -> genuinely stuck.
        if direction * (t_reached - chunk_start) <= _TIME_TOL:
            raise RuntimeError(
                "IAS15 interpolation stitching made no forward progress at "
                f"relative_time={t_reached} (target={target}). The integration may be "
                "stuck (e.g. a NaN acceleration or a degenerate step). This is a "
                "genuine failure, not a buffer truncation."
            )

        state = out[2]
        integrator_state = out[3]
        chunk_start = t_reached


def stitched_per_query_gather(
    initial_system_state: SystemState,
    acceleration_func: Callable,
    times: jnp.ndarray,
    initial_integrator_state: IAS15IntegratorState,
    step_scheduler: Callable,
) -> tuple:
    """Gather per-query dense-output slices across as many chunks as needed (no cap).

    For each requested time this returns the converged 7th-order ``b`` coefficients plus
    the start-of-step ``a0``/``x0``/``v0``, the step length ``dt``, and the fractional
    position ``h`` of the query within its step, drawn from whichever stitched chunk
    actually covers that time. The result is exactly what a single
    :func:`ias15_evolve_with_dense_output` call would return for those times if its
    buffer were large enough, but with no silent truncation.

    Feed the gather to :func:`interpolate_from_dense_output` for positions/velocities, or
    to :func:`jorbit.integrators.make_ltt_propagator` for dense-output ephemerides.

    Returns:
        ``(b_q, a0_q, x0_q, v0_q, dt_q, h_q, total_steps)``. With ``n = len(times)`` and
        ``P`` particles: ``b_q`` is ``(n, 7, P, 3)``; ``a0_q``/``x0_q``/``v0_q`` are
        ``(n, P, 3)``; ``dt_q``/``h_q`` are ``(n,)``; ``total_steps`` is the summed
        iteration count across all chunks.
    """
    n_times = times.shape[0]
    b_q = a0_q = x0_q = v0_q = dt_q = h_q = None
    covered = jnp.zeros(n_times, dtype=bool)
    total_steps = 0

    for out, _chunk_start, t_reached, direction in _iterate_evolve_chunks(
        initial_system_state,
        acceleration_func,
        times,
        initial_integrator_state,
        step_scheduler,
    ):
        (
            _positions,
            _velocities,
            _final_system_state,
            _final_integrator_state,
            iter_num,
            b_buf,
            a0_buf,
            x0_buf,
            v0_buf,
            dts_buf,
            _t_step_starts,
            step_indices,
            h_values,
        ) = out
        total_steps += int(iter_num)

        if b_q is None:
            # Allocate accumulators now that we know the per-step shapes.
            b_q = jnp.zeros((n_times, *b_buf.shape[1:]))
            a0_q = jnp.zeros((n_times, *a0_buf.shape[1:]))
            x0_q = jnp.zeros((n_times, *x0_buf.shape[1:]))
            v0_q = jnp.zeros((n_times, *v0_buf.shape[1:]))
            dt_q = jnp.zeros((n_times,))
            h_q = jnp.zeros((n_times,))

        # Times this chunk newly covers: not yet covered and at/before t_reached in the
        # integration direction. Already-covered times keep their earlier gather; times
        # past t_reached are left for a later chunk.
        reached = direction * (t_reached - times) >= -_TIME_TOL
        newly = (~covered) & reached

        b_chunk = b_buf[step_indices]
        a0_chunk = a0_buf[step_indices]
        x0_chunk = x0_buf[step_indices]
        v0_chunk = v0_buf[step_indices]
        dt_chunk = dts_buf[step_indices]

        b_q = jnp.where(newly[:, None, None, None], b_chunk, b_q)
        a0_q = jnp.where(newly[:, None, None], a0_chunk, a0_q)
        x0_q = jnp.where(newly[:, None, None], x0_chunk, x0_q)
        v0_q = jnp.where(newly[:, None, None], v0_chunk, v0_q)
        dt_q = jnp.where(newly, dt_chunk, dt_q)
        h_q = jnp.where(newly, h_values, h_q)

        covered = covered | newly

    return b_q, a0_q, x0_q, v0_q, dt_q, h_q, total_steps


def stitched_interpolate(
    initial_system_state: SystemState,
    acceleration_func: Callable,
    times: jnp.ndarray,
    initial_integrator_state: IAS15IntegratorState,
    step_scheduler: Callable,
) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    """Interpolated positions/velocities at ``times``, stitched to avoid the 15k buffer.

    Drop-in replacement for the interpolation-path :func:`ias15_evolve` call used by the
    public ``integrate_or_interpolate`` / ``System.integrate`` methods.

    Returns:
        ``(positions, velocities, total_steps)`` with positions/velocities of shape
        ``(len(times), P, 3)``.
    """
    b_q, a0_q, x0_q, v0_q, dt_q, h_q, total_steps = stitched_per_query_gather(
        initial_system_state,
        acceleration_func,
        times,
        initial_integrator_state,
        step_scheduler,
    )
    # interpolate_from_dense_output indexes its buffers by step_indices; the gather is
    # already per-query, so identity indices recover one polynomial evaluation per time.
    identity = jnp.arange(times.shape[0])
    positions, velocities = interpolate_from_dense_output(
        b_q, a0_q, x0_q, v0_q, dt_q, identity, h_q
    )
    return positions, velocities, total_steps


def discover_natural_step_times(
    initial_system_state: SystemState,
    acceleration_func: Callable,
    times: jnp.ndarray,
    initial_integrator_state: IAS15IntegratorState,
    step_scheduler: Callable,
) -> jnp.ndarray:
    """Cumulative end-times of every natural adaptive step from ``t0`` past ``max(times)``.

    Unlike :func:`jorbit.accelerations.static_helpers.get_natural_dynamic_dts` (a slow
    per-step Python loop capped at 10000 steps), this rides the fast JIT'd chunk loop and
    is uncapped. Used to place forced-landing dummy times at natural step boundaries.
    """
    all_dts = []
    for out, _chunk_start, _t_reached, _direction in _iterate_evolve_chunks(
        initial_system_state,
        acceleration_func,
        times,
        initial_integrator_state,
        step_scheduler,
    ):
        dts_buf = out[9]
        # Keep only the filled prefix; unused slots hold the large sentinel.
        valid = dts_buf[dts_buf < _DTS_SENTINEL]
        all_dts.append(valid)

    t0 = initial_system_state.relative_time
    dts = jnp.concatenate(all_dts) if all_dts else jnp.array([])
    return t0 + jnp.cumsum(dts)


def insert_budget_dummy_times(
    natural_step_times: jnp.ndarray,
    requested_times: jnp.ndarray,
    t0: float,
    budget: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Insert dummy landing times so no requested interval holds more than ``budget`` steps.

    Mirrors the contract of :func:`jorbit.integrators.create_leapfrog_times`: returns an
    expanded, ascending time array plus the indices of the original ``requested_times``
    within it. Dummy times are placed at natural step boundaries inside any interval that
    would otherwise exceed ``budget`` natural steps.

    Args:
        natural_step_times: Cumulative natural step end-times from
            :func:`discover_natural_step_times`.
        requested_times: The originally requested output times.
        t0: Integration start time (offset frame).
        budget: Maximum natural steps allowed per (sub-)interval.

    Returns:
        ``(augmented_times, relevant_inds)``.
    """
    nst = jnp.asarray(natural_step_times)
    augmented = []
    relevant_inds = []
    prev = float(t0)

    for tq in [float(t) for t in requested_times]:
        lo, hi = (prev, tq) if tq >= prev else (tq, prev)
        # Natural step boundaries strictly inside this interval, in integration order.
        interior = nst[(nst > lo) & (nst < hi)]
        if tq < prev:
            interior = interior[::-1]
        n = int(interior.shape[0])
        if n > budget:
            for k in range(budget, n, budget):
                augmented.append(float(interior[k]))
        augmented.append(tq)
        relevant_inds.append(len(augmented) - 1)
        prev = tq

    return jnp.array(augmented), jnp.array(relevant_inds, dtype=int)


def budgeted_forced_landing(
    initial_system_state: SystemState,
    acceleration_func: Callable,
    times: jnp.ndarray,
    initial_integrator_state: IAS15IntegratorState,
    step_scheduler: Callable,
) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    """Forced-landing integration that never silently truncates between requested times.

    Runs :func:`ias15_evolve_forced_landing` on ``times``; if any interval truncated (the
    integrator failed to reach the last requested time), discovers the natural step
    structure, inserts dummy landing times so each sub-interval stays under the backend's
    per-interval cap, and re-runs. The dummy times are dropped from the returned arrays.

    Inserting a dummy landing splits one step into two clamped steps, a perturbation of
    the same kind forced-landing already incurs at every requested time and far below the
    mas-level accuracy target; the budget margin absorbs the extra clamp steps.

    Returns:
        ``(positions, velocities, total_steps)`` at the originally requested ``times``.
    """
    positions, velocities, final_system_state, _final_integrator_state, tot_steps = (
        ias15_evolve_forced_landing(
            initial_system_state,
            acceleration_func,
            times,
            initial_integrator_state,
            step_scheduler,
        )
    )

    last_time = float(times[-1])
    t0 = float(initial_system_state.relative_time)
    direction = 1.0 if last_time >= t0 else -1.0
    reached = (
        direction * (float(final_system_state.relative_time) - last_time) >= -_TIME_TOL
    )
    if reached:
        return positions, velocities, int(tot_steps)

    # A run truncated somewhere; targets only advance, so a short final time is a
    # reliable signal. Discover the natural step density and subdivide.
    natural_step_times = discover_natural_step_times(
        initial_system_state,
        acceleration_func,
        times,
        initial_integrator_state,
        step_scheduler,
    )
    augmented_times, relevant_inds = insert_budget_dummy_times(
        natural_step_times, times, t0, FORCED_LANDING_STEP_BUDGET
    )
    positions, velocities, final_system_state, _final_integrator_state, tot_steps = (
        ias15_evolve_forced_landing(
            initial_system_state,
            acceleration_func,
            augmented_times,
            initial_integrator_state,
            step_scheduler,
        )
    )

    last_aug = float(augmented_times[-1])
    if direction * (float(final_system_state.relative_time) - last_aug) < -_TIME_TOL:
        # Essentially unreachable (would require a single ~8000-natural-step sub-interval
        # to still overflow the 10000-iteration cap). Raise rather than silently truncate.
        raise RuntimeError(
            "Forced-landing integration still truncated after inserting dummy landing "
            "times. Try integrate_or_interpolate (interpolation path) instead, or "
            "request more closely spaced output times."
        )
    return positions[relevant_inds], velocities[relevant_inds], int(tot_steps)


def ias15_span_probe(
    initial_system_state: SystemState,
    acceleration_func: Callable,
    times: jnp.ndarray,
    initial_integrator_state: IAS15IntegratorState,
    step_scheduler: Callable,
) -> tuple[bool, int]:
    """Probe a single nominal :func:`ias15_evolve` chunk over ``times``.

    Returns ``(would_truncate, total_steps)`` where ``would_truncate`` is True if one
    dense-output buffer fails to reach ``max(times)``. A cheap forward integration (no
    autodiff) used by the detect-and-raise guards on the covariance-ephemeris and
    likelihood paths, where the host-side stitching loop cannot be threaded through
    ``jax.jacfwd``.
    """
    out = ias15_evolve(
        initial_system_state,
        acceleration_func,
        times,
        initial_integrator_state,
        step_scheduler,
    )
    final_system_state = out[2]
    total_steps = int(out[4])
    target = float(jnp.max(times))
    t0 = float(initial_system_state.relative_time)
    direction = 1.0 if target >= t0 else -1.0
    would_truncate = bool(
        direction * (float(final_system_state.relative_time) - target) < -_TIME_TOL
    )
    return would_truncate, total_steps

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
- :func:`budgeted_forced_landing` detects a truncating forced-landing run -- per
  landing, since a truncated interval leaves the ones after it correct -- and inserts
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

from jorbit.ephemeris.ephemeris_processors import (
    EphemerisPostProcessor,
    EphemerisProcessor,
)
from jorbit.integrators.ias15 import (
    DenseOutput,
    apply_ltt_seed_floor,
    assert_ltt_span_covered,
    ias15_evolve,
    ias15_evolve_with_dense_output,
    interpolate_from_dense_output,
    ltt_backward_times,
    ltt_seed_floor,
    ltt_span_shortfall,
    precompute_interpolation_indices,
)
from jorbit.integrators.ias15.evolve import (
    IAS15_MAX_FORCED_LANDING_ITERS,
    _forced_landing_with_times,
)
from jorbit.integrators.ias15.interpolation import _TIME_TOL
from jorbit.utils.states import IAS15IntegratorState, SystemState

# Natural-step budget per forced-landing interval. The backend caps a single interval at
# IAS15_MAX_FORCED_LANDING_ITERS *iterations* (accepted + rejected); we budget on accepted
# natural steps and keep generous headroom for the occasional rejected step plus the clamp
# step each dummy adds.
FORCED_LANDING_STEP_BUDGET = int(0.8 * IAS15_MAX_FORCED_LANDING_ITERS)

# Sentinel value used by the backend to fill unused dense-output slots (see ias15.py).
_DTS_SENTINEL = 1e29

# J2000 epoch in JD (TDB); EphemerisProcessor.init values are seconds past this.
_J2000_JD = 2451545.0

# Concatenated dense buffers are padded up to a multiple of this, so that the jitted
# consumers see a handful of distinct shapes rather than one per distinct step count.
_BUFFER_QUANTUM = 1024


def _loaded_ephemeris_bounds_jd(
    acceleration_func: Callable,
) -> tuple[float, float] | None:
    """Best-effort ``(start_jd, end_jd)`` of the ephemeris data bound to an acceleration function.

    Walks the ``jax.tree_util.Partial``-bound arguments of ``acceleration_func``
    looking for :class:`EphemerisProcessor` / :class:`EphemerisPostProcessor`
    instances and intersects the time spans their Chebyshev data actually cover
    (the requested ``earliest_time``/``latest_time``, padded by 100 days and snapped
    to interval boundaries by :func:`jorbit.ephemeris.process_bsp.merge_data`).
    Returns None when no ephemeris data is found (e.g. an ephemeris-free custom
    gravity function). Used only to enrich error messages.
    """
    starts: list[float] = []
    ends: list[float] = []

    def visit(obj: object) -> None:
        if isinstance(obj, EphemerisPostProcessor):
            for eph in obj.ephs:
                visit(eph)
        elif isinstance(obj, EphemerisProcessor):
            # Per body the coefficients cover init .. init + n_intervals * intlen
            # (seconds past J2000). merge_data tiles bodies with longer intervals up
            # to a shared interval count, so the intersection across bodies (max of
            # starts, min of ends) recovers the true loaded span.
            starts.append(_J2000_JD + float(jnp.max(obj.init)) / 86400.0)
            ends.append(
                _J2000_JD
                + float(jnp.min(obj.init + obj.coeffs.shape[-1] * obj.intlen)) / 86400.0
            )
        elif isinstance(obj, jax.tree_util.Partial):
            for a in obj.args:
                visit(a)
            for a in obj.keywords.values():
                visit(a)
        elif isinstance(obj, (list, tuple)):
            for a in obj:
                visit(a)

    visit(acceleration_func)
    if not starts:
        return None
    return max(starts), min(ends)


def _no_progress_error(
    t_reached: float,
    target: float,
    direction: float,
    time_reference: float,
    acceleration_func: Callable,
) -> RuntimeError:
    """Build the error for a stitching chunk that made no forward progress.

    A stall at (or a target beyond) the edge of the loaded ephemeris span produces
    the same no-progress signature as a dynamical failure, but the fix is to widen
    the ephemeris window, not to hunt for a NaN. When ephemeris data is reachable
    from ``acceleration_func``, check the bound in the direction of integration
    before blaming the dynamics.
    """
    base = (
        "IAS15 interpolation stitching made no forward progress at "
        f"relative_time={t_reached} (target={target})."
    )
    bounds = _loaded_ephemeris_bounds_jd(acceleration_func)
    if bounds is not None:
        start_jd, end_jd = bounds
        stall_jd = t_reached + time_reference
        target_jd = target + time_reference
        if direction < 0:
            bound_jd, bound_name, widen_arg = start_jd, "earliest", "earliest_time"
            # 1-day cushion: the last accepted step can end slightly inside the
            # bound before the garbage accelerations stall the stepper.
            off_window = min(stall_jd, target_jd) < bound_jd + 1.0
        else:
            bound_jd, bound_name, widen_arg = end_jd, "latest", "latest_time"
            off_window = max(stall_jd, target_jd) > bound_jd - 1.0
        if off_window:
            return RuntimeError(
                f"{base} The integration stalled at JD {stall_jd:.2f} (TDB) while "
                f"targeting JD {target_jd:.2f}, at or beyond the {bound_name} time "
                f"loaded into the ephemeris (JD {bound_jd:.2f}; loaded span "
                f"JD {start_jd:.2f} to JD {end_jd:.2f}). This looks like the edge of "
                "the loaded ephemeris window, not a dynamical failure: rebuild the "
                f"Particle/System/Ephemeris with `{widen_arg}` covering the "
                "requested times to widen the window."
            )
    return RuntimeError(
        f"{base} The integration may be stuck (e.g. a NaN acceleration or a "
        "degenerate step). This is a genuine failure, not a buffer truncation."
    )


def _iterate_evolve_chunks(
    initial_system_state: SystemState,
    acceleration_func: Callable,
    times: jnp.ndarray,
    initial_integrator_state: IAS15IntegratorState,
    step_scheduler: Callable,
    max_steps: int | None = None,
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
    t0 = float(state.relative_time)
    max_t = float(jnp.max(times))
    min_t = float(jnp.min(times))
    target = max_t if abs(max_t - t0) >= abs(min_t - t0) else min_t
    is_forward = target >= t0
    direction = 0.0
    chunk_start = t0

    while True:
        # Re-clamp times to the current state so _ias15_evolve_core's direction
        # inference stays correct across multi-chunk stitching passes.
        cur_t0 = float(state.relative_time)
        current_times = (
            jnp.maximum(times, cur_t0) if is_forward else jnp.minimum(times, cur_t0)
        )
        out = ias15_evolve_with_dense_output(
            state,
            acceleration_func,
            current_times,
            integrator_state,
            step_scheduler,
            max_steps,
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
            raise _no_progress_error(
                t_reached,
                target,
                direction,
                float(final_system_state.time_reference),
                acceleration_func,
            )

        state = out[2]
        integrator_state = out[3]
        chunk_start = t_reached


def _direction_buffers(
    initial_system_state: SystemState,
    acceleration_func: Callable,
    chunk_times: jnp.ndarray,
    initial_integrator_state: IAS15IntegratorState,
    step_scheduler: Callable,
    max_steps: int | None,
) -> tuple[DenseOutput, int]:
    """Concatenate the accepted steps of every stitched chunk of one directional pass.

    Stitching continues the adaptive sequence bit-identically, so pasting each chunk's
    filled prefix end to end yields exactly the step sequence a single run with a bigger
    buffer would have produced: ``t_step_starts`` stays monotone and
    :func:`precompute_interpolation_indices` can search it directly.
    """
    parts: list[list[jnp.ndarray]] = [[] for _ in range(6)]
    total_steps = 0

    for out, _chunk_start, _t_reached, _direction in _iterate_evolve_chunks(
        initial_system_state,
        acceleration_func,
        chunk_times,
        initial_integrator_state,
        step_scheduler,
        max_steps,
    ):
        iter_num = out[4]
        total_steps += int(iter_num)
        dts_buf = out[9]
        # Accepted steps only. A zero-span pass accepts none; keep slot 0, which the
        # backend seeds with the initial state and a sentinel dt, so a query there
        # collapses to the Taylor expansion about the epoch.
        n = max(int(jnp.sum(jnp.abs(dts_buf) < _DTS_SENTINEL)), 1)
        for part, buf in zip(parts, out[5:11], strict=True):
            part.append(buf[:n])

    b, a0, x0, v0, dts, t_step_starts = (jnp.concatenate(part) for part in parts)

    n_pad = -b.shape[0] % _BUFFER_QUANTUM
    if n_pad:
        pad_state = jnp.zeros((n_pad, *a0.shape[1:]))
        b = jnp.concatenate((b, jnp.zeros((n_pad, *b.shape[1:]))))
        a0 = jnp.concatenate((a0, pad_state))
        x0 = jnp.concatenate((x0, pad_state))
        v0 = jnp.concatenate((v0, pad_state))
        dts = jnp.concatenate((dts, jnp.full((n_pad,), 1e30)))
        t_step_starts = jnp.concatenate((t_step_starts, jnp.zeros((n_pad,))))

    return DenseOutput(b, a0, x0, v0, dts, t_step_starts), total_steps


def stitched_dense_buffers(
    initial_system_state: SystemState,
    acceleration_func: Callable,
    times: jnp.ndarray,
    initial_integrator_state: IAS15IntegratorState,
    step_scheduler: Callable,
    max_steps: int | None = None,
    backward_pad: float = 0.0,
    obs_times: jnp.ndarray | None = None,
    observer_positions: jnp.ndarray | None = None,
) -> tuple[DenseOutput, DenseOutput, int]:
    """Dense output covering ``times``, as one buffer per integration direction (no cap).

    Stitches as many chunks as the span requires and hands back the concatenated per-step
    polynomial data, rather than per-query slices of it. Consumers that need to evaluate the
    trajectory at times they cannot know in advance -- the light-travel-time correction,
    whose retarded times depend on the state -- need the whole buffer.

    Both directions are always integrated, even when every requested time is on one side of
    the epoch: a query at the epoch itself has a retarded time *before* it, which only the
    backward pass can cover. ``backward_pad`` (days) extends the backward pass past the
    earliest requested time so those retarded times land inside a real step. An unused
    direction costs one zero-span kernel call.

    **Light-travel-time coverage.** Passing ``observer_positions`` (and ``obs_times``) opts
    into the full dense-LTT contract, and is what every ``interpolate=True`` ephemeris path
    uses. It does three things ``backward_pad`` alone does not:

    1. floors the integrator's first proposed step via :func:`apply_ltt_seed_floor`,
    2. derives the backward pad from :func:`ltt_seed_floor` (added to any explicit
       ``backward_pad``), and
    3. measures the resulting coverage and, if the backward pass still falls short,
       **extends it until it covers** -- see below -- before asserting.

    Without ``observer_positions`` none of that happens and the behaviour is byte-for-byte
    what it was: no seed floor (which would change the first proposed step, and so every
    subsequent one), no derived pad, no coverage check. :func:`stitched_per_query_gather`
    and hence ``Particle.integrate`` rely on that.

    The extension exists because the pad is a guess and the requirement is not knowable
    until a pass exists: :func:`ltt_seed_floor` sizes the pad from the particle's distance at
    the *epoch*, while what has to be covered is the light travel time at each
    *observation*. The new target is the measured retarded time itself rather than a
    multiple of the old pad, because a backward pass stops at the end of the first natural
    step past its target and so already covers the pad *plus* that overshoot; sizing an
    extension from the pad alone can land inside the existing buffer and accomplish nothing.
    Since the adaptive steps are natural (the target only ends the loop, it never clamps a
    step), the extended buffer is a prefix-identical superset of the original and every
    already-covered observation keeps a bitwise identical position.

    Args:
        initial_system_state (SystemState): State at the integration epoch.
        acceleration_func (Callable): The system's acceleration function.
        times (jnp.ndarray): Times the buffers must cover, shape (n_times,).
        initial_integrator_state (IAS15IntegratorState): Starting integrator state.
        step_scheduler (Callable): The adaptive step-size controller.
        max_steps (int | None): Per-chunk dense-output buffer depth (None uses
            ``IAS15_MAX_DYNAMIC_STEPS``); smaller buffers just mean more chunks.
        backward_pad (float): Days by which to extend the backward pass past the earliest
            requested time (and past the epoch when nothing precedes it).
        obs_times (jnp.ndarray | None): The subset of ``times`` that are real observations,
            shape (n_obs,). Defaults to all of ``times``. Only used with
            ``observer_positions``.
        observer_positions (jnp.ndarray | None): Observer position at each observation time,
            shape (n_obs, 3). Opts into the light-travel-time contract described above.

    Returns:
        tuple[DenseOutput, DenseOutput, int]:
            The forward buffers, the backward buffers, and the summed iteration count. The
            count includes any discarded backward pass, since it is a measure of work done.
    """
    t0 = float(initial_system_state.relative_time)

    pad = backward_pad
    if observer_positions is not None:
        if obs_times is None:
            obs_times = times
        positions = jnp.concatenate(
            (
                initial_system_state.massive_positions,
                initial_system_state.tracer_positions,
            )
        )
        initial_integrator_state = apply_ltt_seed_floor(
            initial_integrator_state, positions, observer_positions
        )
        pad = pad + float(ltt_seed_floor(positions, observer_positions))

    fwd, fwd_steps = _direction_buffers(
        initial_system_state,
        acceleration_func,
        jnp.maximum(times, t0),
        initial_integrator_state,
        step_scheduler,
        max_steps,
    )
    bwd, bwd_steps = _direction_buffers(
        initial_system_state,
        acceleration_func,
        ltt_backward_times(times, t0, pad),
        initial_integrator_state,
        step_scheduler,
        max_steps,
    )

    if observer_positions is not None:
        retarded_min, shortfall = ltt_span_shortfall(
            fwd, bwd, t0, obs_times, observer_positions
        )
        if shortfall > 0.0:
            # Target the measured retarded time absolutely. ltt_span_shortfall already
            # inflates its light travel time to cover on_sky's converged value, so the only
            # margin needed here is float slack on the summation.
            new_pad = (
                float(jnp.min(jnp.minimum(times, t0))) - retarded_min + 10.0 * _TIME_TOL
            )
            # ponytail: rebuilds the backward pass from the epoch rather than continuing the
            # existing one from its final state. _iterate_evolve_chunks already continues
            # the adaptive sequence bit-identically, so a true continuation is possible and
            # would cost nothing -- it needs _direction_buffers to return its final
            # state/integrator state and the _BUFFER_QUANTUM padding moved out to here. Not
            # worth it at the observed ~5e-6 trigger rate; revisit if that rate climbs.
            bwd, extra_steps = _direction_buffers(
                initial_system_state,
                acceleration_func,
                ltt_backward_times(times, t0, new_pad),
                initial_integrator_state,
                step_scheduler,
                max_steps,
            )
            bwd_steps += extra_steps
        # Backstop: after the extension this can only fire on a genuine buffer truncation.
        assert_ltt_span_covered(fwd, bwd, t0, obs_times, observer_positions)

    return fwd, bwd, fwd_steps + bwd_steps


def stitched_per_query_gather(
    initial_system_state: SystemState,
    acceleration_func: Callable,
    times: jnp.ndarray,
    initial_integrator_state: IAS15IntegratorState,
    step_scheduler: Callable,
    max_steps: int | None = None,
) -> tuple:
    """Gather per-query dense-output slices across as many chunks as needed (no cap).

    For each requested time this returns the converged 7th-order ``b`` coefficients plus
    the start-of-step ``a0``/``x0``/``v0``, the step length ``dt``, and the fractional
    position ``h`` of the query within its step, drawn from whichever direction's
    stitched buffer (see :func:`stitched_dense_buffers`) covers that time. The result is
    exactly what a single :func:`ias15_evolve_with_dense_output` call would return for
    those times if its buffer were large enough, but with no silent truncation.

    Feed the gather to :func:`interpolate_from_dense_output` for positions/velocities.
    The light-travel-time paths need :func:`stitched_dense_buffers` directly, since a
    single step per query cannot cover the retarded times.

    Returns:
        ``(b_q, a0_q, x0_q, v0_q, dt_q, h_q, total_steps)``. With ``n = len(times)`` and
        ``P`` particles: ``b_q`` is ``(n, 7, P, 3)``; ``a0_q``/``x0_q``/``v0_q`` are
        ``(n, P, 3)``; ``dt_q``/``h_q`` are ``(n,)``; ``total_steps`` is the summed
        iteration count across all chunks.
    """
    fwd, bwd, total_steps = stitched_dense_buffers(
        initial_system_state,
        acceleration_func,
        times,
        initial_integrator_state,
        step_scheduler,
        max_steps,
    )

    def gathered(dense: DenseOutput) -> tuple:
        idx, h = precompute_interpolation_indices(dense.t_step_starts, dense.dts, times)
        return (
            dense.b[idx],
            dense.a0[idx],
            dense.x0[idx],
            dense.v0[idx],
            dense.dts[idx],
            # Safety rail against floating-point drift at a step boundary, matching
            # _ias15_evolve_core's own clip of the h values it returns.
            jnp.clip(h, 0.0, 1.0),
        )

    is_fwd = times >= float(initial_system_state.relative_time)
    picked = []
    for f_i, b_i in zip(gathered(fwd), gathered(bwd), strict=True):
        mask = is_fwd.reshape((-1,) + (1,) * (f_i.ndim - 1))
        picked.append(jnp.where(mask, f_i, b_i))
    return (*picked, total_steps)


def stitched_interpolate(
    initial_system_state: SystemState,
    acceleration_func: Callable,
    times: jnp.ndarray,
    initial_integrator_state: IAS15IntegratorState,
    step_scheduler: Callable,
    max_steps: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    """Interpolated positions/velocities at ``times``, stitched to avoid the 15k buffer.

    Drop-in replacement for the interpolation-path :func:`ias15_evolve` call used by the
    public ``integrate_or_interpolate`` / ``System.integrate`` methods. ``max_steps``
    sets the per-chunk dense-output buffer depth (see
    :func:`stitched_per_query_gather`).

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
        max_steps,
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
    t0 = float(initial_system_state.relative_time)
    nst_all = []

    for forward_pass in [True, False]:
        pass_mask = times >= t0 if forward_pass else times < t0
        if not jnp.any(pass_mask):
            continue

        chunk_times = jnp.where(pass_mask, times, t0)
        all_dts = []
        for out, _chunk_start, _t_reached, _direction in _iterate_evolve_chunks(
            initial_system_state,
            acceleration_func,
            chunk_times,
            initial_integrator_state,
            step_scheduler,
        ):
            dts_buf = out[9]
            # Keep only the filled prefix; unused slots hold the large sentinel.
            valid = dts_buf[dts_buf < _DTS_SENTINEL]
            all_dts.append(valid)

        if all_dts:
            dts = jnp.concatenate(all_dts)
            nst_all.append(t0 + jnp.cumsum(dts))

    return jnp.concatenate(nst_all) if nst_all else jnp.array([])


def insert_budget_dummy_times(
    natural_step_times: jnp.ndarray,
    requested_times: jnp.ndarray,
    t0: float,
    budget: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Insert dummy landing times so no requested interval holds more than ``budget`` steps.

    Mirrors the contract of :func:`jorbit.integrators.create_leapfrog_times`: returns an
    expanded time array plus the indices of the original ``requested_times`` within it.
    The expansion is in *marching* order, which follows ``requested_times`` and so is not
    ascending when those cross the epoch or are unsorted. Dummy times are placed at
    natural step boundaries inside any interval that would otherwise exceed ``budget``
    natural steps.

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
        interior = nst[(nst > lo) & (nst < hi)]

        interior_sorted = jnp.sort(interior)
        if tq < prev:
            interior_sorted = interior_sorted[::-1]

        n = int(interior_sorted.shape[0])
        if n > budget:
            for k in range(budget, n, budget):
                augmented.append(float(interior_sorted[k]))
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

    Runs the forced-landing backend on ``times``; if any interval truncated, discovers
    the natural step structure, inserts dummy landing times so each sub-interval stays
    under the backend's per-interval cap, and re-runs. The dummy times are dropped from
    the returned arrays.

    Truncation is detected per landing, against the reached times the backend emits, not
    from the final state: a truncated interval stops at a state that is correct for the
    time it stopped at, so every interval after it starts on the true trajectory and
    lands normally. Checking only the last requested time therefore misses a truncation
    anywhere else -- which is exactly what happened to an ascending array straddling the
    epoch, where the whole pre-epoch arc is traversed as the single interval from the
    epoch back to ``times[0]``: that one interval overran the cap, the (post-epoch) last
    time was reached anyway, and the earliest landing came back silently wrong. A single
    requested time was unaffected only because there the last-time check *is* the whole
    check.

    Inserting a dummy landing splits one step into two clamped steps, a perturbation of
    the same kind forced-landing already incurs at every requested time and far below the
    mas-level accuracy target; the budget margin absorbs the extra clamp steps.

    Returns:
        ``(positions, velocities, total_steps)`` at the originally requested ``times``.
    """
    positions, velocities, landing_times, _fss, _fis, tot_steps = (
        _forced_landing_with_times(
            initial_system_state,
            acceleration_func,
            times,
            initial_integrator_state,
            step_scheduler,
        )
    )

    # Forced landing clamps its step to land exactly on each requested time, so an
    # untruncated landing matches bit-for-bit; _TIME_TOL only absorbs round-trip noise.
    if float(jnp.max(jnp.abs(landing_times - times))) <= _TIME_TOL:
        return positions, velocities, int(tot_steps)

    # Some interval truncated. Discover the natural step density and subdivide.
    t0 = float(initial_system_state.relative_time)
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
    positions, velocities, landing_times, _fss, _fis, tot_steps = (
        _forced_landing_with_times(
            initial_system_state,
            acceleration_func,
            augmented_times,
            initial_integrator_state,
            step_scheduler,
        )
    )

    if float(jnp.max(jnp.abs(landing_times - augmented_times))) > _TIME_TOL:
        # Essentially unreachable (would require a single FORCED_LANDING_STEP_BUDGET-step
        # sub-interval to still overflow the iteration cap), or an interval that stalled
        # for some reason other than the cap. Raise rather than silently truncate.
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
    max_steps: int | None = None,
) -> tuple[bool, int]:
    """Probe a single nominal :func:`ias15_evolve` chunk over ``times``.

    Returns ``(would_truncate, total_steps)`` where ``would_truncate`` is True if one
    dense-output buffer (of depth ``max_steps``; None uses
    ``IAS15_MAX_DYNAMIC_STEPS``) fails to reach ``max(times)``. A cheap forward
    integration with no autodiff, for the paths that run inside ``jax.jacfwd`` and so
    cannot be threaded through the host-side stitching loop.

    Called automatically by the detect-and-raise guards in
    ``Particle.ephemeris(uncertainty=True)`` and ``Particle.max_likelihood``. The
    ``System(observations=...)`` forward model is deliberately *not* guarded this way:
    its callables are compiled once and scored on candidate states supplied as
    arguments, so an automatic probe would force a device sync and a full extra
    integration on every likelihood evaluation. ``System.probe_span`` exposes this
    function for those callers to invoke at their own discretion.
    """
    t0 = float(initial_system_state.relative_time)
    would_truncate = False
    total_steps = 0

    for forward_pass in [True, False]:
        pass_mask = times >= t0 if forward_pass else times < t0
        if not jnp.any(pass_mask):
            continue

        chunk_times = jnp.where(pass_mask, times, t0)
        target = (
            float(jnp.max(chunk_times)) if forward_pass else float(jnp.min(chunk_times))
        )
        out = ias15_evolve(
            initial_system_state,
            acceleration_func,
            chunk_times,
            initial_integrator_state,
            step_scheduler,
            max_steps,
        )
        final_system_state = out[2]
        total_steps += int(out[4])
        direction = 1.0 if forward_pass else -1.0
        would_truncate = would_truncate or bool(
            direction * (float(final_system_state.relative_time) - target) < -_TIME_TOL
        )

    return would_truncate, total_steps

"""Dense-output interpolation and light-travel-time utilities for IAS15.

Evaluate the converged 7th-order IAS15 polynomial at arbitrary times within completed
steps, without re-integrating.
"""

import warnings
from typing import NamedTuple

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jorbit.data.constants import INV_SPEED_OF_LIGHT
from jorbit.integrators.ias15.helpers import _estimate_x_v_from_b


class DenseOutput(NamedTuple):
    """The per-step dense output of one IAS15 pass, plus the step start times.

    Bundles what :func:`interpolate_from_dense_output` and :func:`dense_position` need
    to evaluate the trajectory at an arbitrary time inside the integrated span. A
    ``NamedTuple`` is already a JAX pytree, so these buffers travel through ``jit``,
    ``vmap``, and — importantly — ``jax.tree_util.Partial`` as *leaves*: they are
    multi-MB arrays, and capturing them in a Python closure instead would re-embed them
    as constants in every compilation.

    Unfilled trailing slots carry a large positive ``dts`` sentinel (``1e30``);
    :func:`precompute_interpolation_indices` routes queries past them.

    Attributes:
        b (jnp.ndarray): Converged b coefficients, shape (n_steps, 7, n_particles, 3).
        a0 (jnp.ndarray): Start-of-step accelerations, shape (n_steps, n_particles, 3).
        x0 (jnp.ndarray): Start-of-step positions, shape (n_steps, n_particles, 3).
        v0 (jnp.ndarray): Start-of-step velocities, shape (n_steps, n_particles, 3).
        dts (jnp.ndarray): Step lengths, shape (n_steps,). Negative for a backward pass.
        t_step_starts (jnp.ndarray): Start time of each step, shape (n_steps,), in the
            same offset frame as ``SystemState.relative_time``.
    """

    b: jnp.ndarray
    a0: jnp.ndarray
    x0: jnp.ndarray
    v0: jnp.ndarray
    dts: jnp.ndarray
    t_step_starts: jnp.ndarray


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

    Handles both integration directions. ``jnp.searchsorted`` requires an ascending
    sequence, but a backward integration (negative ``dts``) produces a *descending*
    ``t_step_starts``, so the lookup is done in direction-normalized coordinates. Unfilled
    buffer slots carry a large positive ``dts`` sentinel; their key is forced past every
    real step so valid queries always route into the filled prefix.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]:
            step_indices: Integer index of the containing step for each query time,
                shape (n_queries,).
            h_values: Fractional time within each step (0 to 1),
                shape (n_queries,).
    """
    direction = jnp.sign(dts[0])
    filled = jnp.abs(dts) < 1e29
    key = jnp.where(filled, direction * t_step_starts, jnp.inf)
    step_indices = jnp.searchsorted(key, direction * query_times, side="right") - 1
    # A query at the integration epoch (zero-span: every dts is the unfilled sentinel, or
    # a query landing exactly on the first step start) keys past every slot and yields -1,
    # which would index the zero-filled buffer tail (origin). Clamp it into slot 0, whose
    # state buffers always hold the initial condition.
    step_indices = jnp.maximum(step_indices, 0)
    h_values = (query_times - t_step_starts[step_indices]) / dts[step_indices]
    return step_indices, h_values


def dense_position(
    dense: DenseOutput,
    t: jnp.ndarray,
    particle_index: jnp.ndarray = 0,
) -> jnp.ndarray:
    """Position of one particle at an arbitrary time inside a dense-output buffer.

    Looks up the step containing ``t`` and evaluates that step's converged polynomial
    there. ``h`` is deliberately *not* clipped to ``[0, 1]``: a query outside the
    integrated span still returns the (inaccurate) extrapolation of the nearest step
    rather than a silently clamped value, and callers who care detect the condition with
    :func:`assert_ltt_span_covered`.

    Args:
        dense (DenseOutput): The dense output of one integration pass.
        t (jnp.ndarray): Scalar query time, in the same offset frame as
            ``dense.t_step_starts``.
        particle_index (jnp.ndarray): Which particle to return. Defaults to 0.

    Returns:
        jnp.ndarray:
            Position at ``t``, shape (3,).
    """
    step_indices, h_values = precompute_interpolation_indices(
        dense.t_step_starts, dense.dts, jnp.atleast_1d(t)
    )
    i = step_indices[0]
    x, _ = _estimate_x_v_from_b(
        dense.a0[i],
        dense.v0[i],
        dense.x0[i],
        h_values[0],
        dense.dts[i],
        dense.b[i][::-1],
    )
    return x[particle_index]


def ltt_seed_floor(
    positions: jnp.ndarray,
    observer_positions: jnp.ndarray,
) -> jnp.ndarray:
    """Twice the largest light travel time between a particle and the observers.

    Serves two purposes in the dense-output light-travel-time (LTT) machinery:

    - it is the amount by which the *backward* pass of every dense path is extended
      past the earliest requested time, so that the retarded time ``t_obs - LTT`` of
      every observation lands inside a real integrated step (see
      :func:`make_ltt_propagator`), and
    - it floors the integrator's first *proposed* step (:func:`apply_ltt_seed_floor`),
      which is what protects the paths that cannot extend their span backward (the
      static-likelihood pipeline, whose step schedule is frozen and forward-only).

    The factor of two is margin: the distance is measured from the particle's position
    at the integration epoch, so it under-estimates the light travel time of an
    observation at which the object is farther away. Doubling covers growth up to ~2x
    the epoch topocentric distance.

    Args:
        positions (jnp.ndarray): Particle position(s) at the integration epoch,
            shape (3,) or (P, 3).
        observer_positions (jnp.ndarray): Observer position at each observation time,
            shape (n_obs, 3).

    Returns:
        jnp.ndarray:
            Scalar: 2x the largest particle-observer distance divided by the speed
            of light, in days.
    """
    pos = jnp.atleast_2d(positions)  # (P, 3)
    dists = jnp.linalg.norm(
        pos[:, None, :] - observer_positions[None, :, :], axis=-1
    )  # (P, n_obs)
    return 2.0 * jnp.max(dists) * INV_SPEED_OF_LIGHT


def apply_ltt_seed_floor(
    integrator_state: "IAS15IntegratorState",  # noqa: F821
    positions: jnp.ndarray,
    observer_positions: jnp.ndarray,
) -> "IAS15IntegratorState":  # noqa: F821
    """Return a copy of ``integrator_state`` with ``dt`` floored via :func:`ltt_seed_floor`.

    Preserves the sign of the existing ``dt`` (a reused integrator state may carry a
    signed proposal) and never mutates the input, so cached integrator states are safe
    to pass in.

    Args:
        integrator_state (IAS15IntegratorState): State whose ``dt`` to floor.
        positions (jnp.ndarray): Particle position(s) at the integration epoch,
            shape (3,) or (P, 3).
        observer_positions (jnp.ndarray): Observer position at each observation time,
            shape (n_obs, 3).

    Returns:
        IAS15IntegratorState:
            A copy with ``dt = sign(dt) * max(|dt|, ltt_seed_floor(...))``.
    """
    floor = ltt_seed_floor(positions, observer_positions)
    dt = integrator_state.dt
    sign = jnp.where(dt == 0.0, 1.0, jnp.sign(dt))
    return integrator_state.replace(dt=sign * jnp.maximum(jnp.abs(dt), floor))


def warn_if_ltt_extrapolating(
    x0_per_obs: jnp.ndarray,
    dt_per_obs: jnp.ndarray,
    observer_positions: jnp.ndarray,
) -> None:
    """Warn if any observation's light travel time exceeds its containing step length.

    Host-side check (concrete arrays only, not jittable) for the *static-likelihood*
    path, whose frozen forward-only step schedule cannot be extended backward and so
    still evaluates its first step's polynomial at ``h < 0`` for observations near the
    epoch. The dense (``interpolate=True``) paths no longer need this: they look the
    retarded time up in the buffer and pad their span backward, and report a genuine
    coverage failure through :func:`assert_ltt_span_covered` instead.

    Args:
        x0_per_obs (jnp.ndarray): Start-of-step positions of the steps containing each
            observation, shape (n_obs, n_particles, 3) or (n_obs, 3).
        dt_per_obs (jnp.ndarray): Lengths of the steps containing each observation,
            shape (n_obs,).
        observer_positions (jnp.ndarray): Observer position at each observation time,
            shape (n_obs, 3).
    """
    if x0_per_obs.ndim == 2:
        x0_per_obs = x0_per_obs[:, None, :]
    dists = jnp.linalg.norm(
        x0_per_obs - observer_positions[:, None, :], axis=-1
    )  # (n_obs, n_particles)
    ltts = jnp.max(dists, axis=-1) * INV_SPEED_OF_LIGHT
    excursions = ltts / jnp.abs(dt_per_obs)
    worst = float(jnp.max(excursions))
    if worst > 1.0:
        warnings.warn(
            "The light travel time of at least one observation exceeds the length of "
            f"the IAS15 step containing it (worst ratio: {worst:.1f}). The dense-output "
            "light-travel-time correction extrapolates that step's polynomial beyond "
            "its reliable range, degrading the predicted on-sky positions. This can "
            "happen when the adaptive integrator is forced to take steps shorter than "
            "the light travel time (e.g. a close encounter while the target is "
            "distant).",
            stacklevel=2,
        )


def _covered_end(dense: DenseOutput) -> float:
    """Farthest time a dense buffer covers, i.e. the end of its last accepted step."""
    filled = jnp.abs(dense.dts) < 1e29
    return float(dense.t_step_starts[0]) + float(
        jnp.sum(jnp.where(filled, dense.dts, 0.0))
    )


def assert_ltt_span_covered(
    fwd: DenseOutput,
    bwd: DenseOutput | None,
    t0: float,
    obs_times: jnp.ndarray,
    observer_positions: jnp.ndarray,
) -> None:
    """Raise if any observation's retarded time falls outside the dense-output span.

    Host-side check (concrete arrays only, not jittable). The backward pass of each
    dense path is padded by :func:`ltt_seed_floor`, which is sized from the particle's
    position at the integration epoch; an object whose topocentric distance grows a lot
    between the epoch and an observation can still need more. That is a real failure —
    :func:`dense_position` would have to extrapolate — so it is reported rather than
    silently clamped.

    Args:
        fwd (DenseOutput): Dense output of the forward pass.
        bwd (DenseOutput | None): Dense output of the backward pass, if there is one.
        t0 (float): Integration epoch, in the offset frame of ``obs_times``.
        obs_times (jnp.ndarray): Observation times, shape (n_obs,).
        observer_positions (jnp.ndarray): Observer positions, shape (n_obs, 3).

    Raises:
        RuntimeError: If a retarded time lies outside the integrated span.
    """
    n_particles = fwd.x0.shape[1]

    def positions_at(t: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(lambda p: _ltt_position(fwd, bwd, t0, t, p, 0.0))(
            jnp.arange(n_particles)
        )

    xs = jax.vmap(positions_at)(obs_times)  # (n_obs, P, 3)
    dists = jnp.linalg.norm(xs - observer_positions[:, None, :], axis=-1)
    ltts = jnp.max(dists, axis=-1) * INV_SPEED_OF_LIGHT  # (n_obs,)
    retarded = obs_times - ltts

    ends = [float(t0), _covered_end(fwd)]
    if bwd is not None:
        ends.append(_covered_end(bwd))
    lo, hi = min(ends), max(ends)

    outside = jnp.maximum(lo - retarded, retarded - hi)
    i = int(jnp.argmax(outside))
    # ~0.1 ms, matching the time tolerance used elsewhere for "did we reach this time".
    if float(outside[i]) > 1e-9:
        raise RuntimeError(
            f"The light-travel-corrected (retarded) time of observation {i} falls "
            f"{float(outside[i]):.4f} days outside the integrated span. That "
            f"observation is at relative_time {float(obs_times[i]):.4f} with a light "
            f"travel time of {float(ltts[i]):.4f} days; the dense output covers "
            f"[{lo:.4f}, {hi:.4f}]. Evaluating there would extrapolate the IAS15 "
            "step polynomial, which is inaccurate. The backward span is padded by "
            "twice the light travel time measured at the state epoch, so this means "
            "the object is much farther from the observer at this observation than it "
            "is at the epoch: re-create the Particle/System with a state epoch closer "
            "to the observations."
        )


def _ltt_position(
    fwd: DenseOutput,
    bwd: DenseOutput | None,
    t0: jnp.ndarray,
    t_obs: jnp.ndarray,
    particle_index: jnp.ndarray,
    dt: jnp.ndarray,
) -> jnp.ndarray:
    """Position at ``t_obs + dt``, taken from whichever pass covers that time."""
    t = t_obs + dt
    x_fwd = dense_position(fwd, t, particle_index)
    if bwd is None:
        return x_fwd
    return jnp.where(t >= t0, x_fwd, dense_position(bwd, t, particle_index))


def make_ltt_propagator(
    fwd: DenseOutput,
    bwd: DenseOutput | None,
    t0: jnp.ndarray,
    t_obs: jnp.ndarray,
    particle_index: jnp.ndarray = 0,
) -> jax.tree_util.Partial:
    """Build a closure that evaluates the IAS15 polynomial at a light-travel-delayed time.

    Used inside ``on_sky`` to propagate a particle backward by the light travel time
    using the converged 7th-order polynomial, instead of a constant-acceleration Taylor
    expansion.

    The returned closure maps a (negative) time offset ``dt`` to the particle's position
    at ``t_obs + dt``. It looks up the step that *contains* that time and evaluates that
    step's polynomial at ``h`` in ``[0, 1]``, rather than extrapolating the step
    containing ``t_obs``: light travel times routinely exceed a single adaptive step
    (the integrator shortens steps near perihelion and through close encounters), and
    extrapolating amplifies converged-tolerance noise in the high-order b coefficients
    by ``~h**7``.

    The retarded time of an observation near the integration epoch precedes the forward
    pass, so both passes are needed; callers pad the backward pass by
    :func:`ltt_seed_floor` to guarantee it reaches far enough. ``bwd=None`` is allowed
    for a forward-only schedule that cannot be padded (the static-likelihood path),
    which keeps relying on the first step being at least as long as the light travel
    time.

    The buffers are bound as ``Partial`` arguments rather than captured in the Python
    closure so that they cross ``jit`` boundaries as pytree leaves, not constants.

    Args:
        fwd (DenseOutput): Dense output of the forward pass.
        bwd (DenseOutput | None): Dense output of the backward pass, if there is one.
        t0 (jnp.ndarray): Integration epoch, the boundary between the two passes.
        t_obs (jnp.ndarray): Observation time (scalar), in the same offset frame.
        particle_index (jnp.ndarray): Which particle to propagate. Defaults to 0.

    Returns:
        jax.tree_util.Partial:
            A pytree-friendly callable ``f(dt) -> x_at_delayed_time`` of shape (3,).
    """
    return jax.tree_util.Partial(_ltt_position, fwd, bwd, t0, t_obs, particle_index)


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

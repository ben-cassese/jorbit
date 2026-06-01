"""Dense-output interpolation and light-travel-time utilities for IAS15.

Evaluate the converged 7th-order IAS15 polynomial at arbitrary times within completed
steps, without re-integrating.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jorbit.integrators.ias15.helpers import _estimate_x_v_from_b


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

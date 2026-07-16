"""Dense-output interpolation and light-travel-time utilities for IAS15.

Evaluate the converged 7th-order IAS15 polynomial at arbitrary times within completed
steps, without re-integrating.
"""

import warnings

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jorbit.data.constants import INV_SPEED_OF_LIGHT
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


def ltt_seed_floor(
    positions: jnp.ndarray,
    observer_positions: jnp.ndarray,
) -> jnp.ndarray:
    """Minimum initial IAS15 step proposal for dense-output light-travel-time work.

    :func:`make_ltt_propagator` evaluates the polynomial of the step containing an
    observation at ``h = h_obs - LTT/dt_step``, i.e. it extrapolates outside ``[0, 1]``
    by the light travel time measured in step lengths. If the steps are much shorter
    than the LTT (e.g. a short observation arc of a distant object, where the
    integration ends before the adaptive steps ramp up from their small initial seed),
    that extrapolation amplifies converged-tolerance noise in the high-order b
    coefficients by ~``h^7``, producing arcsec-level, jagged RA/Dec errors.

    Seeding the integrator's first *proposed* step at twice the largest topocentric
    light travel time bounds the excursion to at most ~1 step length (O(1)
    amplification, i.e. errors at the per-step tolerance level) whenever the proposal
    is accepted. Because it is only a proposal, IAS15's accuracy control can still
    reject and shrink it, so integration accuracy is never compromised; steps only end
    up shorter than this floor when the dynamics demand it (close encounters), a regime
    where the topocentric distance — and hence the LTT — is small anyway.

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

    Host-side check (concrete arrays only, not jittable) for the dense-LTT paths: with
    the :func:`ltt_seed_floor` seeding this should never trigger, unless the adaptive
    controller shrank the steps below the floor for accuracy (e.g. a close encounter)
    while an observation still has a long light travel time. In that case
    :func:`make_ltt_propagator` extrapolates its step polynomial by more than ~1 step
    length and the on-sky positions degrade.

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
    same step's polynomial). The excursion is only safe when it is at most ~1 step
    length: beyond that, converged-tolerance noise in the high-order b coefficients
    is amplified by ~``h^7`` (arcsec-level, jagged errors for a distant object
    observed over a short arc). Callers must therefore seed the integration so the
    step containing each observation is at least as long as its light travel time —
    see :func:`ltt_seed_floor`. For close flybys where accuracy forces steps below
    that floor, the LTT is small too, and this still gives a much higher-order
    correction than the constant-acceleration Taylor.

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

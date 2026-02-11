"""DoubleDouble-precision Newtonian gravity acceleration."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jorbit.utils.doubledouble import (
    DoubleDouble,
    dd_concatenate,
    dd_exp,
    dd_sqrt,
    dd_sum,
    dd_where,
    dd_zeros,
)
from jorbit.utils.states import SystemState


def _to_dd(x: jnp.ndarray) -> DoubleDouble:
    """Convert a plain jnp.ndarray to DoubleDouble (lo = 0)."""
    return DoubleDouble(x, jnp.zeros_like(x))


def _newtonian_gravity_dd_core(
    all_pos: DoubleDouble,
    src_pos: DoubleDouble,
    src_gms: DoubleDouble,
    mask: jnp.ndarray,
) -> DoubleDouble:
    """Core Newtonian gravity computation in DoubleDouble precision.

    Computes accelerations on all targets from all sources, with self-interaction
    masked out.

    Args:
        all_pos: Positions of all targets, shape (N, 3).
        src_pos: Positions of all sources, shape (S, 3).
        src_gms: GMs of all sources (already exponentiated), shape (S,).
        mask: Boolean mask for valid pairs, shape (N, S). False where
            self-interaction should be excluded.

    Returns:
        DoubleDouble: Accelerations on all targets, shape (N, 3).
    """
    # Geometry: all targets -> all sources
    all_pos_exp = DoubleDouble(
        all_pos.hi[:, None, :], all_pos.lo[:, None, :]
    )  # (N, 1, 3)
    src_pos_exp = DoubleDouble(
        src_pos.hi[None, :, :], src_pos.lo[None, :, :]
    )  # (1, S, 3)
    dx = all_pos_exp - src_pos_exp  # (N, S, 3)

    r2 = dd_sum(dx * dx, axis=-1)  # (N, S)
    r = dd_sqrt(r2)
    r3 = r2 * r

    one_over_r3 = DoubleDouble(1.0) / r3
    zero_ns = dd_zeros((all_pos.hi.shape[0], src_pos.hi.shape[0]))
    prefac = dd_where(mask, one_over_r3, zero_ns)  # (N, S)

    prefac_exp = DoubleDouble(prefac.hi[:, :, None], prefac.lo[:, :, None])  # (N, S, 1)
    src_gms_exp = DoubleDouble(
        src_gms.hi[None, :, None], src_gms.lo[None, :, None]
    )  # (1, S, 1)

    a_all = -dd_sum(prefac_exp * dx * src_gms_exp, axis=1)  # (N, 3)
    return a_all


@jax.jit
def newtonian_gravity_dd(inputs: SystemState) -> DoubleDouble:
    """Compute Newtonian gravity accelerations in DoubleDouble precision.

    Uses a unified approach: sources = perturbers + massive (all with GM > 0),
    targets = perturbers + massive + tracers. Self-interaction is masked out.
    This naturally handles configurations with 0 massive or 0 tracer particles.

    Args:
        inputs (SystemState): The instantaneous state of the system.

    Returns:
        DoubleDouble:
            The 3D acceleration felt by each particle, ordered by
            massive particles first followed by tracer particles.
    """
    P = inputs.fixed_perturber_positions.shape[0]
    M = inputs.massive_positions.shape[0]
    T = inputs.tracer_positions.shape[0]
    N = P + M + T  # all targets
    S = P + M  # all sources (with GM > 0)

    p_pos = _to_dd(inputs.fixed_perturber_positions)  # (P, 3)
    p_gms = dd_exp(_to_dd(inputs.fixed_perturber_log_gms))  # (P,)

    m_pos = _to_dd(inputs.massive_positions)  # (M, 3)
    m_gms = dd_exp(_to_dd(inputs.log_gms))  # (M,)

    t_pos = _to_dd(inputs.tracer_positions)  # (T, 3)

    # All targets
    all_pos = dd_concatenate([p_pos, m_pos, t_pos], axis=0)  # (N, 3)

    # All sources
    src_pos = dd_concatenate([p_pos, m_pos], axis=0)  # (S, 3)
    src_gms = dd_concatenate([p_gms, m_gms], axis=0)  # (S,)

    # Self-interaction mask: target i == source j when i < S and i == j
    mask = jnp.ones((N, S), dtype=bool)
    mask = mask.at[:S, :].set(~jnp.eye(S, dtype=bool))

    a_all = _newtonian_gravity_dd_core(all_pos, src_pos, src_gms, mask)

    # Return only M+T particles (skip perturbers)
    return DoubleDouble(a_all.hi[P:], a_all.lo[P:])

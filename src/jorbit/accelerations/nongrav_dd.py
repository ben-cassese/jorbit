"""DoubleDouble-precision Marsden-style nongravitational accelerations."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jorbit.utils.doubledouble import (
    DoubleDouble,
    dd_cross,
    dd_norm,
)
from jorbit.utils.states import SystemState


def _to_dd(x: jnp.ndarray) -> DoubleDouble:
    """Convert a plain jnp.ndarray to DoubleDouble (lo = 0)."""
    return DoubleDouble(x, jnp.zeros_like(x))


def nongrav_acceleration_dd(state: SystemState) -> DoubleDouble:
    """Compute nongravitational accelerations in DoubleDouble precision.

    Marsden model for asteroid nongravitational forces. Takes a standard
    SystemState and returns DoubleDouble.

    Args:
        state (SystemState): The instantaneous state of the system.
            Uses a1, a2, a3 from state.acceleration_func_kwargs.

    Returns:
        DoubleDouble:
            The 3D nongravitational acceleration felt by each particle,
            ordered by massive particles first followed by tracer particles.
    """
    x_raw = jnp.concatenate([state.massive_positions, state.tracer_positions])
    v_raw = jnp.concatenate([state.massive_velocities, state.tracer_velocities])

    x = _to_dd(x_raw)  # (N, 3)
    v = _to_dd(v_raw)  # (N, 3)

    # Coefficients: (N,) scalars per particle
    a1_raw = state.acceleration_func_kwargs.get("a1", jnp.zeros(x_raw.shape[0]))
    a2_raw = state.acceleration_func_kwargs.get("a2", jnp.zeros(x_raw.shape[0]))
    a3_raw = state.acceleration_func_kwargs.get("a3", jnp.zeros(x_raw.shape[0]))
    a1 = _to_dd(a1_raw)  # (N,)
    a2 = _to_dd(a2_raw)
    a3 = _to_dd(a3_raw)

    # r = ||x||, per particle
    r = dd_norm(x, axis=1)  # (N,)

    # r x v
    r_cross_v = dd_cross(x, v, axis=1)  # (N, 3)

    # (r x v) x r
    r_cross_v_cross_r = dd_cross(r_cross_v, x, axis=1)  # (N, 3)

    # g(r) = 1/r^2 (asteroid model)
    g_prefactor = DoubleDouble(1.0) / (r * r)  # (N,)

    # Expand scalars to broadcast with (N, 3)
    r_exp = DoubleDouble(r.hi[:, None], r.lo[:, None])  # (N, 1)
    g_exp = DoubleDouble(g_prefactor.hi[:, None], g_prefactor.lo[:, None])

    # term1 = a1 * r_hat
    a1_exp = DoubleDouble(a1.hi[:, None], a1.lo[:, None])
    term1 = a1_exp * x / r_exp  # (N, 3)

    # term2 = a2 * (r x v) x r / ||(r x v) x r||
    norm_rcvr = dd_norm(r_cross_v_cross_r, axis=1)  # (N,)
    norm_rcvr_exp = DoubleDouble(norm_rcvr.hi[:, None], norm_rcvr.lo[:, None])
    a2_exp = DoubleDouble(a2.hi[:, None], a2.lo[:, None])
    term2 = a2_exp * r_cross_v_cross_r / norm_rcvr_exp  # (N, 3)

    # term3 = a3 * (r x v) / ||r x v||
    norm_rcv = dd_norm(r_cross_v, axis=1)  # (N,)
    norm_rcv_exp = DoubleDouble(norm_rcv.hi[:, None], norm_rcv.lo[:, None])
    a3_exp = DoubleDouble(a3.hi[:, None], a3.lo[:, None])
    term3 = a3_exp * r_cross_v / norm_rcv_exp  # (N, 3)

    result = g_exp * (term1 + term2 + term3)
    return result

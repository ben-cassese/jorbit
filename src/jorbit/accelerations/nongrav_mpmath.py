"""Arbitrary-precision Marsden-style nongravitational accelerations using mpmath.

Reference implementation for testing the float64 and DoubleDouble versions.
"""

import numpy as np
from mpmath import matrix, mp, mpf, sqrt

from jorbit.utils.states import SystemState

mp.dps = 70


def _np_to_mpmat(arr: np.ndarray) -> matrix:
    """Convert a numpy array to an mpmath matrix."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        m = matrix(len(arr), 1)
        for i in range(len(arr)):
            m[i] = mpf(float(arr[i]))
        return m
    elif arr.ndim == 2:
        rows, cols = arr.shape
        m = matrix(rows, cols)
        for i in range(rows):
            for j in range(cols):
                m[i, j] = mpf(float(arr[i, j]))
        return m
    else:
        raise ValueError(f"Only 1D and 2D arrays supported, got ndim={arr.ndim}")


def _mpmat_to_np(m: matrix) -> np.ndarray:
    """Convert an mpmath matrix to a numpy float64 array."""
    rows = m.rows
    cols = m.cols
    if cols == 1:
        return np.array([float(m[i]) for i in range(rows)])
    return np.array([[float(m[i, j]) for j in range(cols)] for i in range(rows)])


def _cross3(a: list, b: list) -> list:
    """3D cross product of two 3-element lists of mpf."""
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _norm3(a: list) -> mpf:
    """Euclidean norm of a 3-element list of mpf."""
    return sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)


def nongrav_acceleration_mpmath(
    positions: matrix,
    velocities: matrix,
    a1_coeffs: matrix,
    a2_coeffs: matrix,
    a3_coeffs: matrix,
) -> matrix:
    """Compute Marsden nongravitational accelerations in arbitrary precision.

    Args:
        positions: (N, 3) mpmath matrix of positions.
        velocities: (N, 3) mpmath matrix of velocities.
        a1_coeffs: (N, 1) mpmath matrix of A1 coefficients.
        a2_coeffs: (N, 1) mpmath matrix of A2 coefficients.
        a3_coeffs: (N, 1) mpmath matrix of A3 coefficients.

    Returns:
        (N, 3) mpmath matrix of nongravitational accelerations.
    """
    N = positions.rows
    result = matrix(N, 3)

    for i in range(N):
        x = [positions[i, k] for k in range(3)]
        v = [velocities[i, k] for k in range(3)]

        r = _norm3(x)
        g = 1 / (r * r)  # asteroid model

        # r_hat = x / r
        r_hat = [x[k] / r for k in range(3)]

        # r x v
        rcv = _cross3(x, v)

        # (r x v) x r
        rcvxr = _cross3(rcv, x)

        # Normalize
        norm_rcv = _norm3(rcv)
        norm_rcvxr = _norm3(rcvxr)

        rcv_hat = [rcv[k] / norm_rcv for k in range(3)]
        rcvxr_hat = [rcvxr[k] / norm_rcvxr for k in range(3)]

        for k in range(3):
            term1 = a1_coeffs[i] * r_hat[k]
            term2 = a2_coeffs[i] * rcvxr_hat[k]
            term3 = a3_coeffs[i] * rcv_hat[k]
            result[i, k] = g * (term1 + term2 + term3)

    return result


def nongrav_acceleration_mpmath_from_state(
    state: SystemState,
    dps: int | None = None,
) -> np.ndarray:
    """Compute nongravitational acceleration from a SystemState.

    Args:
        state: SystemState with the instantaneous state.
        dps: Decimal places of precision. If None, uses current mp.dps.

    Returns:
        np.ndarray: (M+T, 3) float64 array of nongravitational accelerations.
    """
    old_dps = mp.dps
    if dps is not None:
        mp.dps = dps

    try:
        x_raw = np.concatenate(
            [np.asarray(state.massive_positions), np.asarray(state.tracer_positions)]
        )
        v_raw = np.concatenate(
            [np.asarray(state.massive_velocities), np.asarray(state.tracer_velocities)]
        )
        N = x_raw.shape[0]

        a1_raw = state.acceleration_func_kwargs.get("a1", np.zeros(N))
        a2_raw = state.acceleration_func_kwargs.get("a2", np.zeros(N))
        a3_raw = state.acceleration_func_kwargs.get("a3", np.zeros(N))

        positions_mp = _np_to_mpmat(x_raw)
        velocities_mp = _np_to_mpmat(v_raw)

        a1_mp = matrix(N, 1)
        a2_mp = matrix(N, 1)
        a3_mp = matrix(N, 1)
        for i in range(N):
            a1_mp[i] = mpf(float(np.asarray(a1_raw)[i]))
            a2_mp[i] = mpf(float(np.asarray(a2_raw)[i]))
            a3_mp[i] = mpf(float(np.asarray(a3_raw)[i]))

        result_mp = nongrav_acceleration_mpmath(
            positions_mp, velocities_mp, a1_mp, a2_mp, a3_mp
        )
        return _mpmat_to_np(result_mp)

    finally:
        if dps is not None:
            mp.dps = old_dps

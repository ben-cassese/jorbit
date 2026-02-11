"""Arbitrary-precision Newtonian gravity using mpmath.

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


def newtonian_gravity_mpmath(
    positions: matrix,
    gms: matrix,
) -> matrix:
    """Compute Newtonian gravity in arbitrary precision.

    All N particles interact with all others (no perturber/tracer distinction).

    Args:
        positions: (N, 3) mpmath matrix of positions.
        gms: (N, 1) or length-N mpmath matrix of GM values.

    Returns:
        (N, 3) mpmath matrix of accelerations.
    """
    N = positions.rows
    result = matrix(N, 3)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dx = [positions[i, k] - positions[j, k] for k in range(3)]
            r2 = dx[0] ** 2 + dx[1] ** 2 + dx[2] ** 2
            r = sqrt(r2)
            r3 = r2 * r
            for k in range(3):
                result[i, k] -= gms[j] * dx[k] / r3

    return result


def newtonian_gravity_mpmath_from_state(
    inputs: SystemState,
    dps: int | None = None,
) -> np.ndarray:
    """Compute Newtonian gravity from a SystemState, returning numpy float64.

    Args:
        inputs: SystemState with the instantaneous state.
        dps: Decimal places of precision. If None, uses current mp.dps.

    Returns:
        np.ndarray: (M+T, 3) float64 array of accelerations.
    """
    old_dps = mp.dps
    if dps is not None:
        mp.dps = dps

    try:
        p_pos = np.asarray(inputs.fixed_perturber_positions)
        p_gms = np.exp(np.asarray(inputs.fixed_perturber_log_gms))

        m_pos = np.asarray(inputs.massive_positions)
        m_gms = np.exp(np.asarray(inputs.log_gms))

        t_pos = np.asarray(inputs.tracer_positions)
        t_gms = np.zeros(t_pos.shape[0])

        all_pos = np.concatenate([p_pos, m_pos, t_pos], axis=0)
        all_gms = np.concatenate([p_gms, m_gms, t_gms])

        P = p_pos.shape[0]
        N = all_pos.shape[0]

        positions_mp = _np_to_mpmat(all_pos)
        gms_mp = matrix(N, 1)
        for i in range(N):
            gms_mp[i] = mpf(float(all_gms[i]))

        result_mp = newtonian_gravity_mpmath(positions_mp, gms_mp)
        result_np = _mpmat_to_np(result_mp)
        return result_np[P:]

    finally:
        if dps is not None:
            mp.dps = old_dps

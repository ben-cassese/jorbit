"""Arbitrary-precision PPN gravity acceleration model using mpmath.

This is a reference implementation of the PPN gravity model for testing the
accuracy of the float64 and DoubleDouble versions. All computations use mpmath
arbitrary-precision arithmetic (controlled by mp.dps). Not suitable for
production use — this is pure Python with explicit loops and no JIT compilation.

The physics and algorithm are identical to ppn_gravity in gr.py, which is based
on REBOUNDx (Tamayo et al. 2020) and Newhall et al. (1983).
"""

import numpy as np
from mpmath import matrix, mp, mpf, sqrt

from jorbit.data.constants import SPEED_OF_LIGHT
from jorbit.utils.states import SystemState

# Default to 70 decimal digits, matching the arbitrary-precision integrator.
# Callers can override by setting mp.dps before calling these functions.
mp.dps = 70


def _np_to_mpmat(arr: np.ndarray) -> matrix:
    """Convert a numpy array to an mpmath matrix, preserving shape."""
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


def _dot3(a: matrix, row_a: int, b: matrix, row_b: int) -> mpf:
    """Dot product of two 3-vectors stored as rows of mpmath matrices."""
    return (
        a[row_a, 0] * b[row_b, 0]
        + a[row_a, 1] * b[row_b, 1]
        + a[row_a, 2] * b[row_b, 2]
    )


def _norm3(a: matrix, row: int) -> mpf:
    """Euclidean norm of a 3-vector stored as a row of an mpmath matrix."""
    return sqrt(_dot3(a, row, a, row))


def ppn_gravity_mpmath(
    positions: matrix,
    velocities: matrix,
    gms: matrix,
    c: mpf | float = SPEED_OF_LIGHT,
    max_iterations: int = 50,
) -> matrix:
    """Compute PPN gravity accelerations in arbitrary precision.

    All N particles are treated symmetrically (no perturber/tracer distinction).
    This matches the physics of ppn_gravity when all particles are "massive".

    Args:
        positions: (N, 3) mpmath matrix of positions in AU.
        velocities: (N, 3) mpmath matrix of velocities in AU/day.
        gms: (N, 1) or length-N mpmath matrix of GM values.
        c: Speed of light in AU/day. Defaults to jorbit's SPEED_OF_LIGHT.
        max_iterations: Maximum iterations for GR correction convergence.

    Returns:
        (N, 3) mpmath matrix of total accelerations (Newtonian + PPN).
    """
    c = mpf(c)
    c2 = c * c
    N = positions.rows

    # Ensure gms is accessible by index
    if gms.cols > 1:
        raise ValueError("gms should be (N, 1) or a column vector")

    # ---- Pairwise geometry ----
    # dx[i][j] = pos[i] - pos[j], stored as list of lists of 3-tuples
    dx = [
        [
            (
                positions[i, 0] - positions[j, 0],
                positions[i, 1] - positions[j, 1],
                positions[i, 2] - positions[j, 2],
            )
            for j in range(N)
        ]
        for i in range(N)
    ]

    # r[i][j], r2[i][j], r3[i][j]
    r2 = [[mpf(0)] * N for _ in range(N)]
    r = [[mpf(0)] * N for _ in range(N)]
    r3 = [[mpf(0)] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j:
                r2[i][j] = dx[i][j][0] ** 2 + dx[i][j][1] ** 2 + dx[i][j][2] ** 2
                r[i][j] = sqrt(r2[i][j])
                r3[i][j] = r2[i][j] * r[i][j]

    # ---- Newtonian acceleration ----
    a_newt = matrix(N, 3)
    for i in range(N):
        for k in range(3):
            s = mpf(0)
            for j in range(N):
                if i != j:
                    s -= gms[j] * dx[i][j][k] / r3[i][j]
            a_newt[i, k] = s

    # ---- COM frame ----
    total_gm = mpf(0)
    for j in range(N):
        total_gm += gms[j]

    v_com = [mpf(0), mpf(0), mpf(0)]
    for j in range(N):
        for k in range(3):
            v_com[k] += velocities[j, k] * gms[j]
    for k in range(3):
        v_com[k] /= total_gm

    # Velocities in COM frame
    vel_com = matrix(N, 3)
    for i in range(N):
        for k in range(3):
            vel_com[i, k] = velocities[i, k] - v_com[k]

    # v² for each particle
    v2 = [mpf(0)] * N
    for i in range(N):
        v2[i] = _dot3(vel_com, i, vel_com, i)

    # ---- a1: sum over k!=i of 4*GM_k/r_ik ----
    a1 = [mpf(0)] * N
    for i in range(N):
        s = mpf(0)
        for k in range(N):
            if k != i:
                s += gms[k] / r[i][k]
        a1[i] = 4 * s / c2

    # ---- a2: sum over k!=j of GM_k/r_jk for each source j ----
    a2 = [mpf(0)] * N
    for j in range(N):
        s = mpf(0)
        for k in range(N):
            if k != j:
                s += gms[k] / r[j][k]
        a2[j] = s / c2

    # ---- Constant PPN terms ----
    # For each target i, sum constant terms over all sources j != i
    a_const = matrix(N, 3)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            # vi·vj
            vdot_ij = _dot3(vel_com, i, vel_com, j)

            a3 = -v2[i] / c2
            a4 = -2 * v2[j] / c2
            a5 = 4 * vdot_ij / c2

            # (dx_ij . v_j)^2 / r2
            dx_dot_vj = (
                dx[i][j][0] * vel_com[j, 0]
                + dx[i][j][1] * vel_com[j, 1]
                + dx[i][j][2] * vel_com[j, 2]
            )
            a6 = mpf(3) / (2 * c2) * dx_dot_vj**2 / r2[i][j]

            # dx_ij . a_newt_j
            dx_dot_aj = (
                dx[i][j][0] * a_newt[j, 0]
                + dx[i][j][1] * a_newt[j, 1]
                + dx[i][j][2] * a_newt[j, 2]
            )
            a7 = dx_dot_aj / (2 * c2)

            factor1 = a1[i] + a2[j] + a3 + a4 + a5 + a6 + a7

            # factor2 = dx . (4*vi - 3*vj)
            factor2 = mpf(0)
            for k in range(3):
                factor2 += dx[i][j][k] * (4 * vel_com[i, k] - 3 * vel_com[j, k])

            # dv = vi - vj (in COM frame)
            dv = [vel_com[i, k] - vel_com[j, k] for k in range(3)]

            for k in range(3):
                part1 = gms[j] * dx[i][j][k] * factor1 / r3[i][j]
                part2 = (gms[j] / c2) * (
                    factor2 * dv[k] / r3[i][j] + mpf(7) / 2 * a_newt[j, k] / r[i][j]
                )
                a_const[i, k] += part1 + part2

    # ---- Iterative non-constant PPN terms ----
    # The non-constant terms depend on the GR-corrected acceleration of each
    # source, which itself depends on the non-constant terms. We iterate.
    a_gr = matrix(N, 3)
    for i in range(N):
        for k in range(3):
            a_gr[i, k] = a_const[i, k]

    for _iteration in range(max_iterations):
        a_gr_new = matrix(N, 3)
        for i in range(N):
            for k in range(3):
                a_gr_new[i, k] = a_const[i, k]

        # Add non-constant terms
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                # rdota = dx_ij . a_gr_j
                rdota = (
                    dx[i][j][0] * a_gr[j, 0]
                    + dx[i][j][1] * a_gr[j, 1]
                    + dx[i][j][2] * a_gr[j, 2]
                )
                for k in range(3):
                    non_const = (gms[j] / (2 * c2)) * (
                        dx[i][j][k] * rdota / r3[i][j] + 7 * a_gr[j, k] / r[i][j]
                    )
                    a_gr_new[i, k] += non_const

        # Check convergence
        max_ratio = mpf(0)
        for i in range(N):
            for k in range(3):
                if a_gr_new[i, k] != 0:
                    ratio = abs((a_gr_new[i, k] - a_gr[i, k]) / a_gr_new[i, k])
                    if ratio > max_ratio:
                        max_ratio = ratio
        a_gr = a_gr_new

        if max_ratio < mpf(10) ** (-(mp.dps - 5)):
            break

    # Total = Newtonian + GR
    result = matrix(N, 3)
    for i in range(N):
        for k in range(3):
            result[i, k] = a_newt[i, k] + a_gr[i, k]

    return result


def ppn_gravity_mpmath_from_state(
    inputs: SystemState,
    max_iterations: int = 50,
    dps: int | None = None,
) -> np.ndarray:
    """Compute PPN gravity from a SystemState, returning numpy float64 result.

    Convenience wrapper that extracts positions, velocities, and GMs from a
    SystemState, converts them to mpmath, runs the arbitrary-precision PPN
    computation, and converts back to numpy. Useful for direct comparison
    against ppn_gravity or ppn_gravity_dd.

    All particles (fixed perturbers, massive, tracers) are treated as full
    N-body participants. The returned accelerations cover massive + tracer
    particles (same ordering as ppn_gravity).

    Args:
        inputs: SystemState with the instantaneous state.
        max_iterations: Maximum GR correction iterations.
        dps: Decimal places of precision. If None, uses current mp.dps.

    Returns:
        np.ndarray: (M+T, 3) float64 array of accelerations.
    """
    old_dps = mp.dps
    if dps is not None:
        mp.dps = dps

    try:
        p_pos = np.asarray(inputs.fixed_perturber_positions)
        p_vel = np.asarray(inputs.fixed_perturber_velocities)
        p_gms = np.exp(np.asarray(inputs.fixed_perturber_log_gms))

        m_pos = np.asarray(inputs.massive_positions)
        m_vel = np.asarray(inputs.massive_velocities)
        m_gms = np.exp(np.asarray(inputs.log_gms))

        t_pos = np.asarray(inputs.tracer_positions)
        t_vel = np.asarray(inputs.tracer_velocities)
        t_gms = np.zeros(t_pos.shape[0])

        all_pos = np.concatenate([p_pos, m_pos, t_pos], axis=0)
        all_vel = np.concatenate([p_vel, m_vel, t_vel], axis=0)
        all_gms = np.concatenate([p_gms, m_gms, t_gms])

        P = p_pos.shape[0]
        N = all_pos.shape[0]

        positions_mp = _np_to_mpmat(all_pos)
        velocities_mp = _np_to_mpmat(all_vel)

        gms_mp = matrix(N, 1)
        for i in range(N):
            gms_mp[i] = mpf(float(all_gms[i]))

        c_val = inputs.acceleration_func_kwargs.get("c2", SPEED_OF_LIGHT**2)
        c_mp = sqrt(mpf(float(c_val)))

        result_mp = ppn_gravity_mpmath(
            positions_mp,
            velocities_mp,
            gms_mp,
            c=c_mp,
            max_iterations=max_iterations,
        )

        # Convert back, returning only M+T (skip perturbers)
        result_np = _mpmat_to_np(result_mp)
        return result_np[P:]

    finally:
        if dps is not None:
            mp.dps = old_dps

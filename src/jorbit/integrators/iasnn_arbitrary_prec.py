"""An experimental/testing module for testing the DoubleDouble integrator stuff.

This implements arbitrary-order IAS15-style integrators using the mpmath library.
It's about as anti-JAX as you can get, but it's useful for testing the DoubleDouble
operations.
"""

import mpmath as mpm
import numpy as np
from mpmath import matrix, mp, mpf

from jorbit.utils.generate_coefficients import create_iasnn_constants

mp.dps = 70


def acceleration_func(x: mpm.matrix) -> mpm.matrix:
    """A dummy acceleration function for testing purposes, unit central potential.

    Args:
        x: (n_particles, 3) array of positions

    Returns:
        a: (n_particles, 3) array of accelerations
    """
    r = mpm.norm(x)
    return -x / (r * r * r)


def precompute(n_internal_points: int) -> tuple:
    """Precompute the constants needed for an arbitrary-order IASNN-style integrator.

    This creates the H, C, R, D arrays you see in the REBOUND source, and some
    precomputed denominators for the Taylor expansion coefficients.

    Args:
        n_internal_points (int):
            The number of internal evaluations to use when estimating the acceleration
            across a given step. IAS15 uses 7.

    Returns:
        tuple:
            b_x_denoms (mpm.matrix):
                The denominators for the Taylor expansion of the position.
            b_v_denoms (mpm.matrix):
                The denominators for the Taylor expansion of the velocity.
            h (mpm.matrix):
                The H array from REBOUND, a list of the roots of the Chebyshev polynomial.
            r (mpm.matrix):
                The R array from REBOUND, a list of the coefficients for the Taylor
                expansion of the acceleration.
            c (mpm.matrix):
                The C array from REBOUND, a list of the coefficients for the Taylor
                expansion of the acceleration.
            d (mpm.matrix):
                The D array from REBOUND, a matrix of coefficients for the Taylor
                expansion of the acceleration.
    """
    b_x_denoms = (1.0 + np.arange(1, n_internal_points + 1, 1)) * (
        2.0 + np.arange(1, n_internal_points + 1, 1)
    )
    b_x_denoms = [str(i) for i in b_x_denoms]
    b_v_denoms = np.arange(2, n_internal_points + 2, 1)
    b_v_denoms = [str(i) for i in b_v_denoms]
    h, r, c, d = create_iasnn_constants(n_internal_points)

    b_x_denoms = matrix(b_x_denoms)
    b_v_denoms = matrix(b_v_denoms)
    h = matrix(h)
    r = matrix(r)
    c = matrix(c)
    d = matrix(d)

    d_matrix = mpm.zeros(n_internal_points, n_internal_points)
    indices = np.tril_indices(n_internal_points, k=-1)
    # z = 0
    for z, (i, j) in enumerate(zip(*indices)):
        d_matrix[i, j] = d[z]
        # z += 1
    for i in range(n_internal_points):
        d_matrix[i, i] = 1.0
    d_matrix = d_matrix.T

    return b_x_denoms, b_v_denoms, h, r.apply(lambda x: 1 / x), c, d_matrix


def estimate_x_v_from_b(
    a0: mpm.matrix,
    v0: mpm.matrix,
    x0: mpm.matrix,
    dt: mpm.matrix,
    b_x_denoms: mpm.matrix,
    b_v_denoms: mpm.matrix,
    h: mpm.matrix,
    bp: mpm.matrix,
) -> tuple:
    """Same as the estimate_x_v_from_b in the DoubleDouble integrator, but using mpmath."""
    n_internal_points = len(bp)
    # Initialize xcoeffs with zeros
    # Shape will be (10, 3) - 7 points from bp + 3 initial conditions
    xcoeffs = matrix(
        [[mp.mpf("0") for _ in range(3)] for _ in range(n_internal_points + 3)]
    )

    # Fill xcoeffs with bp values
    for i in range(n_internal_points):  # 7 internal points
        for k in range(3):  # 3 dimensions
            xcoeffs[i + 3, k] = bp[i, k] * dt * dt / b_x_denoms[i]

    # Set initial conditions
    for k in range(3):
        xcoeffs[2, k] = a0[k] * dt * dt / mp.mpf("2.0")
        xcoeffs[1, k] = v0[k] * dt
        xcoeffs[0, k] = x0[k]

    # Reverse xcoeffs
    xcoeffs = xcoeffs[::-1, :]

    # Initialize results
    estimated_x = matrix([mp.mpf("0") for _ in range(3)])

    # Compute estimated_x using Horner's method
    for k in range(3):
        result = mp.mpf("0")
        for coeff in xcoeffs[:, k]:
            result = result * h + coeff
        estimated_x[k] = result

    # Similar process for velocity coefficients
    # Shape will be (9, 3) - 7 points from bp + 2 initial conditions
    vcoeffs = matrix(
        [[mp.mpf("0") for _ in range(3)] for _ in range(n_internal_points + 2)]
    )

    for i in range(n_internal_points):  # 7 internal points
        for k in range(3):
            vcoeffs[i + 2, k] = bp[i, k] * dt / b_v_denoms[i]

    for k in range(3):
        vcoeffs[1, k] = a0[k] * dt
        vcoeffs[0, k] = v0[k]

    vcoeffs = vcoeffs[::-1, :]

    estimated_v = matrix([mp.mpf("0") for _ in range(3)])

    for k in range(3):
        result = mp.mpf("0")
        for coeff in vcoeffs[:, k]:
            result = result * h + coeff
        estimated_v[k] = result

    return estimated_x.T, estimated_v.T


def refine_intermediate_g(
    substep_num: int, g: mpm.matrix, r: mpm.matrix, at: mpm.matrix, a0: mpm.matrix
) -> mpm.matrix:
    """Same as the refine_intermediate_g in the DoubleDouble integrator, but using mpmath."""
    substep_num -= 1
    start_pos = (substep_num * (substep_num + 1)) // 2

    # Initial result computation
    result = (at - a0) * r[start_pos]

    # Iterate through previous substeps
    for idx in range(substep_num):
        result = (result - g[idx, :]) * r[start_pos + idx + 1]

    return result


def refine_b_and_g(
    r: mpm.matrix,
    c: mpm.matrix,
    b: mpm.matrix,
    g: mpm.matrix,
    at: mpm.matrix,
    a0: mpm.matrix,
    substep_num: int,
    return_g_diff: bool,
) -> tuple:
    """Same as the refine_b_and_g in the DoubleDouble integrator, but using mpmath."""
    old_g = g[substep_num - 1, :]
    new_g = refine_intermediate_g(substep_num=substep_num, g=g, r=r, at=at, a0=a0)
    g_diff = new_g - old_g
    g[substep_num - 1, :] = new_g

    c_start = (substep_num - 1) * (substep_num - 2) // 2
    c_vals = [mp.mpf("1") for _ in range(substep_num)]

    for i in range(substep_num - 1):
        c_vals[i] = c[c_start + i]

    # Update b array - now just one particle
    for idx in range(substep_num):
        for j in range(3):  # 3 dimensions
            b[idx, j] = b[idx, j] + (g_diff[j] * c_vals[idx])

    if return_g_diff:
        return b, g, g_diff
    return b, g


def step(
    x0: mpm.matrix,
    v0: mpm.matrix,
    b: mpm.matrix,
    dt: mpm.matrix,
    precomputed_setup: tuple,
    verbose: bool = False,
    convergence_threshold: mpm.mpf = mpf("1e-40"),
) -> tuple:
    """Same as the step in the DoubleDouble integrator, but using mpmath."""
    b_x_denoms, b_v_denoms, h, r, c, d = precomputed_setup
    n_internal_points = len(b)
    a0 = acceleration_func(x0)

    # initialize the gs from the bs
    g = d * b

    def predictor_corrector_iteration(
        b: mpm.matrix, g: mpm.matrix, predictor_corrector_error: mpm.mpf
    ) -> tuple:
        predictor_corrector_error_last = predictor_corrector_error
        predictor_corrector_error = 0.0

        for n in range(1, n_internal_points):
            x, v = estimate_x_v_from_b(
                a0=a0,
                v0=v0,
                x0=x0,
                dt=dt,
                b_x_denoms=b_x_denoms,
                b_v_denoms=b_v_denoms,
                h=h[n],
                bp=b,
            )
            at = acceleration_func(x)
            b, g = refine_b_and_g(
                r=r, c=c, b=b, g=g, at=at, a0=a0, substep_num=n, return_g_diff=False
            )
        n = n_internal_points
        x, v = estimate_x_v_from_b(
            a0=a0,
            v0=v0,
            x0=x0,
            dt=dt,
            b_x_denoms=b_x_denoms,
            b_v_denoms=b_v_denoms,
            h=h[n],
            bp=b,
        )
        at = acceleration_func(x)
        b, g, g_diff = refine_b_and_g(
            r=r, c=c, b=b, g=g, at=at, a0=a0, substep_num=n, return_g_diff=True
        )

        maxa = max(at.apply(abs))
        maxb6tmp = max(g_diff.apply(abs))
        predictor_corrector_error = abs(maxb6tmp / maxa)

        return b, g, predictor_corrector_error, predictor_corrector_error_last

    predictor_corrector_error = 1e300
    for i in range(200):
        if verbose:
            print(f"i: {i}")
            print(f"predictor_corrector_error: {predictor_corrector_error}")
        b, g, predictor_corrector_error, predictor_corrector_error_last = (
            predictor_corrector_iteration(b, g, predictor_corrector_error)
        )

        condition = (predictor_corrector_error < mpf("1e-60")) | (
            (i > 2) & (predictor_corrector_error > predictor_corrector_error_last)
        )

        if condition:
            if verbose:
                print("stopping early!")
                if predictor_corrector_error < convergence_threshold:
                    print("error is small")
                else:
                    print("error is increasing")
            break

    x, v = estimate_x_v_from_b(
        a0=a0,
        v0=v0,
        x0=x0,
        dt=dt,
        b_x_denoms=b_x_denoms,
        b_v_denoms=b_v_denoms,
        h=mpf("1.0"),
        bp=b,
    )

    return x, v, b

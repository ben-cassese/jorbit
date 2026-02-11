"""Arbitrary-precision gravitational harmonics acceleration using mpmath.

Reference implementation for testing the float64 and DoubleDouble versions.
Uses analytical derivatives of the Legendre polynomial potential.
"""

import numpy as np
from mpmath import cos, mp, mpf, sin, sqrt

mp.dps = 70


def _legendre_and_deriv_mp(u: mpf, n_max: int) -> tuple:
    """Compute Legendre polynomials P_n(u) and derivatives P_n'(u) for n=0..n_max.

    Args:
        u: mpf scalar (cos theta).
        n_max: Maximum degree.

    Returns:
        p_vals: list of mpf, P_0(u) through P_{n_max}(u).
        dp_vals: list of mpf, P_0'(u) through P_{n_max}'(u).
    """
    p_vals = [mpf(1), u]
    dp_vals = [mpf(0), mpf(1)]

    for n in range(1, n_max):
        p_new = ((2 * n + 1) * u * p_vals[n] - n * p_vals[n - 1]) / (n + 1)
        p_vals.append(p_new)

        dp_new = ((2 * n + 1) * (p_vals[n] + u * dp_vals[n]) - n * dp_vals[n - 1]) / (
            n + 1
        )
        dp_vals.append(dp_new)

    return p_vals, dp_vals


def grav_harmonics_mpmath(
    body_gm: float,
    body_req: float,
    body_pos: np.ndarray,
    pole_ra: float,
    pole_dec: float,
    jns: np.ndarray,
    particle_xs: np.ndarray,
    dps: int | None = None,
) -> np.ndarray:
    """Compute gravitational harmonics acceleration in arbitrary precision.

    Args:
        body_gm (float): Gravitational parameter of the body.
        body_req (float): Equatorial radius of the body.
        body_pos (np.ndarray): Position of the body's center, shape (3,).
        pole_ra (float): Right ascension of the pole in radians.
        pole_dec (float): Declination of the pole in radians.
        jns (np.ndarray): J coefficients starting from J2, shape (N_j,).
        particle_xs (np.ndarray): Particle positions, shape (P, 3).
        dps (int | None): Decimal places. If None, uses current mp.dps.

    Returns:
        np.ndarray: Accelerations, shape (P, 3), in float64.
    """
    old_dps = mp.dps
    if dps is not None:
        mp.dps = dps

    try:
        N_j = len(jns)
        n_max = N_j + 1  # degrees 2 through N_j+1
        P = particle_xs.shape[0]

        # Rotation matrix
        sa = sin(mpf(float(pole_ra)))
        ca = cos(mpf(float(pole_ra)))
        sd = sin(mpf(float(pole_dec)))
        cd = cos(mpf(float(pole_dec)))

        rot = [
            [-sa, -ca * sd, ca * cd],
            [ca, -sa * sd, sa * cd],
            [mpf(0), cd, sd],
        ]

        gm = mpf(float(body_gm))
        req = mpf(float(body_req))
        bp = [mpf(float(body_pos[k])) for k in range(3)]
        jns_mp = [mpf(float(jns[j])) for j in range(N_j)]

        result = np.zeros((P, 3))

        for ip in range(P):
            # Center on body
            x = [mpf(float(particle_xs[ip, k])) - bp[k] for k in range(3)]

            # Rotate into body frame
            xb = [mpf(0)] * 3
            for i in range(3):
                for j in range(3):
                    xb[i] += x[j] * rot[j][i]

            # r, cos(theta)
            r2 = xb[0] ** 2 + xb[1] ** 2 + xb[2] ** 2
            r = sqrt(r2)
            u = xb[2] / r

            p_vals, dp_vals = _legendre_and_deriv_mp(u, n_max + 1)

            ax = mpf(0)
            ay = mpf(0)
            az = mpf(0)

            sin2_theta = 1 - u * u

            for j_idx in range(N_j):
                n = j_idx + 2
                jn = jns_mp[j_idx]

                # C = GM * Jn * (Req/r)^n / r^3
                req_over_r = req / r
                ratio_n = req_over_r**n
                r3 = r2 * r
                coeff = gm * jn * ratio_n / r3

                pn = p_vals[n]
                dpn = dp_vals[n]

                xy_factor = (n + 1) * pn + u * dpn

                ax += coeff * xb[0] * xy_factor
                ay += coeff * xb[1] * xy_factor
                az += coeff * ((n + 1) * pn * xb[2] - dpn * sin2_theta * r)

            # Rotate back: a_orig = a_body @ rot^T
            a_body = [ax, ay, az]
            a_orig = [mpf(0)] * 3
            for i in range(3):
                for j in range(3):
                    a_orig[i] += a_body[j] * rot[j][i]

            # Note: rot^T[j,i] = rot[i,j], but since rot is stored as
            # rot[row][col], rot^T dotted with a vector is
            # sum_j a[j] * rot[j][i] (which is what we wrote above,
            # same as the forward rotation). Let me fix this.
            # Forward: x_body[i] = sum_j x[j] * rot[j][i]
            # Backward: a_orig[i] = sum_j a_body[j] * rot_T[j][i]
            #         = sum_j a_body[j] * rot[i][j]
            a_orig = [mpf(0)] * 3
            for i in range(3):
                for j in range(3):
                    a_orig[i] += a_body[j] * rot[i][j]

            for k in range(3):
                result[ip, k] = float(a_orig[k])

        return result

    finally:
        if dps is not None:
            mp.dps = old_dps

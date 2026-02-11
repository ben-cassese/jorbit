"""DoubleDouble-precision gravitational harmonics acceleration.

Computes accelerations from J2, J3, ... harmonics of a single extended body
using analytical derivatives of the Legendre polynomial potential, rather
than JAX's automatic differentiation. All intermediate computations use
DoubleDouble arithmetic.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jorbit.utils.doubledouble import (
    DoubleDouble,
    dd_sqrt,
    dd_sum,
)


def _to_dd(x: jnp.ndarray) -> DoubleDouble:
    """Convert a plain jnp.ndarray to DoubleDouble (lo = 0)."""
    return DoubleDouble(x, jnp.zeros_like(x))


def _legendre_and_deriv(u: DoubleDouble, n_max: int) -> tuple:
    """Compute Legendre polynomials P_n(u) and derivatives P_n'(u) for n=0..n_max.

    Uses the standard recurrence relations:
        P_0(u) = 1, P_1(u) = u
        (n+1) P_{n+1}(u) = (2n+1) u P_n(u) - n P_{n-1}(u)

        P_0'(u) = 0, P_1'(u) = 1
        P_{n+1}'(u) = ((2n+1)(P_n(u) + u P_n'(u)) - n P_{n-1}'(u)) / (n+1)

    Args:
        u: DoubleDouble scalar (cos theta).
        n_max: Maximum degree.

    Returns:
        p_vals: list of DoubleDouble scalars, P_0(u) through P_{n_max}(u).
        dp_vals: list of DoubleDouble scalars, P_0'(u) through P_{n_max}'(u).
    """
    one = DoubleDouble(jnp.ones_like(u.hi))
    zero = DoubleDouble(jnp.zeros_like(u.hi))

    p_vals = [one, u]
    dp_vals = [zero, one]

    for n in range(1, n_max):
        # (n+1) P_{n+1} = (2n+1) u P_n - n P_{n-1}
        p_new = (DoubleDouble(2 * n + 1) * u * p_vals[n] - n * p_vals[n - 1]) / (n + 1)
        p_vals.append(p_new)

        # P_{n+1}' = ((2n+1)(P_n + u P_n') - n P_{n-1}') / (n+1)
        dp_new = (
            DoubleDouble(2 * n + 1) * (p_vals[n] + u * dp_vals[n]) - n * dp_vals[n - 1]
        ) / (n + 1)
        dp_vals.append(dp_new)

    return p_vals, dp_vals


def grav_harmonics_dd(
    body_gm: float,
    body_req: float,
    body_pos: jnp.ndarray,
    pole_ra: float,
    pole_dec: float,
    jns: jnp.ndarray,
    particle_xs: jnp.ndarray,
) -> DoubleDouble:
    """Compute gravitational harmonics acceleration in DoubleDouble precision.

    Uses analytical derivatives of the Legendre polynomial potential rather
    than automatic differentiation. The potential for each Jn is:

        V_n = -GM * J_n * (R_eq/r)^n * P_n(cos theta) / r

    and the acceleration is a = -grad(V).

    Args:
        body_gm (float): Gravitational parameter of the body.
        body_req (float): Equatorial radius of the body.
        body_pos (jnp.ndarray): Position of the body's center, shape (3,).
        pole_ra (float): Right ascension of the pole in radians.
        pole_dec (float): Declination of the pole in radians.
        jns (jnp.ndarray): Spherical harmonic coefficients, shape (N_j,),
            starting from J2.
        particle_xs (jnp.ndarray): Particle positions, shape (P, 3).

    Returns:
        DoubleDouble: Accelerations, shape (P, 3).
    """
    N_j = jns.shape[0]
    n_max = N_j + 1  # degrees 2 through N_j+1

    # Rotation matrix (computed in float64, wrapped to DD)
    sin_a = jnp.sin(pole_ra)
    cos_a = jnp.cos(pole_ra)
    sin_d = jnp.sin(pole_dec)
    cos_d = jnp.cos(pole_dec)
    rot_matrix = jnp.array(
        [
            [-sin_a, -cos_a * sin_d, cos_a * cos_d],
            [cos_a, -sin_a * sin_d, sin_a * cos_d],
            [0.0, cos_d, sin_d],
        ]
    )
    rot_dd = _to_dd(rot_matrix)  # (3, 3)
    rot_T_dd = _to_dd(rot_matrix.T)  # (3, 3)

    body_pos_dd = _to_dd(body_pos)
    gm_dd = DoubleDouble(jnp.float64(body_gm))
    req_dd = DoubleDouble(jnp.float64(body_req))

    P = particle_xs.shape[0]
    result_hi = jnp.zeros((P, 3))
    result_lo = jnp.zeros((P, 3))

    # Process each particle (no vectorization for DD clarity)
    for ip in range(P):
        x_raw = _to_dd(particle_xs[ip])  # (3,)

        # Center on body
        x_centered = x_raw - body_pos_dd  # (3,)

        # Rotate into body frame: x_body = x_centered @ rot_matrix
        # Manual matrix-vector multiply in DD
        x_body_hi = jnp.zeros(3)
        x_body_lo = jnp.zeros(3)
        for i in range(3):
            s = DoubleDouble(jnp.float64(0.0))
            for j in range(3):
                xj = DoubleDouble(x_centered.hi[j], x_centered.lo[j])
                rij = DoubleDouble(rot_dd.hi[j, i], rot_dd.lo[j, i])
                s = s + xj * rij
            x_body_hi = x_body_hi.at[i].set(s.hi)
            x_body_lo = x_body_lo.at[i].set(s.lo)
        x_body = DoubleDouble(x_body_hi, x_body_lo)

        # r, u = cos(theta) = z/r
        r2 = dd_sum(x_body * x_body)
        r = dd_sqrt(r2)
        z = DoubleDouble(x_body.hi[2], x_body.lo[2])
        u = z / r  # cos(theta)

        # Legendre polynomials and derivatives up to degree n_max
        p_vals, dp_vals = _legendre_and_deriv(u, n_max + 1)

        # Accumulate acceleration in body frame
        # For degree n (starting from 2), index into jns is n-2
        # V_n = -GM * J_n * R_eq^n * P_n(u) / r^{n+1}
        # a_x = -dV/dx = -C * x * [(n+1)*P_n(u) + u*P_n'(u)]
        # a_y = -dV/dy = -C * y * [(n+1)*P_n(u) + u*P_n'(u)]
        # a_z = -dV/dz = -C * [(n+1)*P_n(u)*z - P_n'(u)*(1-u^2)*r]
        # where C = GM * J_n * R_eq^n / r^{n+3}

        ax = DoubleDouble(jnp.float64(0.0))
        ay = DoubleDouble(jnp.float64(0.0))
        az = DoubleDouble(jnp.float64(0.0))

        x_val = DoubleDouble(x_body.hi[0], x_body.lo[0])
        y_val = DoubleDouble(x_body.hi[1], x_body.lo[1])

        one = DoubleDouble(jnp.float64(1.0))
        u2 = u * u
        sin2_theta = one - u2  # 1 - cos^2(theta)

        for j_idx in range(N_j):
            n = j_idx + 2  # degree
            jn = DoubleDouble(jnp.float64(float(jns[j_idx])))

            # C = GM * Jn * Req^n / r^{n+3}
            # Compute Req^n / r^{n+3} = (Req/r)^n / r^3
            req_over_r = req_dd / r
            ratio_n = one
            for _ in range(n):
                ratio_n = ratio_n * req_over_r
            r3 = r2 * r
            coeff = gm_dd * jn * ratio_n / r3

            pn = p_vals[n]
            dpn = dp_vals[n]

            n_plus_1 = DoubleDouble(jnp.float64(n + 1))
            xy_factor = n_plus_1 * pn + u * dpn

            # a = -grad(V), so a_i = -dV/dx_i
            # dV/dx = C * x * [(n+1)Pn + u*Pn']
            # dV/dy = C * y * [(n+1)Pn + u*Pn']
            # dV/dz = C * [(n+1)*Pn*z - Pn'*(1-u^2)*r]
            ax = ax + coeff * x_val * xy_factor
            ay = ay + coeff * y_val * xy_factor
            az = az + coeff * (n_plus_1 * pn * z - dpn * sin2_theta * r)

        # Rotate acceleration back to original frame: a_orig = a_body @ rot_matrix.T
        a_body_hi = jnp.array([ax.hi, ay.hi, az.hi])
        a_body_lo = jnp.array([ax.lo, ay.lo, az.lo])

        a_orig_hi = jnp.zeros(3)
        a_orig_lo = jnp.zeros(3)
        for i in range(3):
            s = DoubleDouble(jnp.float64(0.0))
            for j in range(3):
                aj = DoubleDouble(a_body_hi[j], a_body_lo[j])
                rji = DoubleDouble(rot_T_dd.hi[j, i], rot_T_dd.lo[j, i])
                s = s + aj * rji
            a_orig_hi = a_orig_hi.at[i].set(s.hi)
            a_orig_lo = a_orig_lo.at[i].set(s.lo)

        result_hi = result_hi.at[ip].set(a_orig_hi)
        result_lo = result_lo.at[ip].set(a_orig_lo)

    return DoubleDouble(result_hi, result_lo)

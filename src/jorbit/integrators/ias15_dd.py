import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from functools import partial

from jorbit.utils.doubledouble import DoubleDouble, dd_sum, dd_max, dd_norm
from jorbit.utils.generate_coefficients import create_iasnn_constants
from jorbit.data.constants import IASNN_DD_EPSILON


# not jitted, not using pure jax here
def setup_iasnn_integrator(n_internal_points):

    # taylor expansion coefficients
    b_x_denoms = (1.0 + jnp.arange(1, n_internal_points + 1, 1, dtype=jnp.float64)) * (
        2.0 + jnp.arange(1, n_internal_points + 1, 1, dtype=jnp.float64)
    )
    b_v_denoms = jnp.arange(2, n_internal_points + 2, 1, dtype=jnp.float64)

    # generate the constant arrays- here they're lists of mpmath.mpf objects
    h, r, c, d = create_iasnn_constants(n_internal_points)

    # convert h to a DoubleDouble, nothing fancy needed
    his = jnp.array([DoubleDouble.from_string(str(x)).hi for x in h])
    los = jnp.array([DoubleDouble.from_string(str(x)).lo for x in h])
    h = DoubleDouble(his, los)

    # same for c
    his = jnp.array([DoubleDouble.from_string(str(x)).hi for x in c])
    los = jnp.array([DoubleDouble.from_string(str(x)).lo for x in c])
    c = DoubleDouble(his, los)

    # same for r, except we only ever need the inverses,
    # so might as well do the division here
    his = jnp.array([DoubleDouble.from_string(str(x)).hi for x in r])
    los = jnp.array([DoubleDouble.from_string(str(x)).lo for x in r])
    r = DoubleDouble(his, los)
    r = DoubleDouble(1.0) / r

    # convert d into a matrix to initialize the g coefficients
    his = jnp.array([DoubleDouble.from_string(str(x)).hi for x in d])
    los = jnp.array([DoubleDouble.from_string(str(x)).lo for x in d])
    d = DoubleDouble(his, los)

    d_matrix_his = jnp.zeros((n_internal_points, n_internal_points))
    indices = jnp.tril_indices(n_internal_points, k=-1)
    d_matrix_his = d_matrix_his.at[indices].set(d.hi)
    d_matrix_his = d_matrix_his.at[jnp.diag_indices(n_internal_points)].set(1.0)
    d_matrix_his = d_matrix_his.T

    d_matrix_los = jnp.zeros((n_internal_points, n_internal_points))
    indices = jnp.tril_indices(n_internal_points, k=-1)
    d_matrix_los = d_matrix_los.at[indices].set(d.lo)
    d_matrix_los = d_matrix_los.T

    d = DoubleDouble(d_matrix_his, d_matrix_los)

    return DoubleDouble(b_x_denoms), DoubleDouble(b_v_denoms), h, r, c, d


def _estimate_x_v_from_b(a0, v0, x0, dt, b_x_denoms, b_v_denoms, h, bp):
    # bp is *not* an IAS15Helper, it's just a DoubleDouble w/ shape
    # (n_internal_points, n_particles, 3)
    # aiming to stay shape-agnostic, enable higher or lower order scheme

    # these are all DoubleDoubles

    xcoeffs = DoubleDouble(
        jnp.zeros((bp.hi.shape[0] + 3, bp.hi.shape[1], bp.hi.shape[2]))
    )
    xcoeffs[3:] = bp * dt * dt / b_x_denoms[:, None, None]
    xcoeffs[2] = a0 * dt * dt / DoubleDouble(2.0)
    xcoeffs[1] = v0 * dt
    xcoeffs[0] = x0
    xcoeffs = xcoeffs[::-1]
    return xcoeffs

    new_x_init = DoubleDouble(jnp.zeros(xcoeffs.hi.shape[1:]))
    estimated_x, _ = jax.lax.scan(lambda y, _p: (y * h + _p, None), new_x_init, xcoeffs)

    vcoeffs = DoubleDouble(
        jnp.zeros((bp.hi.shape[0] + 2, bp.hi.shape[1], bp.hi.shape[2]))
    )
    vcoeffs[2:] = bp * dt / b_v_denoms[:, None, None]
    vcoeffs[1] = a0 * dt
    vcoeffs[0] = v0
    vcoeffs = vcoeffs[::-1]

    new_v_init = DoubleDouble(jnp.zeros(vcoeffs.hi.shape[1:]))
    estimated_v, _ = jax.lax.scan(lambda y, _p: (y * h + _p, None), new_v_init, vcoeffs)

    return estimated_x, estimated_v


@partial(jax.jit, static_argnums=(0,))
def refine_intermediate_g(substep_num, g, r, at, a0):
    # substep_num starts at 1, 1->h1, etc
    substep_num -= 1

    def scan_body(carry, idx):
        result, start_pos = carry
        result = (result - g[idx]) * r[start_pos + idx + 1]
        return (result, start_pos), result

    start_pos = (substep_num * (substep_num + 1)) // 2
    initial_result = (at - a0) * r[start_pos]
    indices = jnp.arange(substep_num)
    (final_result, _), _ = jax.lax.scan(scan_body, (initial_result, start_pos), indices)
    return final_result


@partial(jax.jit, static_argnums=(6, 7))
def _refine_b_and_g(r, c, b, g, at, a0, substep_num, return_g_diff):
    old_g = g
    new_g = refine_intermediate_g(substep_num=substep_num, g=g, r=r, at=at, a0=a0)
    g_diff = new_g - old_g[substep_num - 1]
    g[substep_num - 1] = new_g

    c_start = (substep_num - 1) * (substep_num - 2) // 2
    c_vals = DoubleDouble(jnp.ones(substep_num, dtype=jnp.float64))
    c_vals[: substep_num - 1] = c[c_start : c_start + substep_num - 1]

    def scan_func(carry, scan_over):
        b_array = carry
        idx = scan_over
        b_array[idx, :] = b_array[idx] + (g_diff * c_vals[idx])
        return b_array, None

    b_indices = jnp.arange(substep_num)
    final_b, _ = jax.lax.scan(scan_func, b, b_indices)

    if return_g_diff:
        return final_b, g, g_diff
    return final_b, g


@jax.jit
def acceleration_func(x):
    r = dd_norm(x, axis=1)
    return -x / (r * r * r)


@jax.jit
def step(x0, v0, b, dt, precomputed_setup):
    # these are all just DoubleDouble here- no IAS15Helpers
    # x0, v0, a0 are all (n_particles, 3)
    # b is (n_internal_points, n_particles, 3)

    b_x_denoms, b_v_denoms, h, r, c, d_matrix = precomputed_setup

    # TODO
    t_beginning = DoubleDouble(0.0)
    a0 = acceleration_func(x0)

    # misc setup, make partialized versions of the helpers that bake in the constants
    n_internal_points = b.hi.shape[0]
    estimate_x_v_from_b = jax.tree_util.Partial(
        _estimate_x_v_from_b, a0, v0, x0, dt, b_x_denoms, b_v_denoms
    )
    refine_b_and_g = jax.tree_util.Partial(_refine_b_and_g, r, c)

    # initialize the g coefficients
    g = dd_sum((b[None, :, :, :] * d_matrix[:, :, None, None]), axis=1)

    # get the initial acceleration
    a0 = acceleration_func(x0)

    # set up the predictor-corrector loop
    def do_nothing(b, g, predictor_corrector_error):
        # jax.debug.print("just chillin")
        return b, g, predictor_corrector_error, predictor_corrector_error

    def predictor_corrector_iteration(b, g, predictor_corrector_error):
        predictor_corrector_error_last = predictor_corrector_error
        predictor_corrector_error = 0.0

        # loop over each subinterval
        ################################################################################
        # really not loving this as a for loop, worried about compile times
        # but, struggled too long to get it to work with scan, even if things are marked
        # static in the subfunctions the scan doesn't like that
        for n in range(1, n_internal_points):
            step_time = t_beginning + dt * h[n]
            x, v = estimate_x_v_from_b(h[n], b)
            at = acceleration_func(x)
            b, g = refine_b_and_g(b, g, at, a0, n, False)

        # last iteration is different only so we can get the change in the last g value
        # to evaluate convergence
        n = n_internal_points
        step_time = t_beginning + dt * h[n]
        x, v = estimate_x_v_from_b(h[n], b)
        at = acceleration_func(x)
        b, g, g_diff = refine_b_and_g(
            b=b, g=g, at=at, a0=a0, substep_num=n, return_g_diff=True
        )

        maxa = dd_max(abs(at))  # abs is overloaded for DoubleDouble
        maxb6tmp = dd_max(abs(g_diff))
        predictor_corrector_error = abs(maxb6tmp / maxa)

        return b, g, predictor_corrector_error, predictor_corrector_error_last

    return predictor_corrector_iteration(b, g, DoubleDouble(1e300))

    def scan_func(carry, scan_over):
        b, g, predictor_corrector_error, predictor_corrector_error_last = carry

        condition = (predictor_corrector_error < IASNN_DD_EPSILON) | (
            (scan_over > 2)
            & (predictor_corrector_error > predictor_corrector_error_last)
        )

        carry = jax.lax.cond(
            condition,
            do_nothing,
            predictor_corrector_iteration,
            b,
            g,
            predictor_corrector_error,
        )

        carry = predictor_corrector_iteration(b, g, predictor_corrector_error)
        # jax.debug.print("{x}, {y}", x=carry[2].hi, y=carry[2].lo)
        return carry, None

    predictor_corrector_error = DoubleDouble(1e300)
    predictor_corrector_error_last = DoubleDouble(2.0)

    (b, g, predictor_corrector_error, predictor_corrector_error_last), _ = jax.lax.scan(
        scan_func,
        (b, g, predictor_corrector_error, predictor_corrector_error_last),
        jnp.arange(10),
    )

    # bits about timescales

    x, v = estimate_x_v_from_b(DoubleDouble(1.0), b)

    return x, v, b

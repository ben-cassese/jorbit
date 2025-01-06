import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.utils.doubledouble import DoubleDouble, dd_sum
from jorbit.utils.generate_coefficients import create_iasnn_constants


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

    return DoubleDouble(b_x_denoms), DoubleDouble(b_v_denoms), h, r, d


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


def acceleration_func(x):
    return -x


@jax.jit
def step(x0, v0, a0, b, precompued_setup):
    # these are all just DoubleDouble here- no IAS15Helpers
    # x0, v0, a0 are all (n_particles, 3)
    # b is (n_internal_points, n_particles, 3)

    b_x_denoms, b_v_denoms, h, r, d_matrix = precompued_setup

    # TODO
    t_beginning = DoubleDouble(0.0)
    dt = DoubleDouble(0.01)
    # end TODO

    n_internal_points = b.hi.shape[0]
    estimate_x_v_from_b = jax.tree_util.Partial(
        _estimate_x_v_from_b, a0, v0, x0, dt, b_x_denoms, b_v_denoms
    )

    # initialize the g coefficients
    g = dd_sum((b[None, :, :, :] * d_matrix[:, :, None, None]), axis=1)

    x_est, v_est = estimate_x_v_from_b(h[-1], b)

    return g, x_est, v_est

    # # set up the predictor-corrector loop
    # def do_nothing(b, g, predictor_corrector_error):
    #     # print("just chillin")
    #     return b, g, predictor_corrector_error, predictor_corrector_error

    # def predictor_corrector_iteration(b, g, predictor_corrector_error):
    #     predictor_corrector_error_last = predictor_corrector_error
    #     predictor_corrector_error = 0.0

    #     # loop over each subinterval
    #     ################################################################################
    #     n = 1
    #     step_time = t_beginning + dt * IAS15_H1
    #     # get the new acceleration value at predicted position
    #     x, v = estimate_x_v_from_b(IAS15_H1, b)
    #     # acc_state = SystemState(
    #     #     massive_positions=x[:M],
    #     #     massive_velocities=v[:M],
    #     #     tracer_positions=x[M:],
    #     #     tracer_velocities=v[M:],
    #     #     log_gms=initial_system_state.log_gms,
    #     #     time=step_time,
    #     #     acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
    #     # )
    #     # at = acceleration_func(acc_state)
    #     at = acceleration_func(x)

    #     tmp = g.p0
    #     gk = at - a0
    #     g.p0 = gk * inv_IAS15_RR00
    #     b.p0 = b.p0 + g.p0 - tmp

    #     ################################################################################
    #     n = 2
    #     step_time = t_beginning + dt * IAS15_H2
    #     x, v = estimate_x_v_from_b(IAS15_H2, b)
    #     # acc_state = SystemState(
    #     #     massive_positions=x[:M],
    #     #     massive_velocities=v[:M],
    #     #     tracer_positions=x[M:],
    #     #     tracer_velocities=v[M:],
    #     #     log_gms=initial_system_state.log_gms,
    #     #     time=step_time,
    #     #     acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
    #     # )
    #     # at = acceleration_func(acc_state)
    #     at = acceleration_func(x)

    #     tmp = g.p1
    #     gk = at - a0
    #     g.p1 = (gk * inv_IAS15_RR01 - g.p0) * inv_IAS15_RR02
    #     tmp = g.p1 - tmp
    #     b.p0 = b.p0 + tmp * IAS15_C00
    #     b.p1 = b.p1 + tmp

    #     ################################################################################
    #     n = 3
    #     step_time = t_beginning + dt * IAS15_H3
    #     x, v = estimate_x_v_from_b(IAS15_H3, b)
    #     # acc_state = SystemState(
    #     #     massive_positions=x[:M],
    #     #     massive_velocities=v[:M],
    #     #     tracer_positions=x[M:],
    #     #     tracer_velocities=v[M:],
    #     #     log_gms=initial_system_state.log_gms,
    #     #     time=step_time,
    #     #     acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
    #     # )
    #     # at = acceleration_func(acc_state)
    #     at = acceleration_func(x)

    #     tmp = g.p2
    #     gk = at - a0
    #     g.p2 = ((gk * inv_IAS15_RR03 - g.p0) * inv_IAS15_RR04 - g.p1) * inv_IAS15_RR05
    #     tmp = g.p2 - tmp
    #     b.p0 = b.p0 + tmp * IAS15_C01
    #     b.p1 = b.p1 + tmp * IAS15_C02
    #     b.p2 = b.p2 + tmp

    #     ################################################################################
    #     n = 4
    #     step_time = t_beginning + dt * IAS15_H4
    #     x, v = estimate_x_v_from_b(IAS15_H4, b)
    #     # acc_state = SystemState(
    #     #     massive_positions=x[:M],
    #     #     massive_velocities=v[:M],
    #     #     tracer_positions=x[M:],
    #     #     tracer_velocities=v[M:],
    #     #     log_gms=initial_system_state.log_gms,
    #     #     time=step_time,
    #     #     acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
    #     # )
    #     # at = acceleration_func(acc_state)
    #     at = acceleration_func(x)

    #     tmp = g.p3
    #     gk = at - a0
    #     g.p3 = (
    #         ((gk * inv_IAS15_RR06 - g.p0) * inv_IAS15_RR07 - g.p1) * inv_IAS15_RR08
    #         - g.p2
    #     ) * inv_IAS15_RR09
    #     tmp = g.p3 - tmp
    #     b.p0 = b.p0 + tmp * IAS15_C03
    #     b.p1 = b.p1 + tmp * IAS15_C04
    #     b.p2 = b.p2 + tmp * IAS15_C05
    #     b.p3 = b.p3 + tmp

    #     ################################################################################
    #     n = 5
    #     step_time = t_beginning + dt * IAS15_H5
    #     x, v = estimate_x_v_from_b(IAS15_H5, b)
    #     # acc_state = SystemState(
    #     #     massive_positions=x[:M],
    #     #     massive_velocities=v[:M],
    #     #     tracer_positions=x[M:],
    #     #     tracer_velocities=v[M:],
    #     #     log_gms=initial_system_state.log_gms,
    #     #     time=step_time,
    #     #     acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
    #     # )
    #     # at = acceleration_func(acc_state)
    #     at = acceleration_func(x)

    #     tmp = g.p4
    #     gk = at - a0
    #     g.p4 = (
    #         (
    #             ((gk * inv_IAS15_RR10 - g.p0) * inv_IAS15_RR11 - g.p1) * inv_IAS15_RR12
    #             - g.p2
    #         )
    #         * inv_IAS15_RR13
    #         - g.p3
    #     ) * inv_IAS15_RR14
    #     tmp = g.p4 - tmp
    #     b.p0 = b.p0 + tmp * IAS15_C06
    #     b.p1 = b.p1 + tmp * IAS15_C07
    #     b.p2 = b.p2 + tmp * IAS15_C08
    #     b.p3 = b.p3 + tmp * IAS15_C09
    #     b.p4 = b.p4 + tmp

    #     ################################################################################
    #     n = 6
    #     step_time = t_beginning + dt * IAS15_H6
    #     x, v = estimate_x_v_from_b(IAS15_H6, b)
    #     # acc_state = SystemState(
    #     #     massive_positions=x[:M],
    #     #     massive_velocities=v[:M],
    #     #     tracer_positions=x[M:],
    #     #     tracer_velocities=v[M:],
    #     #     log_gms=initial_system_state.log_gms,
    #     #     time=step_time,
    #     #     acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
    #     # )
    #     # at = acceleration_func(acc_state)
    #     at = acceleration_func(x)

    #     tmp = g.p5
    #     gk = at - a0
    #     g.p5 = (
    #         (
    #             (
    #                 ((gk * inv_IAS15_RR15 - g.p0) * inv_IAS15_RR16 - g.p1)
    #                 * inv_IAS15_RR17
    #                 - g.p2
    #             )
    #             * inv_IAS15_RR18
    #             - g.p3
    #         )
    #         * inv_IAS15_RR19
    #         - g.p4
    #     ) * inv_IAS15_RR20
    #     tmp = g.p5 - tmp
    #     b.p0 = b.p0 + tmp * IAS15_C10
    #     b.p1 = b.p1 + tmp * IAS15_C11
    #     b.p2 = b.p2 + tmp * IAS15_C12
    #     b.p3 = b.p3 + tmp * IAS15_C13
    #     b.p4 = b.p4 + tmp * IAS15_C14
    #     b.p5 = b.p5 + tmp

    #     ################################################################################
    #     n = 7
    #     step_time = t_beginning + dt * IAS15_H7
    #     x, v = estimate_x_v_from_b(IAS15_H7, b)
    #     # acc_state = SystemState(
    #     #     massive_positions=x[:M],
    #     #     massive_velocities=v[:M],
    #     #     tracer_positions=x[M:],
    #     #     tracer_velocities=v[M:],
    #     #     log_gms=initial_system_state.log_gms,
    #     #     time=step_time,
    #     #     acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
    #     # )
    #     # at = acceleration_func(acc_state)
    #     at = acceleration_func(x)

    #     tmp = g.p6
    #     gk = at - a0
    #     g.p6 = (
    #         (
    #             (
    #                 (
    #                     ((gk * inv_IAS15_RR21 - g.p0) * inv_IAS15_RR22 - g.p1)
    #                     * inv_IAS15_RR23
    #                     - g.p2
    #                 )
    #                 * inv_IAS15_RR24
    #                 - g.p3
    #             )
    #             * inv_IAS15_RR25
    #             - g.p4
    #         )
    #         * inv_IAS15_RR26
    #         - g.p5
    #     ) * inv_IAS15_RR27
    #     tmp = g.p6 - tmp
    #     b.p0 = b.p0 + tmp * IAS15_C15
    #     b.p1 = b.p1 + tmp * IAS15_C16
    #     b.p2 = b.p2 + tmp * IAS15_C17
    #     b.p3 = b.p3 + tmp * IAS15_C18
    #     b.p4 = b.p4 + tmp * IAS15_C19
    #     b.p5 = b.p5 + tmp * IAS15_C20
    #     b.p6 = b.p6 + tmp

    #     # maxa = jnp.max(jnp.abs(at))
    #     maxa = DoubleDouble.dd_max(abs(at))  # abs is overloaded for DoubleDouble
    #     # maxb6tmp = jnp.max(jnp.abs(tmp))
    #     maxb6tmp = DoubleDouble.dd_max(abs(tmp))

    #     # predictor_corrector_error = jnp.abs(maxb6tmp / maxa)
    #     predictor_corrector_error = abs(maxb6tmp / maxa)

    #     return b, g, predictor_corrector_error, predictor_corrector_error_last

    # def scan_func(carry, scan_over):
    #     b, g, predictor_corrector_error, predictor_corrector_error_last = carry

    #     # condition = (predictor_corrector_error < EPSILON) | (
    #     #     (scan_over > 2)
    #     #     & (predictor_corrector_error > predictor_corrector_error_last)
    #     # )

    #     # carry = jax.lax.cond(
    #     #     condition,
    #     #     do_nothing,
    #     #     predictor_corrector_iteration,
    #     #     b,
    #     #     csb,
    #     #     g,
    #     #     predictor_corrector_error,
    #     # )

    #     carry = predictor_corrector_iteration(b, g, predictor_corrector_error)
    #     jax.debug.print("{x}, {y}", x=carry[2].hi, y=carry[2].lo)
    #     return carry, None

    # predictor_corrector_error = DoubleDouble(1e300)
    # predictor_corrector_error_last = DoubleDouble(2.0)

    # (b, g, predictor_corrector_error, predictor_corrector_error_last), _ = jax.lax.scan(
    #     scan_func,
    #     (b, g, predictor_corrector_error, predictor_corrector_error_last),
    #     jnp.arange(10),
    # )

    # return b, g, predictor_corrector_error, predictor_corrector_error_last

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.utils.doubledouble import DoubleDouble, dd_sum
from jorbit.utils.generate_coefficients import create_iasnn_constants

# IAS15_RR00 = DoubleDouble.from_string("0.05626256053692214646565219103180")
# IAS15_RR01 = DoubleDouble.from_string("0.18024069173689236498757994278000")
# IAS15_RR02 = DoubleDouble.from_string("0.12397813119997021852192775174820")
# IAS15_RR03 = DoubleDouble.from_string("0.35262471711316963737390776964800")
# IAS15_RR04 = DoubleDouble.from_string("0.29636215657624749090825557861620")
# IAS15_RR05 = DoubleDouble.from_string("0.17238402537627727238632782686800")
# IAS15_RR06 = DoubleDouble.from_string("0.54715362633055538300144855476600")
# IAS15_RR07 = DoubleDouble.from_string("0.49089106579363323653579636373420")
# IAS15_RR08 = DoubleDouble.from_string("0.36691293459366301801386861198600")
# IAS15_RR09 = DoubleDouble.from_string("0.19452890921738574562754078511800")
# IAS15_RR10 = DoubleDouble.from_string("0.73421017721541053152321060555800")
# IAS15_RR11 = DoubleDouble.from_string("0.67794761667848838505755841452620")
# IAS15_RR12 = DoubleDouble.from_string("0.55396948547851816653563066277800")
# IAS15_RR13 = DoubleDouble.from_string("0.38158546010224089414930283591000")
# IAS15_RR14 = DoubleDouble.from_string("0.18705655088485514852176205079200")
# IAS15_RR15 = DoubleDouble.from_string("0.88532094683909576809035977103000")
# IAS15_RR16 = DoubleDouble.from_string("0.82905838630217362162470757999820")
# IAS15_RR17 = DoubleDouble.from_string("0.70508025510220340310277982825000")
# IAS15_RR18 = DoubleDouble.from_string("0.53269622972592613071645200138200")
# IAS15_RR19 = DoubleDouble.from_string("0.33816732050854038508891121626400")
# IAS15_RR20 = DoubleDouble.from_string("0.15111076962368523656714916547200")
# IAS15_RR21 = DoubleDouble.from_string("0.97752061356128750189117448862600")
# IAS15_RR22 = DoubleDouble.from_string("0.92125805302436535542552229759420")
# IAS15_RR23 = DoubleDouble.from_string("0.79727992182439513690359454584600")
# IAS15_RR24 = DoubleDouble.from_string("0.62489589644811786451726671897800")
# IAS15_RR25 = DoubleDouble.from_string("0.43036698723073211888972593386000")
# IAS15_RR26 = DoubleDouble.from_string("0.24331043634587697036796388306800")
# IAS15_RR27 = DoubleDouble.from_string("0.09219966672219173380081471759600")
# IAS15_C00 = DoubleDouble.from_string("-0.05626256053692214646565219103180")
# IAS15_C01 = DoubleDouble.from_string("0.01014080283006362998648180478597")
# IAS15_C02 = DoubleDouble.from_string("-0.23650325227381451145323213381180")
# IAS15_C03 = DoubleDouble.from_string("-0.00357589772925161759493445889941")
# IAS15_C04 = DoubleDouble.from_string("0.09353769525946206589574846114536")
# IAS15_C05 = DoubleDouble.from_string("-0.58912796938698414882713990345980")
# IAS15_C06 = DoubleDouble.from_string("0.00195656540994722107690056706032")
# IAS15_C07 = DoubleDouble.from_string("-0.05475538688906868644080842943942")
# IAS15_C08 = DoubleDouble.from_string("0.41588120008230686168862191119100")
# IAS15_C09 = DoubleDouble.from_string("-1.13628159571753953182858845822580")
# IAS15_C10 = DoubleDouble.from_string("-0.00143653023637089154244595529986")
# IAS15_C11 = DoubleDouble.from_string("0.04215852772126870770729734703561")
# IAS15_C12 = DoubleDouble.from_string("-0.36009959650205681228976646105759")
# IAS15_C13 = DoubleDouble.from_string("1.25015071184069102585054407510960")
# IAS15_C14 = DoubleDouble.from_string("-1.87049177293295006335179906378380")
# IAS15_C15 = DoubleDouble.from_string("0.00127179030902686774929431161483")
# IAS15_C16 = DoubleDouble.from_string("-0.03876035791590677036990462482059")
# IAS15_C17 = DoubleDouble.from_string("0.36096224345284598322533980803401")
# IAS15_C18 = DoubleDouble.from_string("-1.46688420840042696437015525831000")
# IAS15_C19 = DoubleDouble.from_string("2.90613625930842930142379130729013")
# IAS15_C20 = DoubleDouble.from_string("-2.75581271977204583144215883481380")


# inv_IAS15_RR00 = DoubleDouble(1.0) / IAS15_RR00
# inv_IAS15_RR01 = DoubleDouble(1.0) / IAS15_RR01
# inv_IAS15_RR02 = DoubleDouble(1.0) / IAS15_RR02
# inv_IAS15_RR03 = DoubleDouble(1.0) / IAS15_RR03
# inv_IAS15_RR04 = DoubleDouble(1.0) / IAS15_RR04
# inv_IAS15_RR05 = DoubleDouble(1.0) / IAS15_RR05
# inv_IAS15_RR06 = DoubleDouble(1.0) / IAS15_RR06
# inv_IAS15_RR07 = DoubleDouble(1.0) / IAS15_RR07
# inv_IAS15_RR08 = DoubleDouble(1.0) / IAS15_RR08
# inv_IAS15_RR09 = DoubleDouble(1.0) / IAS15_RR09
# inv_IAS15_RR10 = DoubleDouble(1.0) / IAS15_RR10
# inv_IAS15_RR11 = DoubleDouble(1.0) / IAS15_RR11
# inv_IAS15_RR12 = DoubleDouble(1.0) / IAS15_RR12
# inv_IAS15_RR13 = DoubleDouble(1.0) / IAS15_RR13
# inv_IAS15_RR14 = DoubleDouble(1.0) / IAS15_RR14
# inv_IAS15_RR15 = DoubleDouble(1.0) / IAS15_RR15
# inv_IAS15_RR16 = DoubleDouble(1.0) / IAS15_RR16
# inv_IAS15_RR17 = DoubleDouble(1.0) / IAS15_RR17
# inv_IAS15_RR18 = DoubleDouble(1.0) / IAS15_RR18
# inv_IAS15_RR19 = DoubleDouble(1.0) / IAS15_RR19
# inv_IAS15_RR20 = DoubleDouble(1.0) / IAS15_RR20
# inv_IAS15_RR21 = DoubleDouble(1.0) / IAS15_RR21
# inv_IAS15_RR22 = DoubleDouble(1.0) / IAS15_RR22
# inv_IAS15_RR23 = DoubleDouble(1.0) / IAS15_RR23
# inv_IAS15_RR24 = DoubleDouble(1.0) / IAS15_RR24
# inv_IAS15_RR25 = DoubleDouble(1.0) / IAS15_RR25
# inv_IAS15_RR26 = DoubleDouble(1.0) / IAS15_RR26
# inv_IAS15_RR27 = DoubleDouble(1.0) / IAS15_RR27


# not jitted, not using pure jax here
def setup_iasnn_integrator(n_internal_evals):

    # taylor expansion coefficients
    b_x_denoms = (1.0 + jnp.arange(1, n_internal_evals + 1, 1, dtype=jnp.float64)) * (
        2.0 + jnp.arange(1, n_internal_evals + 1, 1, dtype=jnp.float64)
    )
    b_v_denoms = jnp.arange(2, n_internal_evals + 2, 1, dtype=jnp.float64)

    # generate the constant arrays- here they're lists of mpmath.mpf objects
    h, r, c, d = create_iasnn_constants(n_internal_evals)

    # convert h to a DoubleDouble, nothing fancy needed
    his = jnp.array([DoubleDouble.from_string(str(x)).hi for x in h])
    los = jnp.array([DoubleDouble.from_string(str(x)).lo for x in h])
    h = DoubleDouble(his, los)

    # convert d into a matrix to initialize the g coefficients
    his = jnp.array([DoubleDouble.from_string(str(x)).hi for x in d])
    los = jnp.array([DoubleDouble.from_string(str(x)).lo for x in d])
    d = DoubleDouble(his, los)

    d_matrix_his = jnp.zeros((n_internal_evals, n_internal_evals))
    indices = jnp.tril_indices(n_internal_evals, k=-1)
    d_matrix_his = d_matrix_his.at[indices].set(d.hi)
    d_matrix_his = d_matrix_his.at[jnp.diag_indices(n_internal_evals)].set(1.0)
    d_matrix_his = d_matrix_his.T

    d_matrix_los = jnp.zeros((n_internal_evals, n_internal_evals))
    indices = jnp.tril_indices(n_internal_evals, k=-1)
    d_matrix_los = d_matrix_los.at[indices].set(d.lo)
    d_matrix_los = d_matrix_los.T

    d = DoubleDouble(d_matrix_his, d_matrix_los)

    return DoubleDouble(b_x_denoms), DoubleDouble(b_v_denoms), h, d


def _estimate_x_v_from_b(a0, v0, x0, dt, b_x_denoms, b_v_denoms, h, bp):
    # bp is *not* an IAS15Helper, it's just a DoubleDouble w/ shape
    # (n_internal_evals, n_particles, 3)
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


def acceleration_func(x):
    return -x


@jax.jit
def step(x0, v0, a0, b, precompued_setup):
    # these are all just DoubleDouble here- no IAS15Helpers
    # x0, v0, a0 are all (n_particles, 3)
    # b is (n_internal_evals, n_particles, 3)

    b_x_denoms, b_v_denoms, h, d_matrix = precompued_setup

    # TODO
    t_beginning = DoubleDouble(0.0)
    dt = DoubleDouble(0.01)
    # end TODO

    n_internal_evals = b.hi.shape[0]
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

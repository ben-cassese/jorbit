import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.utils.doubledouble import DoubleDouble

IAS15_H1 = DoubleDouble.from_string("0.0562625605369221464656521910318")
IAS15_H2 = DoubleDouble.from_string("0.180240691736892364987579942780")
IAS15_H3 = DoubleDouble.from_string("0.352624717113169637373907769648")
IAS15_H4 = DoubleDouble.from_string("0.547153626330555383001448554766")
IAS15_H5 = DoubleDouble.from_string("0.734210177215410531523210605558")
IAS15_H6 = DoubleDouble.from_string("0.885320946839095768090359771030")
IAS15_H7 = DoubleDouble.from_string("0.977520613561287501891174488626")
IAS15_RR00 = DoubleDouble.from_string("0.05626256053692214646565219103180")
IAS15_RR01 = DoubleDouble.from_string("0.18024069173689236498757994278000")
IAS15_RR02 = DoubleDouble.from_string("0.12397813119997021852192775174820")
IAS15_RR03 = DoubleDouble.from_string("0.35262471711316963737390776964800")
IAS15_RR04 = DoubleDouble.from_string("0.29636215657624749090825557861620")
IAS15_RR05 = DoubleDouble.from_string("0.17238402537627727238632782686800")
IAS15_RR06 = DoubleDouble.from_string("0.54715362633055538300144855476600")
IAS15_RR07 = DoubleDouble.from_string("0.49089106579363323653579636373420")
IAS15_RR08 = DoubleDouble.from_string("0.36691293459366301801386861198600")
IAS15_RR09 = DoubleDouble.from_string("0.19452890921738574562754078511800")
IAS15_RR10 = DoubleDouble.from_string("0.73421017721541053152321060555800")
IAS15_RR11 = DoubleDouble.from_string("0.67794761667848838505755841452620")
IAS15_RR12 = DoubleDouble.from_string("0.55396948547851816653563066277800")
IAS15_RR13 = DoubleDouble.from_string("0.38158546010224089414930283591000")
IAS15_RR14 = DoubleDouble.from_string("0.18705655088485514852176205079200")
IAS15_RR15 = DoubleDouble.from_string("0.88532094683909576809035977103000")
IAS15_RR16 = DoubleDouble.from_string("0.82905838630217362162470757999820")
IAS15_RR17 = DoubleDouble.from_string("0.70508025510220340310277982825000")
IAS15_RR18 = DoubleDouble.from_string("0.53269622972592613071645200138200")
IAS15_RR19 = DoubleDouble.from_string("0.33816732050854038508891121626400")
IAS15_RR20 = DoubleDouble.from_string("0.15111076962368523656714916547200")
IAS15_RR21 = DoubleDouble.from_string("0.97752061356128750189117448862600")
IAS15_RR22 = DoubleDouble.from_string("0.92125805302436535542552229759420")
IAS15_RR23 = DoubleDouble.from_string("0.79727992182439513690359454584600")
IAS15_RR24 = DoubleDouble.from_string("0.62489589644811786451726671897800")
IAS15_RR25 = DoubleDouble.from_string("0.43036698723073211888972593386000")
IAS15_RR26 = DoubleDouble.from_string("0.24331043634587697036796388306800")
IAS15_RR27 = DoubleDouble.from_string("0.09219966672219173380081471759600")
IAS15_C00 = DoubleDouble.from_string("-0.05626256053692214646565219103180")
IAS15_C01 = DoubleDouble.from_string("0.01014080283006362998648180478597")
IAS15_C02 = DoubleDouble.from_string("-0.23650325227381451145323213381180")
IAS15_C03 = DoubleDouble.from_string("-0.00357589772925161759493445889941")
IAS15_C04 = DoubleDouble.from_string("0.09353769525946206589574846114536")
IAS15_C05 = DoubleDouble.from_string("-0.58912796938698414882713990345980")
IAS15_C06 = DoubleDouble.from_string("0.00195656540994722107690056706032")
IAS15_C07 = DoubleDouble.from_string("-0.05475538688906868644080842943942")
IAS15_C08 = DoubleDouble.from_string("0.41588120008230686168862191119100")
IAS15_C09 = DoubleDouble.from_string("-1.13628159571753953182858845822580")
IAS15_C10 = DoubleDouble.from_string("-0.00143653023637089154244595529986")
IAS15_C11 = DoubleDouble.from_string("0.04215852772126870770729734703561")
IAS15_C12 = DoubleDouble.from_string("-0.36009959650205681228976646105759")
IAS15_C13 = DoubleDouble.from_string("1.25015071184069102585054407510960")
IAS15_C14 = DoubleDouble.from_string("-1.87049177293295006335179906378380")
IAS15_C15 = DoubleDouble.from_string("0.00127179030902686774929431161483")
IAS15_C16 = DoubleDouble.from_string("-0.03876035791590677036990462482059")
IAS15_C17 = DoubleDouble.from_string("0.36096224345284598322533980803401")
IAS15_C18 = DoubleDouble.from_string("-1.46688420840042696437015525831000")
IAS15_C19 = DoubleDouble.from_string("2.90613625930842930142379130729013")
IAS15_C20 = DoubleDouble.from_string("-2.75581271977204583144215883481380")
IAS15_D00 = DoubleDouble.from_string("0.05626256053692214646565219103180")
IAS15_D01 = DoubleDouble.from_string("0.00316547571817082924999048003940")
IAS15_D02 = DoubleDouble.from_string("0.23650325227381451145323213381180")
IAS15_D03 = DoubleDouble.from_string("0.00017809776922174338811252792197")
IAS15_D04 = DoubleDouble.from_string("0.04579298550602791889545387301118")
IAS15_D05 = DoubleDouble.from_string("0.58912796938698414882713990345980")
IAS15_D06 = DoubleDouble.from_string("0.00001002023652232912720956721522")
IAS15_D07 = DoubleDouble.from_string("0.00843185715352570154449997416277")
IAS15_D08 = DoubleDouble.from_string("0.25353406905456926652146159710636")
IAS15_D09 = DoubleDouble.from_string("1.13628159571753953182858845822580")
IAS15_D10 = DoubleDouble.from_string("0.00000056376416393182076103838501")
IAS15_D11 = DoubleDouble.from_string("0.00152978400250046581894900795890")
IAS15_D12 = DoubleDouble.from_string("0.09783423653244400536536483964234")
IAS15_D13 = DoubleDouble.from_string("0.87525466468409109122972458836859")
IAS15_D14 = DoubleDouble.from_string("1.87049177293295006335179906378380")
IAS15_D15 = DoubleDouble.from_string("0.00000003171881540176136647585482")
IAS15_D16 = DoubleDouble.from_string("0.00027629309098264765931302263937")
IAS15_D17 = DoubleDouble.from_string("0.03602855398373645960038707412661")
IAS15_D18 = DoubleDouble.from_string("0.57673300027707873135445960613389")
IAS15_D19 = DoubleDouble.from_string("2.24858876076915979339268952600833")
IAS15_D20 = DoubleDouble.from_string("2.75581271977204583144215883481380")


# H1AX = 7.0 * IAS15_H1 / 9.0
# H1BX = 3.0 * IAS15_H1 / 4.0
# H1CX = 5.0 * IAS15_H1 / 7.0
# H1DX = 2.0 * IAS15_H1 / 3.0
# H1EX = 3.0 * IAS15_H1 / 5.0
# H1FX = IAS15_H1 / 2.0
# H1GX = IAS15_H1 / 3.0
# H1AV = 7.0 * IAS15_H1 / 8.0
# H1BV = 6.0 * IAS15_H1 / 7.0
# H1CV = 5.0 * IAS15_H1 / 6.0
# H1DV = 4.0 * IAS15_H1 / 5.0
# H1EV = 3.0 * IAS15_H1 / 4.0
# H1FV = 2.0 * IAS15_H1 / 3.0
# H2AX = 7.0 * IAS15_H2 / 9.0
# H2BX = 3.0 * IAS15_H2 / 4.0
# H2CX = 5.0 * IAS15_H2 / 7.0
# H2DX = 2.0 * IAS15_H2 / 3.0
# H2EX = 3.0 * IAS15_H2 / 5.0
# H2FX = IAS15_H2 / 2.0
# H2GX = IAS15_H2 / 3.0
# H2AV = 7.0 * IAS15_H2 / 8.0
# H2BV = 6.0 * IAS15_H2 / 7.0
# H2CV = 5.0 * IAS15_H2 / 6.0
# H2DV = 4.0 * IAS15_H2 / 5.0
# H2EV = 3.0 * IAS15_H2 / 4.0
# H2FV = 2.0 * IAS15_H2 / 3.0
# H3AX = 7.0 * IAS15_H3 / 9.0
# H3BX = 3.0 * IAS15_H3 / 4.0
# H3CX = 5.0 * IAS15_H3 / 7.0
# H3DX = 2.0 * IAS15_H3 / 3.0
# H3EX = 3.0 * IAS15_H3 / 5.0
# H3FX = IAS15_H3 / 2.0
# H3GX = IAS15_H3 / 3.0
# H3AV = 7.0 * IAS15_H3 / 8.0
# H3BV = 6.0 * IAS15_H3 / 7.0
# H3CV = 5.0 * IAS15_H3 / 6.0
# H3DV = 4.0 * IAS15_H3 / 5.0
# H3EV = 3.0 * IAS15_H3 / 4.0
# H3FV = 2.0 * IAS15_H3 / 3.0
# H4AX = 7.0 * IAS15_H4 / 9.0
# H4BX = 3.0 * IAS15_H4 / 4.0
# H4CX = 5.0 * IAS15_H4 / 7.0
# H4DX = 2.0 * IAS15_H4 / 3.0
# H4EX = 3.0 * IAS15_H4 / 5.0
# H4FX = IAS15_H4 / 2.0
# H4GX = IAS15_H4 / 3.0
# H4AV = 7.0 * IAS15_H4 / 8.0
# H4BV = 6.0 * IAS15_H4 / 7.0
# H4CV = 5.0 * IAS15_H4 / 6.0
# H4DV = 4.0 * IAS15_H4 / 5.0
# H4EV = 3.0 * IAS15_H4 / 4.0
# H4FV = 2.0 * IAS15_H4 / 3.0
# H5AX = 7.0 * IAS15_H5 / 9.0
# H5BX = 3.0 * IAS15_H5 / 4.0
# H5CX = 5.0 * IAS15_H5 / 7.0
# H5DX = 2.0 * IAS15_H5 / 3.0
# H5EX = 3.0 * IAS15_H5 / 5.0
# H5FX = IAS15_H5 / 2.0
# H5GX = IAS15_H5 / 3.0
# H5AV = 7.0 * IAS15_H5 / 8.0
# H5BV = 6.0 * IAS15_H5 / 7.0
# H5CV = 5.0 * IAS15_H5 / 6.0
# H5DV = 4.0 * IAS15_H5 / 5.0
# H5EV = 3.0 * IAS15_H5 / 4.0
# H5FV = 2.0 * IAS15_H5 / 3.0
# H6AX = 7.0 * IAS15_H6 / 9.0
# H6BX = 3.0 * IAS15_H6 / 4.0
# H6CX = 5.0 * IAS15_H6 / 7.0
# H6DX = 2.0 * IAS15_H6 / 3.0
# H6EX = 3.0 * IAS15_H6 / 5.0
# H6FX = IAS15_H6 / 2.0
# H6GX = IAS15_H6 / 3.0
# H6AV = 7.0 * IAS15_H6 / 8.0
# H6BV = 6.0 * IAS15_H6 / 7.0
# H6CV = 5.0 * IAS15_H6 / 6.0
# H6DV = 4.0 * IAS15_H6 / 5.0
# H6EV = 3.0 * IAS15_H6 / 4.0
# H6FV = 2.0 * IAS15_H6 / 3.0
# H7AX = 7.0 * IAS15_H7 / 9.0
# H7BX = 3.0 * IAS15_H7 / 4.0
# H7CX = 5.0 * IAS15_H7 / 7.0
# H7DX = 2.0 * IAS15_H7 / 3.0
# H7EX = 3.0 * IAS15_H7 / 5.0
# H7FX = IAS15_H7 / 2.0
# H7GX = IAS15_H7 / 3.0
# H7AV = 7.0 * IAS15_H7 / 8.0
# H7BV = 6.0 * IAS15_H7 / 7.0
# H7CV = 5.0 * IAS15_H7 / 6.0
# H7DV = 4.0 * IAS15_H7 / 5.0
# H7EV = 3.0 * IAS15_H7 / 4.0
# H7FV = 2.0 * IAS15_H7 / 3.0


@jax.tree_util.register_pytree_node_class
class IAS15Helper:
    # the equivalent of the reb_dp7 struct in rebound, but obviously without pointers
    # kinda just a spicy dictionary, not sure if this is how I want to do it
    def __init__(self, p0, p1, p2, p3, p4, p5, p6):
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.p5 = p5
        self.p6 = p6

    def tree_flatten(self):
        children = (
            self.p0,
            self.p1,
            self.p2,
            self.p3,
            self.p4,
            self.p5,
            self.p6,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def initialize_ias15_helper(n_particles):
    return IAS15Helper(
        p0=DoubleDouble(jnp.zeros((n_particles, 3), dtype=jnp.float64)),
        p1=DoubleDouble(jnp.zeros((n_particles, 3), dtype=jnp.float64)),
        p2=DoubleDouble(jnp.zeros((n_particles, 3), dtype=jnp.float64)),
        p3=DoubleDouble(jnp.zeros((n_particles, 3), dtype=jnp.float64)),
        p4=DoubleDouble(jnp.zeros((n_particles, 3), dtype=jnp.float64)),
        p5=DoubleDouble(jnp.zeros((n_particles, 3), dtype=jnp.float64)),
        p6=DoubleDouble(jnp.zeros((n_particles, 3), dtype=jnp.float64)),
    )


def acceleration_func(x):
    return -x


def step(x0, v0, b):

    g = initialize_ias15_helper(1)

    # get the initial g terms from the b terms
    g.p0 = (
        b.p6 * IAS15_D15
        + b.p5 * IAS15_D10
        + b.p4 * IAS15_D06
        + b.p3 * IAS15_D03
        + b.p2 * IAS15_D01
        + b.p1 * IAS15_D00
        + b.p0
    )
    g.p1 = (
        b.p6 * IAS15_D16
        + b.p5 * IAS15_D11
        + b.p4 * IAS15_D07
        + b.p3 * IAS15_D04
        + b.p2 * IAS15_D02
        + b.p1
    )
    g.p2 = (
        b.p6 * IAS15_D17 + b.p5 * IAS15_D12 + b.p4 * IAS15_D08 + b.p3 * IAS15_D05 + b.p2
    )
    g.p3 = b.p6 * IAS15_D18 + b.p5 * IAS15_D13 + b.p4 * IAS15_D09 + b.p3
    g.p4 = b.p6 * IAS15_D19 + b.p5 * IAS15_D14 + b.p4
    g.p5 = b.p6 * IAS15_D20 + b.p5
    g.p6 = b.p6

    return g

    # set up the predictor-corrector loop
    def do_nothing(b, g, predictor_corrector_error):
        # print("just chillin")
        return b, g, predictor_corrector_error, predictor_corrector_error

    def predictor_corrector_iteration(b, csb, g, predictor_corrector_error):
        # loop over each subinterval
        ################################################################################
        n = 1
        step_time = t_beginning + dt * IAS15_H1
        # get the new acceleration value at predicted position
        # fmt: off
        x = x0 + ((((((((b.p6*H1AX + b.p5)*H1BX + b.p4)*H1CX + b.p3)*H1DX + b.p2)*H1EX + b.p1)*H1FX + b.p0)* + a0)*dt*H1FX + v0)*dt*IAS15_H1
        v = v0 + (((((((b.p6*H1AV + b.p5)*H1BV + b.p4)*H1CV + b.p3)*H1DV + b.p2)*H1EV + b.p1)*H1FV + b.p0)*H1FX + a0)*dt*IAS15_H1
        # fmt: on

        at = acceleration_func(x)

        tmp = g.p0
        gk = at - a0
        g.p0 = gk / IAS15_RR[0]
        b.p0, csb.p0 = add_cs(b.p0, csb.p0, g.p0 - tmp)

        ################################################################################
        n = 2
        step_time = t_beginning + dt * IAS15_H[n]
        # fmt: off
        x = x0 + ((((((((b.p6*H2AX + b.p5)*H2BX + b.p4)*H2CX + b.p3)*H2DX + b.p2)*H2EX + b.p1)*H2FX + b.p0)* + a0)*dt*H2FX + v0)*dt*IAS15_H2
        v = v0 + (((((((b.p6*H2AV + b.p5)*H2BV + b.p4)*H2CV + b.p3)*H2DV + b.p2)*H2EV + b.p1)*H2FV + b.p0)*H2FX + a0)*dt*IAS15_H2
        # fmt: on
        acc_state = SystemState(
            massive_positions=x[:M],
            massive_velocities=v[:M],
            tracer_positions=x[M:],
            tracer_velocities=v[M:],
            log_gms=initial_system_state.log_gms,
            time=step_time,
            acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
        )
        at = acceleration_func(acc_state)

        tmp = g.p1
        gk = at - a0
        g.p1 = (gk / IAS15_RR[1] - g.p0) / IAS15_RR[2]
        tmp = g.p1 - tmp
        b.p0, csb.p0 = add_cs(b.p0, csb.p0, tmp * IAS15_C[0])
        b.p1, csb.p1 = add_cs(b.p1, csb.p1, tmp)

        ################################################################################
        n = 3
        step_time = t_beginning + dt * IAS15_H[n]
        # fmt: off
        x = x0 + ((((((((b.p6*H3AX + b.p5)*H3BX + b.p4)*H3CX + b.p3)*H3DX + b.p2)*H3EX + b.p1)*H3FX + b.p0)* + a0)*dt*H3FX + v0)*dt*IAS15_H3
        v = v0 + (((((((b.p6*H3AV + b.p5)*H3BV + b.p4)*H3CV + b.p3)*H3DV + b.p2)*H3EV + b.p1)*H3FV + b.p0)*H3FX + a0)*dt*IAS15_H3
        # fmt: on
        acc_state = SystemState(
            massive_positions=x[:M],
            massive_velocities=v[:M],
            tracer_positions=x[M:],
            tracer_velocities=v[M:],
            log_gms=initial_system_state.log_gms,
            time=step_time,
            acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
        )
        at = acceleration_func(acc_state)

        tmp = g.p2
        gk = at - a0
        g.p2 = ((gk / IAS15_RR[3] - g.p0) / IAS15_RR[4] - g.p1) / IAS15_RR[5]
        tmp = g.p2 - tmp
        b.p0, csb.p0 = add_cs(b.p0, csb.p0, tmp * IAS15_C[1])
        b.p1, csb.p1 = add_cs(b.p1, csb.p1, tmp * IAS15_C[2])
        b.p2, csb.p2 = add_cs(b.p2, csb.p2, tmp)

        ################################################################################
        n = 4
        step_time = t_beginning + dt * IAS15_H[n]
        # fmt: off
        x = x0 + ((((((((b.p6*H4AX + b.p5)*H4BX + b.p4)*H4CX + b.p3)*H4DX + b.p2)*H4EX + b.p1)*H4FX + b.p0)* + a0)*dt*H4FX + v0)*dt*IAS15_H4
        v = v0 + (((((((b.p6*H4AV + b.p5)*H4BV + b.p4)*H4CV + b.p3)*H4DV + b.p2)*H4EV + b.p1)*H4FV + b.p0)*H4FX + a0)*dt*IAS15_H4
        # fmt: on
        acc_state = SystemState(
            massive_positions=x[:M],
            massive_velocities=v[:M],
            tracer_positions=x[M:],
            tracer_velocities=v[M:],
            log_gms=initial_system_state.log_gms,
            time=step_time,
            acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
        )
        at = acceleration_func(acc_state)

        tmp = g.p3
        gk = at - a0
        g.p3 = (
            ((gk / IAS15_RR[6] - g.p0) / IAS15_RR[7] - g.p1) / IAS15_RR[8] - g.p2
        ) / IAS15_RR[9]
        tmp = g.p3 - tmp
        b.p0, csb.p0 = add_cs(b.p0, csb.p0, tmp * IAS15_C[3])
        b.p1, csb.p1 = add_cs(b.p1, csb.p1, tmp * IAS15_C[4])
        b.p2, csb.p2 = add_cs(b.p2, csb.p2, tmp * IAS15_C[5])
        b.p3, csb.p3 = add_cs(b.p3, csb.p3, tmp)

        ################################################################################
        n = 5
        step_time = t_beginning + dt * IAS15_H[n]
        # fmt: off
        x = x0 + ((((((((b.p6*H5AX + b.p5)*H5BX + b.p4)*H5CX + b.p3)*H5DX + b.p2)*H5EX + b.p1)*H5FX + b.p0)* + a0)*dt*H5FX + v0)*dt*IAS15_H5
        v = v0 + (((((((b.p6*H5AV + b.p5)*H5BV + b.p4)*H5CV + b.p3)*H5DV + b.p2)*H5EV + b.p1)*H5FV + b.p0)*H5FX + a0)*dt*IAS15_H5
        # fmt: on
        acc_state = SystemState(
            massive_positions=x[:M],
            massive_velocities=v[:M],
            tracer_positions=x[M:],
            tracer_velocities=v[M:],
            log_gms=initial_system_state.log_gms,
            time=step_time,
            acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
        )
        at = acceleration_func(acc_state)

        tmp = g.p4
        gk = at - a0
        g.p4 = (
            (((gk / IAS15_RR[10] - g.p0) / IAS15_RR[11] - g.p1) / IAS15_RR[12] - g.p2)
            / IAS15_RR[13]
            - g.p3
        ) / IAS15_RR[14]
        tmp = g.p4 - tmp
        b.p0, csb.p0 = add_cs(b.p0, csb.p0, tmp * IAS15_C[6])
        b.p1, csb.p1 = add_cs(b.p1, csb.p1, tmp * IAS15_C[7])
        b.p2, csb.p2 = add_cs(b.p2, csb.p2, tmp * IAS15_C[8])
        b.p3, csb.p3 = add_cs(b.p3, csb.p3, tmp * IAS15_C[9])
        b.p4, csb.p4 = add_cs(b.p4, csb.p4, tmp)

        ################################################################################
        n = 6
        step_time = t_beginning + dt * IAS15_H[n]
        # fmt: off
        x = x0 + ((((((((b.p6*H6AX + b.p5)*H6BX + b.p4)*H6CX + b.p3)*H6DX + b.p2)*H6EX + b.p1)*H6FX + b.p0)* + a0)*dt*H6FX + v0)*dt*IAS15_H6
        v = v0 + (((((((b.p6*H6AV + b.p5)*H6BV + b.p4)*H6CV + b.p3)*H6DV + b.p2)*H6EV + b.p1)*H6FV + b.p0)*H6FX + a0)*dt*IAS15_H6
        # fmt: on
        acc_state = SystemState(
            massive_positions=x[:M],
            massive_velocities=v[:M],
            tracer_positions=x[M:],
            tracer_velocities=v[M:],
            log_gms=initial_system_state.log_gms,
            time=step_time,
            acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
        )
        at = acceleration_func(acc_state)

        tmp = g.p5
        gk = at - a0
        g.p5 = (
            (
                (
                    ((gk / IAS15_RR[15] - g.p0) / IAS15_RR[16] - g.p1) / IAS15_RR[17]
                    - g.p2
                )
                / IAS15_RR[18]
                - g.p3
            )
            / IAS15_RR[19]
            - g.p4
        ) / IAS15_RR[20]
        tmp = g.p5 - tmp
        b.p0, csb.p0 = add_cs(b.p0, csb.p0, tmp * IAS15_C[10])
        b.p1, csb.p1 = add_cs(b.p1, csb.p1, tmp * IAS15_C[11])
        b.p2, csb.p2 = add_cs(b.p2, csb.p2, tmp * IAS15_C[12])
        b.p3, csb.p3 = add_cs(b.p3, csb.p3, tmp * IAS15_C[13])
        b.p4, csb.p4 = add_cs(b.p4, csb.p4, tmp * IAS15_C[14])
        b.p5, csb.p5 = add_cs(b.p5, csb.p5, tmp)

        ################################################################################
        n = 7
        step_time = t_beginning + dt * IAS15_H[n]
        # fmt: off
        x = x0 + ((((((((b.p6*H7AX + b.p5)*H7BX + b.p4)*H7CX + b.p3)*H7DX + b.p2)*H7EX + b.p1)*H7FX + b.p0)* + a0)*dt*H7FX + v0)*dt*IAS15_H7
        v = v0 + (((((((b.p6*H7AV + b.p5)*H7BV + b.p4)*H7CV + b.p3)*H7DV + b.p2)*H7EV + b.p1)*H7FV + b.p0)*H7FX + a0)*dt*IAS15_H7
        # fmt: on
        acc_state = SystemState(
            massive_positions=x[:M],
            massive_velocities=v[:M],
            tracer_positions=x[M:],
            tracer_velocities=v[M:],
            log_gms=initial_system_state.log_gms,
            time=step_time,
            acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
        )
        at = acceleration_func(acc_state)

        tmp = g.p6
        gk = at - a0
        g.p6 = (
            (
                (
                    (
                        ((gk / IAS15_RR[21] - g.p0) / IAS15_RR[22] - g.p1)
                        / IAS15_RR[23]
                        - g.p2
                    )
                    / IAS15_RR[24]
                    - g.p3
                )
                / IAS15_RR[25]
                - g.p4
            )
            / IAS15_RR[26]
            - g.p5
        ) / IAS15_RR[27]
        tmp = g.p6 - tmp
        b.p0, csb.p0 = add_cs(b.p0, csb.p0, tmp * IAS15_C[15])
        b.p1, csb.p1 = add_cs(b.p1, csb.p1, tmp * IAS15_C[16])
        b.p2, csb.p2 = add_cs(b.p2, csb.p2, tmp * IAS15_C[17])
        b.p3, csb.p3 = add_cs(b.p3, csb.p3, tmp * IAS15_C[18])
        b.p4, csb.p4 = add_cs(b.p4, csb.p4, tmp * IAS15_C[19])
        b.p5, csb.p5 = add_cs(b.p5, csb.p5, tmp * IAS15_C[20])
        b.p6, csb.p6 = add_cs(b.p6, csb.p6, tmp)

        maxa = jnp.max(jnp.abs(at))
        maxb6tmp = jnp.max(jnp.abs(tmp))

        predictor_corrector_error = jnp.abs(maxb6tmp / maxa)

        return b, g, predictor_corrector_error, predictor_corrector_error_last

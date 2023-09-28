# This is a pythonized/jaxified version of the IAS15 integrator from
# Rein & Spiegel (2015) (DOI: 10.1093/mnras/stu2164), currently implemented in REBOUND.
# The original code is available at https://github.com/hannorein/rebound/blob/main/src/integrator_ias15.c,
# originally accessed Summer 2023

# Many thanks to the REBOUND developers for their work on this integrator,
# and for making it open source!

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.data.constants import (
    IAS15_EPSILON,
    IAS15_MIN_DT,
    IAS15_SAFETY_FACTOR,
    IAS15_RR,
    IAS15_C,
    IAS15_D,
    IAS15_H,
)


def comp_sum(a, comp, b):
    y = b - comp
    t = a + y
    comp = (t - a) - y
    return t, comp


def sqrt7(a):
    def scan_fn(x, scan_over):
        x6 = x * x * x * x * x * x
        x += (a / x6 - x) / 7.0
        return x, None

    return jax.lax.scan(scan_fn, 1, None, length=20)[0]


def substep_acceleration(
    acc,
    acc_free_kwargs,
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    csx,
    csv,
    x0,
    v0,
    a0,
    t,
    dt,
    n,
):
    t_sub = t + IAS15_H[n] * dt
    x_sub = (
        -csx
        + (
            (
                (
                    (
                        (
                            (
                                (
                                    (b6 * 7.0 * IAS15_H[n] / 9.0 + b5)
                                    * 3.0
                                    * IAS15_H[n]
                                    / 4.0
                                    + b4
                                )
                                * 5.0
                                * IAS15_H[n]
                                / 7.0
                                + b3
                            )
                            * 2.0
                            * IAS15_H[n]
                            / 3.0
                            + b2
                        )
                        * 3.0
                        * IAS15_H[n]
                        / 5.0
                        + b1
                    )
                    * IAS15_H[n]
                    / 2.0
                    + b0
                )
                * IAS15_H[n]
                / 3.0
                + a0
            )
            * dt
            * IAS15_H[n]
            / 2.0
            + v0
        )
        * dt
        * IAS15_H[n]
    )
    v_sub = (
        -csv
        + (
            (
                (
                    (
                        (
                            (
                                (b6 * 7.0 * IAS15_H[n] / 8.0 + b5)
                                * 6.0
                                * IAS15_H[n]
                                / 7.0
                                + b4
                            )
                            * 5.0
                            * IAS15_H[n]
                            / 6.0
                            + b3
                        )
                        * 4.0
                        * IAS15_H[n]
                        / 5.0
                        + b2
                    )
                    * 3.0
                    * IAS15_H[n]
                    / 4.0
                    + b1
                )
                * 2.0
                * IAS15_H[n]
                / 3.0
                + b0
            )
            * IAS15_H[n]
            / 2.0
            + a0
        )
        * dt
        * IAS15_H[n]
    )
    x_sub = x_sub + x0
    v_sub = v_sub + v0
    a_sub = acc(x=x_sub, v=v_sub, time=t_sub, **acc_free_kwargs)
    return a_sub


def initialize_gs(b0, b1, b2, b3, b4, b5, b6):
    g0 = (
        b6 * IAS15_D[15]
        + b5 * IAS15_D[10]
        + b4 * IAS15_D[6]
        + b3 * IAS15_D[3]
        + b2 * IAS15_D[1]
        + b1 * IAS15_D[0]
        + b0
    )
    g1 = (
        b6 * IAS15_D[16]
        + b5 * IAS15_D[11]
        + b4 * IAS15_D[7]
        + b3 * IAS15_D[4]
        + b2 * IAS15_D[2]
        + b1
    )
    g2 = b6 * IAS15_D[17] + b5 * IAS15_D[12] + b4 * IAS15_D[8] + b3 * IAS15_D[5] + b2
    g3 = b6 * IAS15_D[18] + b5 * IAS15_D[13] + b4 * IAS15_D[9] + b3
    g4 = b6 * IAS15_D[19] + b5 * IAS15_D[14] + b4
    g5 = b6 * IAS15_D[20] + b5
    g6 = b6
    return g0, g1, g2, g3, g4, g5, g6


def predictor_corrector_iteration(
    acc,
    acc_free_kwargs,
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    csb0,
    csb1,
    csb2,
    csb3,
    csb4,
    csb5,
    csb6,
    g0,
    g1,
    g2,
    g3,
    g4,
    g5,
    g6,
    csx,
    csv,
    x0,
    v0,
    a0,
    t,
    dt,
    a_sub,
    predictor_corrector_error,
):
    #######################################################
    n = 1
    a_sub = substep_acceleration(
        acc,
        acc_free_kwargs,
        b0,
        b1,
        b2,
        b3,
        b4,
        b5,
        b6,
        csx,
        csv,
        x0,
        v0,
        a0,
        t,
        dt,
        n,
    )
    tmp = g0
    gk = a_sub
    gk_cs = 0.0
    gk, gk_cs = comp_sum(gk, gk_cs, -a0)
    gk, gk_cs = comp_sum(gk, gk_cs, 0)  # csa0
    g0 = gk / IAS15_RR[0]
    b0, csb0 = comp_sum(b0, csb0, g0 - tmp)

    n = 2
    a_sub = substep_acceleration(
        acc,
        acc_free_kwargs,
        b0,
        b1,
        b2,
        b3,
        b4,
        b5,
        b6,
        csx,
        csv,
        x0,
        v0,
        a0,
        t,
        dt,
        n,
    )
    tmp = g1
    gk = a_sub
    gk_cs = 0.0
    gk, gk_cs = comp_sum(gk, gk_cs, -a0)
    gk, gk_cs = comp_sum(gk, gk_cs, 0)  # csa0
    g1 = (gk / IAS15_RR[1] - g0) / IAS15_RR[2]
    tmp = g1 - tmp
    b0, csb0 = comp_sum(b0, csb0, tmp * IAS15_C[0])
    b1, csb1 = comp_sum(b1, csb1, tmp)

    n = 3
    a_sub = substep_acceleration(
        acc,
        acc_free_kwargs,
        b0,
        b1,
        b2,
        b3,
        b4,
        b5,
        b6,
        csx,
        csv,
        x0,
        v0,
        a0,
        t,
        dt,
        n,
    )
    tmp = g2
    gk = a_sub
    gk_cs = 0.0
    gk, gk_cs = comp_sum(gk, gk_cs, -a0)
    gk, gk_cs = comp_sum(gk, gk_cs, 0)  # csa0
    g2 = ((gk / IAS15_RR[3] - g0) / IAS15_RR[4] - g1) / IAS15_RR[5]
    tmp = g2 - tmp
    b0, csb0 = comp_sum(b0, csb0, tmp * IAS15_C[1])
    b1, csb1 = comp_sum(b1, csb1, tmp * IAS15_C[2])
    b2, csb2 = comp_sum(b2, csb2, tmp)

    n = 4
    a_sub = substep_acceleration(
        acc,
        acc_free_kwargs,
        b0,
        b1,
        b2,
        b3,
        b4,
        b5,
        b6,
        csx,
        csv,
        x0,
        v0,
        a0,
        t,
        dt,
        n,
    )
    tmp = g3
    gk = a_sub
    gk_cs = 0.0
    gk, gk_cs = comp_sum(gk, gk_cs, -a0)
    gk, gk_cs = comp_sum(gk, gk_cs, 0)  # csa0
    g3 = (((gk / IAS15_RR[6] - g0) / IAS15_RR[7] - g1) / IAS15_RR[8] - g2) / IAS15_RR[9]
    tmp = g3 - tmp
    b0, csb0 = comp_sum(b0, csb0, tmp * IAS15_C[3])
    b1, csb1 = comp_sum(b1, csb1, tmp * IAS15_C[4])
    b2, csb2 = comp_sum(b2, csb2, tmp * IAS15_C[5])
    b3, csb3 = comp_sum(b3, csb3, tmp)

    n = 5
    a_sub = substep_acceleration(
        acc,
        acc_free_kwargs,
        b0,
        b1,
        b2,
        b3,
        b4,
        b5,
        b6,
        csx,
        csv,
        x0,
        v0,
        a0,
        t,
        dt,
        n,
    )
    tmp = g4
    gk = a_sub
    gk_cs = 0.0
    gk, gk_cs = comp_sum(gk, gk_cs, -a0)
    gk, gk_cs = comp_sum(gk, gk_cs, 0)  # csa0
    g4 = (
        (((gk / IAS15_RR[10] - g0) / IAS15_RR[11] - g1) / IAS15_RR[12] - g2)
        / IAS15_RR[13]
        - g3
    ) / IAS15_RR[14]
    tmp = g4 - tmp
    b0, csb0 = comp_sum(b0, csb0, tmp * IAS15_C[6])
    b1, csb1 = comp_sum(b1, csb1, tmp * IAS15_C[7])
    b2, csb2 = comp_sum(b2, csb2, tmp * IAS15_C[8])
    b3, csb3 = comp_sum(b3, csb3, tmp * IAS15_C[9])
    b4, csb4 = comp_sum(b4, csb4, tmp)

    n = 6
    a_sub = substep_acceleration(
        acc,
        acc_free_kwargs,
        b0,
        b1,
        b2,
        b3,
        b4,
        b5,
        b6,
        csx,
        csv,
        x0,
        v0,
        a0,
        t,
        dt,
        n,
    )
    tmp = g5
    gk = a_sub
    gk_cs = 0.0
    gk, gk_cs = comp_sum(gk, gk_cs, -a0)
    gk, gk_cs = comp_sum(gk, gk_cs, 0)  # csa0
    g5 = (
        (
            (((gk / IAS15_RR[15] - g0) / IAS15_RR[16] - g1) / IAS15_RR[17] - g2)
            / IAS15_RR[18]
            - g3
        )
        / IAS15_RR[19]
        - g4
    ) / IAS15_RR[20]
    tmp = g5 - tmp
    b0, csb0 = comp_sum(b0, csb0, tmp * IAS15_C[10])
    b1, csb1 = comp_sum(b1, csb1, tmp * IAS15_C[11])
    b2, csb2 = comp_sum(b2, csb2, tmp * IAS15_C[12])
    b3, csb3 = comp_sum(b3, csb3, tmp * IAS15_C[13])
    b4, csb4 = comp_sum(b4, csb4, tmp * IAS15_C[14])
    b5, csb5 = comp_sum(b5, csb5, tmp)

    n = 7
    a_sub = substep_acceleration(
        acc,
        acc_free_kwargs,
        b0,
        b1,
        b2,
        b3,
        b4,
        b5,
        b6,
        csx,
        csv,
        x0,
        v0,
        a0,
        t,
        dt,
        n,
    )
    tmp = g6
    gk = a_sub
    gk_cs = 0.0
    gk, gk_cs = comp_sum(gk, gk_cs, -a0)
    gk, gk_cs = comp_sum(gk, gk_cs, 0)  # csa0
    g6 = (
        (
            (
                (((gk / IAS15_RR[21] - g0) / IAS15_RR[22] - g1) / IAS15_RR[23] - g2)
                / IAS15_RR[24]
                - g3
            )
            / IAS15_RR[25]
            - g4
        )
        / IAS15_RR[26]
        - g5
    ) / IAS15_RR[27]
    tmp = g6 - tmp
    b0, csb0 = comp_sum(b0, csb0, tmp * IAS15_C[15])
    b1, csb1 = comp_sum(b1, csb1, tmp * IAS15_C[16])
    b2, csb2 = comp_sum(b2, csb2, tmp * IAS15_C[17])
    b3, csb3 = comp_sum(b3, csb3, tmp * IAS15_C[18])
    b4, csb4 = comp_sum(b4, csb4, tmp * IAS15_C[19])
    b5, csb5 = comp_sum(b5, csb5, tmp * IAS15_C[20])
    b6, csb6 = comp_sum(b6, csb6, tmp)

    # global error:
    predictor_corrector_error = jnp.max(jnp.abs(tmp)) / jnp.max(jnp.abs(a_sub))

    # # local error:
    # predictor_corrector_error = jnp.nanmax(jnp.abs(tmp / a_sub))

    return (
        b0,
        b1,
        b2,
        b3,
        b4,
        b5,
        b6,
        csb0,
        csb1,
        csb2,
        csb3,
        csb4,
        csb5,
        csb6,
        g0,
        g1,
        g2,
        g3,
        g4,
        g5,
        g6,
        csx,
        csv,
        x0,
        v0,
        a0,
        t,
        dt,
        a_sub,
        predictor_corrector_error,
    )


def predictor_corrector(
    acc,
    acc_free_kwargs,
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    csx,
    csv,
    x0,
    v0,
    a0,
    t,
    dt,
):
    predictor_corrector_error = 1e300
    g0, g1, g2, g3, g4, g5, g6 = initialize_gs(b0, b1, b2, b3, b4, b5, b6)
    csb0 = jnp.zeros_like(b0)
    csb1 = jnp.zeros_like(b1)
    csb2 = jnp.zeros_like(b2)
    csb3 = jnp.zeros_like(b3)
    csb4 = jnp.zeros_like(b4)
    csb5 = jnp.zeros_like(b5)
    csb6 = jnp.zeros_like(b6)

    iteration = jax.tree_util.Partial(
        predictor_corrector_iteration, acc, acc_free_kwargs
    )

    def iterate(carry, scan_over):
        def iteration_needed(carry):
            return iteration(*carry)

        def iteration_not_needed(carry):
            return carry

        carry = jax.lax.cond(
            carry[-1] < 1e-16, iteration_not_needed, iteration_needed, carry
        )

        return carry, None

    return jax.lax.scan(
        iterate,
        (
            b0,
            b1,
            b2,
            b3,
            b4,
            b5,
            b6,
            csb0,
            csb1,
            csb2,
            csb3,
            csb4,
            csb5,
            csb6,
            g0,
            g1,
            g2,
            g3,
            g4,
            g5,
            g6,
            csx,
            csv,
            x0,
            v0,
            a0,
            t,
            dt,
            a0,
            predictor_corrector_error,
        ),
        None,
        length=10,
    )[0]


def predict_next_step(
    ratio,
    _e0,
    _e1,
    _e2,
    _e3,
    _e4,
    _e5,
    _e6,
    _b0,
    _b1,
    _b2,
    _b3,
    _b4,
    _b5,
    _b6,
    e0,
    e1,
    e2,
    e3,
    e4,
    e5,
    e6,
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
):
    def large_ratio(params):
        (
            ratio,
            _e0,
            _e1,
            _e2,
            _e3,
            _e4,
            _e5,
            _e6,
            _b0,
            _b1,
            _b2,
            _b3,
            _b4,
            _b5,
            _b6,
            e0,
            e1,
            e2,
            e3,
            e4,
            e5,
            e6,
            b0,
            b1,
            b2,
            b3,
            b4,
            b5,
            b6,
        ) = params
        e0 = jnp.zeros_like(e0)
        e1 = jnp.zeros_like(e1)
        e2 = jnp.zeros_like(e2)
        e3 = jnp.zeros_like(e3)
        e4 = jnp.zeros_like(e4)
        e5 = jnp.zeros_like(e5)
        e6 = jnp.zeros_like(e6)
        b0 = jnp.zeros_like(b0)
        b1 = jnp.zeros_like(b1)
        b2 = jnp.zeros_like(b2)
        b3 = jnp.zeros_like(b3)
        b4 = jnp.zeros_like(b4)
        b5 = jnp.zeros_like(b5)
        b6 = jnp.zeros_like(b6)
        return e0, e1, e2, e3, e4, e5, e6, b0, b1, b2, b3, b4, b5, b6

    def ok_ratio(params):
        (
            ratio,
            _e0,
            _e1,
            _e2,
            _e3,
            _e4,
            _e5,
            _e6,
            _b0,
            _b1,
            _b2,
            _b3,
            _b4,
            _b5,
            _b6,
            e0,
            e1,
            e2,
            e3,
            e4,
            e5,
            e6,
            b0,
            b1,
            b2,
            b3,
            b4,
            b5,
            b6,
        ) = params
        q1 = ratio
        q2 = q1 * q1
        q3 = q1 * q2
        q4 = q2 * q2
        q5 = q2 * q3
        q6 = q3 * q3
        q7 = q3 * q4

        be0 = _b0 - _e0
        be1 = _b1 - _e1
        be2 = _b2 - _e2
        be3 = _b3 - _e3
        be4 = _b4 - _e4
        be5 = _b5 - _e5
        be6 = _b6 - _e6

        e0 = q1 * (
            _b6 * 7.0 + _b5 * 6.0 + _b4 * 5.0 + _b3 * 4.0 + _b2 * 3.0 + _b1 * 2.0 + _b0
        )
        e1 = q2 * (_b6 * 21.0 + _b5 * 15.0 + _b4 * 10.0 + _b3 * 6.0 + _b2 * 3.0 + _b1)
        e2 = q3 * (_b6 * 35.0 + _b5 * 20.0 + _b4 * 10.0 + _b3 * 4.0 + _b2)
        e3 = q4 * (_b6 * 35.0 + _b5 * 15.0 + _b4 * 5.0 + _b3)
        e4 = q5 * (_b6 * 21.0 + _b5 * 6.0 + _b4)
        e5 = q6 * (_b6 * 7.0 + _b5)
        e6 = q7 * _b6

        b0 = e0 + be0
        b1 = e1 + be1
        b2 = e2 + be2
        b3 = e3 + be3
        b4 = e4 + be4
        b5 = e5 + be5
        b6 = e6 + be6

        return e0, e1, e2, e3, e4, e5, e6, b0, b1, b2, b3, b4, b5, b6

    return jax.lax.cond(
        ratio < 20.0,
        ok_ratio,
        large_ratio,
        (
            ratio,
            _e0,
            _e1,
            _e2,
            _e3,
            _e4,
            _e5,
            _e6,
            _b0,
            _b1,
            _b2,
            _b3,
            _b4,
            _b5,
            _b6,
            e0,
            e1,
            e2,
            e3,
            e4,
            e5,
            e6,
            b0,
            b1,
            b2,
            b3,
            b4,
            b5,
            b6,
        ),
    )


def ias15_step(
    acc,
    acc_free_kwargs,
    x0,
    v0,
    a0,
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    e0,
    e1,
    e2,
    e3,
    e4,
    e5,
    e6,
    csx,
    csv,
    t,
    dt,
    tf,
):
    dt = jnp.copysign(dt, tf - t)
    (
        b0,
        b1,
        b2,
        b3,
        b4,
        b5,
        b6,
        csb0,
        csb1,
        csb2,
        csb3,
        csb4,
        csb5,
        csb6,
        g0,
        g1,
        g2,
        g3,
        g4,
        g5,
        g6,
        csx,
        csv,
        x0,
        v0,
        a0,
        t,
        dt,
        a_sub,
        predictor_corrector_error,
    ) = predictor_corrector(
        acc,
        acc_free_kwargs,
        b0,
        b1,
        b2,
        b3,
        b4,
        b5,
        b6,
        csx,
        csv,
        x0,
        v0,
        a0,
        t,
        dt,
    )

    # global error
    maxak = jnp.max(jnp.abs(a_sub))
    maxb6k = jnp.max(jnp.abs(b6))

    integrator_error = maxb6k / maxak

    dt_new = sqrt7(IAS15_EPSILON / integrator_error) * dt
    dt_new = jnp.where(
        jnp.abs(dt_new) < IAS15_MIN_DT, jnp.copysign(IAS15_MIN_DT, dt_new), dt_new
    )
    dt_new = jnp.where(
        jnp.abs(dt_new / dt) > (1.0 / IAS15_SAFETY_FACTOR),
        dt / IAS15_SAFETY_FACTOR,
        dt_new,
    )

    def successful_step(params):
        x0, csx, v0, csv, b0, b1, b2, b3, b4, b5, b6, a0, dt, t = params
        x0, csx = comp_sum(x0, csx, b6 / 72.0 * dt * dt)
        x0, csx = comp_sum(x0, csx, b5 / 56.0 * dt * dt)
        x0, csx = comp_sum(x0, csx, b4 / 42.0 * dt * dt)
        x0, csx = comp_sum(x0, csx, b3 / 30.0 * dt * dt)
        x0, csx = comp_sum(x0, csx, b2 / 20.0 * dt * dt)
        x0, csx = comp_sum(x0, csx, b1 / 12.0 * dt * dt)
        x0, csx = comp_sum(x0, csx, b0 / 6.0 * dt * dt)
        x0, csx = comp_sum(x0, csx, a0 / 2.0 * dt * dt)
        x0, csx = comp_sum(x0, csx, v0 * dt)
        v0, csv = comp_sum(v0, csv, b6 / 8.0 * dt)
        v0, csv = comp_sum(v0, csv, b5 / 7.0 * dt)
        v0, csv = comp_sum(v0, csv, b4 / 6.0 * dt)
        v0, csv = comp_sum(v0, csv, b3 / 5.0 * dt)
        v0, csv = comp_sum(v0, csv, b2 / 4.0 * dt)
        v0, csv = comp_sum(v0, csv, b1 / 3.0 * dt)
        v0, csv = comp_sum(v0, csv, b0 / 2.0 * dt)
        v0, csv = comp_sum(v0, csv, a0 * dt)
        return x0, csx, v0, csv, t + dt

    def failed_step(params):
        jax.debug.print("FAILED STEP")
        x0, csx, v0, csv, b0, b1, b2, b3, b4, b5, b6, a0, dt, t = params
        return x0, csx, v0, csv, t

    x0, csx, v0, csv, t = jax.lax.cond(
        dt_new / dt > IAS15_SAFETY_FACTOR,
        successful_step,
        failed_step,
        (x0, csx, v0, csv, b0, b1, b2, b3, b4, b5, b6, a0, dt, t),
    )

    remaining_time = tf - (t)
    tmp = dt_new
    dt_new = jnp.where(
        jnp.abs(remaining_time) < jnp.abs(dt_new), remaining_time, dt_new
    )
    dt_new = jnp.where(dt_new == 0, tmp, dt_new)  # don't let the next dt go to zero

    a0 = acc(x=x0, v=v0, time=t, **acc_free_kwargs)
    ratio = dt_new / dt
    e0, e1, e2, e3, e4, e5, e6, b0, b1, b2, b3, b4, b5, b6 = predict_next_step(
        ratio,
        e0,
        e1,
        e2,
        e3,
        e4,
        e5,
        e6,
        b0,
        b1,
        b2,
        b3,
        b4,
        b5,
        b6,
        e0,
        e1,
        e2,
        e3,
        e4,
        e5,
        e6,
        b0,
        b1,
        b2,
        b3,
        b4,
        b5,
        b6,
    )
    return (
        x0,
        v0,
        a0,
        b0,
        b1,
        b2,
        b3,
        b4,
        b5,
        b6,
        e0,
        e1,
        e2,
        e3,
        e4,
        e5,
        e6,
        csx,
        csv,
        t,
        dt_new,
        tf,
    )


def ias15_integrate(
    acc,
    acc_free_kwargs,
    x0,
    v0,
    a0,
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    e0,
    e1,
    e2,
    e3,
    e4,
    e5,
    e6,
    csx,
    csv,
    t0,
    dt,
    tf,
):
    step = jax.tree_util.Partial(ias15_step, acc, acc_free_kwargs)

    def iterate(carry, scan_over):
        remaining_time = carry[-1] - carry[-3]

        def step_needed(carry):
            return step(*carry)

        def reached_end(carry):
            return carry

        carry = jax.lax.cond(remaining_time == 0, reached_end, step_needed, carry)
        return carry, None

    return jax.lax.scan(
        iterate,
        (
            x0,
            v0,
            a0,
            b0,
            b1,
            b2,
            b3,
            b4,
            b5,
            b6,
            e0,
            e1,
            e2,
            e3,
            e4,
            e5,
            e6,
            csx,
            csv,
            t0,
            dt,
            tf,
        ),
        None,
        length=10000,
    )[0]


def ias15_integrate_multiple(
    acc,
    acc_free_kwargs,
    x0,
    v0,
    a0,
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    e0,
    e1,
    e2,
    e3,
    e4,
    e5,
    e6,
    csx,
    csv,
    t0,
    dt,
    tfs,
):
    integrate = jax.tree_util.Partial(ias15_integrate, acc, acc_free_kwargs)

    def true_fn(x):
        carry, scan_over = x
        return integrate(*carry, scan_over)[:-1]  # everything but tf

    def scan_fn(carry, scan_over):
        q = jax.lax.cond(
            scan_over != 999.0, true_fn, lambda x: x[0], (carry, scan_over)
        )
        return q, (q[:2] + (q[-2],))  # x0, v0, t0

    s = jax.lax.scan(
        scan_fn,
        (
            x0,
            v0,
            a0,
            b0,
            b1,
            b2,
            b3,
            b4,
            b5,
            b6,
            e0,
            e1,
            e2,
            e3,
            e4,
            e5,
            e6,
            csx,
            csv,
            t0,
            dt,
        ),
        tfs,
    )
    return s[1]


def ias15_initial_params(n):
    return {
        "dt": 0.0001,
        "b0": jnp.zeros((n, 3)),
        "b1": jnp.zeros((n, 3)),
        "b2": jnp.zeros((n, 3)),
        "b3": jnp.zeros((n, 3)),
        "b4": jnp.zeros((n, 3)),
        "b5": jnp.zeros((n, 3)),
        "b6": jnp.zeros((n, 3)),
        "e0": jnp.zeros((n, 3)),
        "e1": jnp.zeros((n, 3)),
        "e2": jnp.zeros((n, 3)),
        "e3": jnp.zeros((n, 3)),
        "e4": jnp.zeros((n, 3)),
        "e5": jnp.zeros((n, 3)),
        "e6": jnp.zeros((n, 3)),
        "csx": jnp.zeros((n, 3)),
        "csv": jnp.zeros((n, 3)),
    }


# Q = jnp.array([[0.05626256053692214646565,0.0015827378590854146249952,0.00002968296153695723135209,8.350197101940939341306e-7,2.818820819659103805192e-8,1.057293846725378882528e-9,4.249004218335851037836e-11,1.792948927918187883134e-12,7.84590314640274393589e-14],
# [0.18024069173689236498758,0.016243353478896729829491,0.0009759044223877345541671,0.00008794884408012883189139,9.511176296677507044754e-6,1.1428673299631253793176e-6,1.4713657008289209466529e-7,1.988999787865066014409e-8,2.78832320378369077318e-9],
# [0.35262471711316963737391,0.062172095559571455858371,0.0073078058696756115908628,0.00128845648875616157965,0.00027260496291616158758097,0.00006408483195463838623104,0.000016141354099463839268237,4.2688803173602030228917e-6,1.1707987777882039898473e-6],
# [0.54715362633055538300145,0.149688545403338514457695,0.027300876812527551715904,0.0074688868749891119686498,0.002451977122981789251434,0.00089440544967936543859197,0.00034955513228705419505859,0.00014344526867498902973857,0.00006104513250537416466755],
# [0.73421017721541053152321,0.269532292163342269000521,0.065964450664841111517343,0.024215885506275098771688,0.0106677297533941989888391,0.0052215705018171083511289,0.0027383787167727851580664,0.0015079091421934913059341,0.0008610950744002607231672],
# [0.88532094683909576809036,0.391896589456036517542395,0.115651419880076881620928,0.051194312275757744192192,0.027194038210050113841565,0.016050301104333410581975,0.0101497626933864567779021,0.0067393481384257715860068,0.0046406002805473123197343],
# [0.97752061356128750189117,0.477773274968617987575421,0.155677741630169726318351,0.076089100758079550432207,0.044627198675018728876142,0.029082671086883856332127,0.020306364632037016645903,0.014887417510731039470401,0.011318811388447780906456]
# ])
# def substep_acceleration(b0,b1,b2,b3,b4,b5,b6, csx, csv, x0, v0, a0, t, dt, n):
#     t_sub = t + IAS15_H[n] * dt
#     dt2 = dt**2
#     consts = dt2*Q[n-1]

#     # x_sub, cx = comp_sum(0, csx, dt*v0*Q[n-1,0])
#     # x_sub, cx = comp_sum(x_sub, cx, a0*consts[1])
#     # x_sub, cx = comp_sum(x_sub, cx, b0*consts[2])
#     # x_sub, cx = comp_sum(x_sub, cx, b1*consts[3])
#     # x_sub, cx = comp_sum(x_sub, cx, b2*consts[4])
#     # x_sub, cx = comp_sum(x_sub, cx, b3*consts[5])
#     # x_sub, cx = comp_sum(x_sub, cx, b4*consts[6])
#     # x_sub, cx = comp_sum(x_sub, cx, b5*consts[7])
#     # x_sub, cx = comp_sum(x_sub, cx, b6*consts[8])
#     # x_sub = -csx + dt*v0*Q[n-1,0] + a0*consts[1] + b0*consts[2] + b1*consts[3] + b2*consts[4] + b3*consts[5] + b4*consts[6] + b5*consts[7] + b6*consts[8]

#     # x_sub = -csx + ((((((((b6 * 7. * IAS15_H[n] / 9. + b5) * 3. * IAS15_H[n] / 4. + b4) * 5. * IAS15_H[n] / 7. + b3) * 2. * IAS15_H[n] / 3. + b2) * 3. * IAS15_H[n] / 5. + b1) * IAS15_H[n] / 2. + b0) * IAS15_H[n] / 3. + a0) * dt * IAS15_H[n] / 2. + v0) * dt * IAS15_H[n]
#     v_sub = -csv + (((((((b6*7.*IAS15_H[n]/8. + b5)*6.*IAS15_H[n]/7. + b4)*5.*IAS15_H[n]/6. + b3)*4.*IAS15_H[n]/5. + b2)*3.*IAS15_H[n]/4. + b1)*2.*IAS15_H[n]/3. + b0)*IAS15_H[n]/2. + a0) * dt * IAS15_H[n]
#     x_sub = x_sub + x0
#     v_sub = v_sub + v0
#     a_sub = acc(x_sub, v_sub, t_sub)
#     return a_sub

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from typing import Callable

from jorbit.utils.states import SystemState
from jorbit.integrators import IAS15Helper, IAS15IntegratorState
from jorbit.data.constants import (
    IAS15_H,
    IAS15_RR,
    IAS15_C,
    IAS15_D,
    IAS15_EPSILON,
    IAS15_MIN_DT,
    IAS15_SAFETY_FACTOR,
)


@jax.jit
def add_cs(p, csp, inp):
    y = inp - csp
    t = p + y
    csp = (t - p) - y
    p = t
    return p, csp


# @jax.jit
def ias15_step(
    initial_system_state: SystemState,
    acceleration_func: Callable[[SystemState], jnp.ndarray],
    initial_integrator_state: IAS15IntegratorState,
) -> SystemState:

    # for convenience, rename initial state
    t_beginning = initial_system_state.time
    x0 = initial_system_state.positions
    v0 = initial_system_state.velocities
    a0 = acceleration_func(initial_system_state)
    initial_integrator_state.a0 = a0  # ias15 specific, so not included in SystemState

    dt = initial_integrator_state.dt
    csx = initial_integrator_state.csx
    csv = initial_integrator_state.csv
    g = initial_integrator_state.g
    e = initial_integrator_state.e
    b = initial_integrator_state.b
    er = initial_integrator_state.er
    br = initial_integrator_state.br

    # always zero the compensation terms
    csb = IAS15Helper(
        p0=jnp.zeros_like(x0, dtype=jnp.float64),
        p1=jnp.zeros_like(x0, dtype=jnp.float64),
        p2=jnp.zeros_like(x0, dtype=jnp.float64),
        p3=jnp.zeros_like(x0, dtype=jnp.float64),
        p4=jnp.zeros_like(x0, dtype=jnp.float64),
        p5=jnp.zeros_like(x0, dtype=jnp.float64),
        p6=jnp.zeros_like(x0, dtype=jnp.float64),
    )

    # get the initial g terms from the b terms
    g.p0 = (
        b.p6 * IAS15_D[15]
        + b.p5 * IAS15_D[10]
        + b.p4 * IAS15_D[6]
        + b.p3 * IAS15_D[3]
        + b.p2 * IAS15_D[1]
        + b.p1 * IAS15_D[0]
        + b.p0
    )
    g.p1 = (
        b.p6 * IAS15_D[16]
        + b.p5 * IAS15_D[11]
        + b.p4 * IAS15_D[7]
        + b.p3 * IAS15_D[4]
        + b.p2 * IAS15_D[2]
        + b.p1
    )
    g.p2 = (
        b.p6 * IAS15_D[17]
        + b.p5 * IAS15_D[12]
        + b.p4 * IAS15_D[8]
        + b.p3 * IAS15_D[5]
        + b.p2
    )
    g.p3 = b.p6 * IAS15_D[18] + b.p5 * IAS15_D[13] + b.p4 * IAS15_D[9] + b.p3
    g.p4 = b.p6 * IAS15_D[19] + b.p5 * IAS15_D[14] + b.p4
    g.p5 = b.p6 * IAS15_D[20] + b.p5
    g.p6 = b.p6

    # set up the predictor-corrector loop
    predictor_corrector_error = 1e300
    predictor_corrector_error_last = 2.0

    # predictor-corrector iterations
    for iteration in range(5):
        # if predictor_corrector_error < 1e-16:
        #     break
        # elif (iteration > 2) and (
        #     predictor_corrector_error > predictor_corrector_error_last
        # ):
        #     break
        predictor_corrector_error_last = predictor_corrector_error
        predictor_corrector_error = 0.0

        # loop over each subinterval
        print(f"n=0, x={x0[1]}")

        # for n in jnp.arange(1, 8):
        ################################################################################
        n = 1
        step_time = t_beginning + dt * IAS15_H[n]
        # get the new acceleration value at predicted position
        # fmt: off
        x = x0 - csx + ((((((((b.p6*7.*IAS15_H[n]/9. + b.p5)*3.*IAS15_H[n]/4. + b.p4)*5.*IAS15_H[n]/7. + b.p3)*2.*IAS15_H[n]/3. + b.p2)*3.*IAS15_H[n]/5. + b.p1)*IAS15_H[n]/2. + b.p0)*IAS15_H[n]/3. + a0)*dt*IAS15_H[n]/2. + v0)*dt*IAS15_H[n]
        v = v0 - csv + (((((((b.p6*7.*IAS15_H[n]/8. + b.p5)*6.*IAS15_H[n]/7. + b.p4)*5.*IAS15_H[n]/6. + b.p3)*4.*IAS15_H[n]/5. + b.p2)*3.*IAS15_H[n]/4. + b.p1)*2.*IAS15_H[n]/3. + b.p0)*IAS15_H[n]/2. + a0)*dt*IAS15_H[n]
        # fmt: on
        print(f"n={n}, x={x[1]}")
        acc_state = SystemState(
            positions=x,
            velocities=v,
            gms=initial_system_state.gms,
            time=step_time,
            acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
        )
        at = acceleration_func(acc_state)

        tmp = g.p0
        gk = at - a0
        g.p0 = gk / IAS15_RR[0]
        b.p0, csb.p0 = add_cs(b.p0, csb.p0, g.p0 - tmp)

        ################################################################################
        n = 2
        step_time = t_beginning + dt * IAS15_H[n]
        # fmt: off
        x = x0 - csx + ((((((((b.p6*7.*IAS15_H[n]/9. + b.p5)*3.*IAS15_H[n]/4. + b.p4)*5.*IAS15_H[n]/7. + b.p3)*2.*IAS15_H[n]/3. + b.p2)*3.*IAS15_H[n]/5. + b.p1)*IAS15_H[n]/2. + b.p0)*IAS15_H[n]/3. + a0)*dt*IAS15_H[n]/2. + v0)*dt*IAS15_H[n]
        v = v0 - csv + (((((((b.p6*7.*IAS15_H[n]/8. + b.p5)*6.*IAS15_H[n]/7. + b.p4)*5.*IAS15_H[n]/6. + b.p3)*4.*IAS15_H[n]/5. + b.p2)*3.*IAS15_H[n]/4. + b.p1)*2.*IAS15_H[n]/3. + b.p0)*IAS15_H[n]/2. + a0)*dt*IAS15_H[n]
        # fmt: on
        print(f"n={n}, x={x[1]}")
        acc_state = SystemState(
            positions=x,
            velocities=v,
            gms=initial_system_state.gms,
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
        x = x0 - csx + ((((((((b.p6*7.*IAS15_H[n]/9. + b.p5)*3.*IAS15_H[n]/4. + b.p4)*5.*IAS15_H[n]/7. + b.p3)*2.*IAS15_H[n]/3. + b.p2)*3.*IAS15_H[n]/5. + b.p1)*IAS15_H[n]/2. + b.p0)*IAS15_H[n]/3. + a0)*dt*IAS15_H[n]/2. + v0)*dt*IAS15_H[n]
        v = v0 - csv + (((((((b.p6*7.*IAS15_H[n]/8. + b.p5)*6.*IAS15_H[n]/7. + b.p4)*5.*IAS15_H[n]/6. + b.p3)*4.*IAS15_H[n]/5. + b.p2)*3.*IAS15_H[n]/4. + b.p1)*2.*IAS15_H[n]/3. + b.p0)*IAS15_H[n]/2. + a0)*dt*IAS15_H[n]
        # fmt: on
        print(f"n={n}, x={x[1]}")
        acc_state = SystemState(
            positions=x,
            velocities=v,
            gms=initial_system_state.gms,
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
        x = x0 - csx + ((((((((b.p6*7.*IAS15_H[n]/9. + b.p5)*3.*IAS15_H[n]/4. + b.p4)*5.*IAS15_H[n]/7. + b.p3)*2.*IAS15_H[n]/3. + b.p2)*3.*IAS15_H[n]/5. + b.p1)*IAS15_H[n]/2. + b.p0)*IAS15_H[n]/3. + a0)*dt*IAS15_H[n]/2. + v0)*dt*IAS15_H[n]
        v = v0 - csv + (((((((b.p6*7.*IAS15_H[n]/8. + b.p5)*6.*IAS15_H[n]/7. + b.p4)*5.*IAS15_H[n]/6. + b.p3)*4.*IAS15_H[n]/5. + b.p2)*3.*IAS15_H[n]/4. + b.p1)*2.*IAS15_H[n]/3. + b.p0)*IAS15_H[n]/2. + a0)*dt*IAS15_H[n]
        # fmt: on
        print(f"n={n}, x={x[1]}")
        acc_state = SystemState(
            positions=x,
            velocities=v,
            gms=initial_system_state.gms,
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
        x = x0 - csx + ((((((((b.p6*7.*IAS15_H[n]/9. + b.p5)*3.*IAS15_H[n]/4. + b.p4)*5.*IAS15_H[n]/7. + b.p3)*2.*IAS15_H[n]/3. + b.p2)*3.*IAS15_H[n]/5. + b.p1)*IAS15_H[n]/2. + b.p0)*IAS15_H[n]/3. + a0)*dt*IAS15_H[n]/2. + v0)*dt*IAS15_H[n]
        v = v0 - csv + (((((((b.p6*7.*IAS15_H[n]/8. + b.p5)*6.*IAS15_H[n]/7. + b.p4)*5.*IAS15_H[n]/6. + b.p3)*4.*IAS15_H[n]/5. + b.p2)*3.*IAS15_H[n]/4. + b.p1)*2.*IAS15_H[n]/3. + b.p0)*IAS15_H[n]/2. + a0)*dt*IAS15_H[n]
        # fmt: on
        print(f"n={n}, x={x[1]}")
        acc_state = SystemState(
            positions=x,
            velocities=v,
            gms=initial_system_state.gms,
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
        x = x0 - csx + ((((((((b.p6*7.*IAS15_H[n]/9. + b.p5)*3.*IAS15_H[n]/4. + b.p4)*5.*IAS15_H[n]/7. + b.p3)*2.*IAS15_H[n]/3. + b.p2)*3.*IAS15_H[n]/5. + b.p1)*IAS15_H[n]/2. + b.p0)*IAS15_H[n]/3. + a0)*dt*IAS15_H[n]/2. + v0)*dt*IAS15_H[n]
        v = v0 - csv + (((((((b.p6*7.*IAS15_H[n]/8. + b.p5)*6.*IAS15_H[n]/7. + b.p4)*5.*IAS15_H[n]/6. + b.p3)*4.*IAS15_H[n]/5. + b.p2)*3.*IAS15_H[n]/4. + b.p1)*2.*IAS15_H[n]/3. + b.p0)*IAS15_H[n]/2. + a0)*dt*IAS15_H[n]
        # fmt: on
        print(f"n={n}, x={x[1]}")
        acc_state = SystemState(
            positions=x,
            velocities=v,
            gms=initial_system_state.gms,
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
        x = x0 - csx + ((((((((b.p6*7.*IAS15_H[n]/9. + b.p5)*3.*IAS15_H[n]/4. + b.p4)*5.*IAS15_H[n]/7. + b.p3)*2.*IAS15_H[n]/3. + b.p2)*3.*IAS15_H[n]/5. + b.p1)*IAS15_H[n]/2. + b.p0)*IAS15_H[n]/3. + a0)*dt*IAS15_H[n]/2. + v0)*dt*IAS15_H[n]
        v = v0 - csv + (((((((b.p6*7.*IAS15_H[n]/8. + b.p5)*6.*IAS15_H[n]/7. + b.p4)*5.*IAS15_H[n]/6. + b.p3)*4.*IAS15_H[n]/5. + b.p2)*3.*IAS15_H[n]/4. + b.p1)*2.*IAS15_H[n]/3. + b.p0)*IAS15_H[n]/2. + a0)*dt*IAS15_H[n]
        # fmt: on
        print(f"n={n}, x={x[1]}")
        acc_state = SystemState(
            positions=x,
            velocities=v,
            gms=initial_system_state.gms,
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

        predictor_corrector_error = maxb6tmp / maxa
        print(predictor_corrector_error)
        print("\n\n\n")

    return maxa, maxb6tmp

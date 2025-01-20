# This is a pythonized/jaxified version of the IAS15 integrator from
# Rein & Spiegel (2015) (DOI: 10.1093/mnras/stu2164), currently implemented in REBOUND.
# The original code is available at https://github.com/hannorein/rebound/blob/0b5c85d836fec20bc284d1f1bb326f418e11f591/src/integrator_ias15.c
# Accessed Summer 2023, re-visited Fall 2024.

# Many thanks to the REBOUND developers for their work on this integrator,
# and for making it open source!

import jax

jax.config.update("jax_enable_x64", True)
from collections.abc import Callable

import chex
import jax.numpy as jnp

from jorbit.data.constants import (
    EPSILON,
    IAS15_C,
    IAS15_D,
    IAS15_H,
    IAS15_RR,
    IAS15_SAFETY_FACTOR,
    IAS15_EPS_Modified,
)
from jorbit.utils.states import IAS15IntegratorState, SystemState


@chex.dataclass
class IAS15Helper:
    # the equivalent of the reb_dp7 struct in rebound, but obviously without pointers
    p0: jnp.ndarray
    p1: jnp.ndarray
    p2: jnp.ndarray
    p3: jnp.ndarray
    p4: jnp.ndarray
    p5: jnp.ndarray
    p6: jnp.ndarray


def initialize_ias15_helper(n_particles):
    return IAS15Helper(
        p0=jnp.zeros((n_particles, 3), dtype=jnp.float64),
        p1=jnp.zeros((n_particles, 3), dtype=jnp.float64),
        p2=jnp.zeros((n_particles, 3), dtype=jnp.float64),
        p3=jnp.zeros((n_particles, 3), dtype=jnp.float64),
        p4=jnp.zeros((n_particles, 3), dtype=jnp.float64),
        p5=jnp.zeros((n_particles, 3), dtype=jnp.float64),
        p6=jnp.zeros((n_particles, 3), dtype=jnp.float64),
    )


def initialize_ias15_integrator_state(a0):
    n_particles = a0.shape[0]
    return IAS15IntegratorState(
        g=initialize_ias15_helper(n_particles),
        b=initialize_ias15_helper(n_particles),
        e=initialize_ias15_helper(n_particles),
        br=initialize_ias15_helper(n_particles),
        er=initialize_ias15_helper(n_particles),
        csx=jnp.zeros((n_particles, 3), dtype=jnp.float64),
        csv=jnp.zeros((n_particles, 3), dtype=jnp.float64),
        a0=a0,
        dt=10.0,  # 10 days
        dt_last_done=0.0,
    )


@jax.jit
def add_cs(p, csp, inp):
    y = inp - csp
    t = p + y
    csp = (t - p) - y
    p = t
    return p, csp


@jax.jit
def predict_next_step(ratio, _e, _b):
    e = IAS15Helper(
        p0=jnp.zeros_like(_e.p0, dtype=jnp.float64),
        p1=jnp.zeros_like(_e.p1, dtype=jnp.float64),
        p2=jnp.zeros_like(_e.p2, dtype=jnp.float64),
        p3=jnp.zeros_like(_e.p3, dtype=jnp.float64),
        p4=jnp.zeros_like(_e.p4, dtype=jnp.float64),
        p5=jnp.zeros_like(_e.p5, dtype=jnp.float64),
        p6=jnp.zeros_like(_e.p6, dtype=jnp.float64),
    )
    b = IAS15Helper(
        p0=jnp.zeros_like(_b.p0, dtype=jnp.float64),
        p1=jnp.zeros_like(_b.p1, dtype=jnp.float64),
        p2=jnp.zeros_like(_b.p2, dtype=jnp.float64),
        p3=jnp.zeros_like(_b.p3, dtype=jnp.float64),
        p4=jnp.zeros_like(_b.p4, dtype=jnp.float64),
        p5=jnp.zeros_like(_b.p5, dtype=jnp.float64),
        p6=jnp.zeros_like(_b.p6, dtype=jnp.float64),
    )

    def large_ratio(ratio, er, br):
        return e, b

    def reasonable_ratio(ratio, er, br):
        q1 = ratio
        q2 = q1 * q1
        q3 = q1 * q2
        q4 = q2 * q2
        q5 = q2 * q3
        q6 = q3 * q3
        q7 = q3 * q4

        be0 = _b.p0 - _e.p0
        be1 = _b.p1 - _e.p1
        be2 = _b.p2 - _e.p2
        be3 = _b.p3 - _e.p3
        be4 = _b.p4 - _e.p4
        be5 = _b.p5 - _e.p5
        be6 = _b.p6 - _e.p6

        e.p0 = q1 * (
            _b.p6 * 7.0
            + _b.p5 * 6.0
            + _b.p4 * 5.0
            + _b.p3 * 4.0
            + _b.p2 * 3.0
            + _b.p1 * 2.0
            + _b.p0
        )
        e.p1 = q2 * (
            _b.p6 * 21.0
            + _b.p5 * 15.0
            + _b.p4 * 10.0
            + _b.p3 * 6.0
            + _b.p2 * 3.0
            + _b.p1
        )
        e.p2 = q3 * (_b.p6 * 35.0 + _b.p5 * 20.0 + _b.p4 * 10.0 + _b.p3 * 4.0 + _b.p2)
        e.p3 = q4 * (_b.p6 * 35.0 + _b.p5 * 15.0 + _b.p4 * 5.0 + _b.p3)
        e.p4 = q5 * (_b.p6 * 21.0 + _b.p5 * 6.0 + _b.p4)
        e.p5 = q6 * (_b.p6 * 7.0 + _b.p5)
        e.p6 = q7 * _b.p6

        b.p0 = e.p0 + be0
        b.p1 = e.p1 + be1
        b.p2 = e.p2 + be2
        b.p3 = e.p3 + be3
        b.p4 = e.p4 + be4
        b.p5 = e.p5 + be5
        b.p6 = e.p6 + be6

        return e, b

    e, b = jax.lax.cond(ratio > 20.0, large_ratio, reasonable_ratio, ratio, _e, _b)
    return e, b


# TODO:
# note, the manual Horner method here isn't really necessary: jnp.polyval does that
# internally. Could equivalently swap the x, v calculations at each substep w/:
# b_len = 7
# b_x_denoms = (1+jnp.arange(1, b_len + 1, 1)) * (2+jnp.arange(1, b_len + 1, 1))
# b_v_denoms = jnp.arange(2, b_len + 2, 1)

# xcoeffs = jnp.zeros(b_len + 3)
# xcoeffs = xcoeffs.at[3:].set(bp * dt * dt / b_x_denoms)
# xcoeffs = xcoeffs.at[2].set(a0 * dt * dt / 2.0)
# xcoeffs = xcoeffs.at[1].set(v0 * dt)
# xcoeffs = xcoeffs.at[0].set(x0)
# xcoeffs = xcoeffs[::-1]

# vcoeffs = jnp.zeros(b_len + 2)
# vcoeffs = vcoeffs.at[2:].set(bp * dt / b_v_denoms)
# vcoeffs = vcoeffs.at[1].set(a0 * dt)
# vcoeffs = vcoeffs.at[0].set(v0)
# vcoeffs = vcoeffs[::-1]

# new_x = jnp.polyval(xcoeffs, h)
# new_v = jnp.polyval(vcoeffs, h)


@jax.jit
def ias15_step(
    initial_system_state: SystemState,
    acceleration_func: Callable[[SystemState], jnp.ndarray],
    initial_integrator_state: IAS15IntegratorState,
) -> SystemState:

    # for convenience, rename initial state
    t_beginning = initial_system_state.time
    M = initial_system_state.massive_positions.shape[0]
    x0 = jnp.concatenate(
        (initial_system_state.massive_positions, initial_system_state.tracer_positions)
    )
    v0 = jnp.concatenate(
        (
            initial_system_state.massive_velocities,
            initial_system_state.tracer_velocities,
        )
    )
    a0 = initial_integrator_state.a0

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
    def do_nothing(b, csb, g, predictor_corrector_error):
        # print("just chillin")
        return b, csb, g, predictor_corrector_error, predictor_corrector_error

    def predictor_corrector_iteration(b, csb, g, predictor_corrector_error):
        predictor_corrector_error_last = predictor_corrector_error
        predictor_corrector_error = 0.0

        # loop over each subinterval
        ################################################################################
        n = 1
        step_time = t_beginning + dt * IAS15_H[n]
        # get the new acceleration value at predicted position
        # fmt: off
        x = x0 - csx + ((((((((b.p6*7.*IAS15_H[n]/9. + b.p5)*3.*IAS15_H[n]/4. + b.p4)*5.*IAS15_H[n]/7. + b.p3)*2.*IAS15_H[n]/3. + b.p2)*3.*IAS15_H[n]/5. + b.p1)*IAS15_H[n]/2. + b.p0)*IAS15_H[n]/3. + a0)*dt*IAS15_H[n]/2. + v0)*dt*IAS15_H[n]
        v = v0 - csv + (((((((b.p6*7.*IAS15_H[n]/8. + b.p5)*6.*IAS15_H[n]/7. + b.p4)*5.*IAS15_H[n]/6. + b.p3)*4.*IAS15_H[n]/5. + b.p2)*3.*IAS15_H[n]/4. + b.p1)*2.*IAS15_H[n]/3. + b.p0)*IAS15_H[n]/2. + a0)*dt*IAS15_H[n]
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
        x = x0 - csx + ((((((((b.p6*7.*IAS15_H[n]/9. + b.p5)*3.*IAS15_H[n]/4. + b.p4)*5.*IAS15_H[n]/7. + b.p3)*2.*IAS15_H[n]/3. + b.p2)*3.*IAS15_H[n]/5. + b.p1)*IAS15_H[n]/2. + b.p0)*IAS15_H[n]/3. + a0)*dt*IAS15_H[n]/2. + v0)*dt*IAS15_H[n]
        v = v0 - csv + (((((((b.p6*7.*IAS15_H[n]/8. + b.p5)*6.*IAS15_H[n]/7. + b.p4)*5.*IAS15_H[n]/6. + b.p3)*4.*IAS15_H[n]/5. + b.p2)*3.*IAS15_H[n]/4. + b.p1)*2.*IAS15_H[n]/3. + b.p0)*IAS15_H[n]/2. + a0)*dt*IAS15_H[n]
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
        x = x0 - csx + ((((((((b.p6*7.*IAS15_H[n]/9. + b.p5)*3.*IAS15_H[n]/4. + b.p4)*5.*IAS15_H[n]/7. + b.p3)*2.*IAS15_H[n]/3. + b.p2)*3.*IAS15_H[n]/5. + b.p1)*IAS15_H[n]/2. + b.p0)*IAS15_H[n]/3. + a0)*dt*IAS15_H[n]/2. + v0)*dt*IAS15_H[n]
        v = v0 - csv + (((((((b.p6*7.*IAS15_H[n]/8. + b.p5)*6.*IAS15_H[n]/7. + b.p4)*5.*IAS15_H[n]/6. + b.p3)*4.*IAS15_H[n]/5. + b.p2)*3.*IAS15_H[n]/4. + b.p1)*2.*IAS15_H[n]/3. + b.p0)*IAS15_H[n]/2. + a0)*dt*IAS15_H[n]
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
        x = x0 - csx + ((((((((b.p6*7.*IAS15_H[n]/9. + b.p5)*3.*IAS15_H[n]/4. + b.p4)*5.*IAS15_H[n]/7. + b.p3)*2.*IAS15_H[n]/3. + b.p2)*3.*IAS15_H[n]/5. + b.p1)*IAS15_H[n]/2. + b.p0)*IAS15_H[n]/3. + a0)*dt*IAS15_H[n]/2. + v0)*dt*IAS15_H[n]
        v = v0 - csv + (((((((b.p6*7.*IAS15_H[n]/8. + b.p5)*6.*IAS15_H[n]/7. + b.p4)*5.*IAS15_H[n]/6. + b.p3)*4.*IAS15_H[n]/5. + b.p2)*3.*IAS15_H[n]/4. + b.p1)*2.*IAS15_H[n]/3. + b.p0)*IAS15_H[n]/2. + a0)*dt*IAS15_H[n]
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
        x = x0 - csx + ((((((((b.p6*7.*IAS15_H[n]/9. + b.p5)*3.*IAS15_H[n]/4. + b.p4)*5.*IAS15_H[n]/7. + b.p3)*2.*IAS15_H[n]/3. + b.p2)*3.*IAS15_H[n]/5. + b.p1)*IAS15_H[n]/2. + b.p0)*IAS15_H[n]/3. + a0)*dt*IAS15_H[n]/2. + v0)*dt*IAS15_H[n]
        v = v0 - csv + (((((((b.p6*7.*IAS15_H[n]/8. + b.p5)*6.*IAS15_H[n]/7. + b.p4)*5.*IAS15_H[n]/6. + b.p3)*4.*IAS15_H[n]/5. + b.p2)*3.*IAS15_H[n]/4. + b.p1)*2.*IAS15_H[n]/3. + b.p0)*IAS15_H[n]/2. + a0)*dt*IAS15_H[n]
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
        x = x0 - csx + ((((((((b.p6*7.*IAS15_H[n]/9. + b.p5)*3.*IAS15_H[n]/4. + b.p4)*5.*IAS15_H[n]/7. + b.p3)*2.*IAS15_H[n]/3. + b.p2)*3.*IAS15_H[n]/5. + b.p1)*IAS15_H[n]/2. + b.p0)*IAS15_H[n]/3. + a0)*dt*IAS15_H[n]/2. + v0)*dt*IAS15_H[n]
        v = v0 - csv + (((((((b.p6*7.*IAS15_H[n]/8. + b.p5)*6.*IAS15_H[n]/7. + b.p4)*5.*IAS15_H[n]/6. + b.p3)*4.*IAS15_H[n]/5. + b.p2)*3.*IAS15_H[n]/4. + b.p1)*2.*IAS15_H[n]/3. + b.p0)*IAS15_H[n]/2. + a0)*dt*IAS15_H[n]
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

        return b, csb, g, predictor_corrector_error, predictor_corrector_error_last

    # return predictor_corrector_iteration(b, csb, g, 1e300)

    def scan_func(carry, scan_over):
        b, csb, g, predictor_corrector_error, predictor_corrector_error_last = carry

        condition = (predictor_corrector_error < EPSILON) | (
            (scan_over > 2)
            & (predictor_corrector_error > predictor_corrector_error_last)
        )

        carry = jax.lax.cond(
            condition,
            do_nothing,
            predictor_corrector_iteration,
            b,
            csb,
            g,
            predictor_corrector_error,
        )
        return carry, None

    predictor_corrector_error = 1e300
    predictor_corrector_error_last = 2.0

    (b, csb, g, predictor_corrector_error, predictor_corrector_error_last), _ = (
        jax.lax.scan(
            scan_func,
            (b, csb, g, predictor_corrector_error, predictor_corrector_error_last),
            jnp.arange(10),
        )
    )

    # check the validity of the step, estimate next timestep
    dt_done = dt

    tmp = a0 + b.p0 + b.p1 + b.p2 + b.p3 + b.p4 + b.p5 + b.p6
    y2 = jnp.sum(tmp * tmp, axis=1)
    tmp = (
        b.p0
        + 2.0 * b.p1
        + 3.0 * b.p2
        + 4.0 * b.p3
        + 5.0 * b.p4
        + 6.0 * b.p5
        + 7.0 * b.p6
    )
    y3 = jnp.sum(tmp * tmp, axis=1)
    tmp = (
        2.0 * b.p1 + 6.0 * b.p2 + 12.0 * b.p3 + 20.0 * b.p4 + 30.0 * b.p5 + 42.0 * b.p6
    )
    y4 = jnp.sum(tmp * tmp, axis=1)
    tmp = 6.0 * b.p2 + 24.0 * b.p3 + 60.0 * b.p4 + 120.0 * b.p5 + 210.0 * b.p6

    timescale2 = 2.0 * y2 / (y3 + jnp.sqrt(y4 * y2))  # PRS23
    min_timescale2 = jnp.nanmin(timescale2)

    dt_new = jnp.sqrt(min_timescale2) * dt_done * IAS15_EPS_Modified
    # not checking for a min dt, since rebound default is 0.0 anyway
    # and we're willing to let it get tiny

    def step_too_ambitious(x0, v0, csx, csv):
        dt_done = 0.0
        return x0, v0, dt_done, dt_new

    def step_was_good(x0, v0, csx, csv):
        dt_neww = jnp.where(
            dt_new / dt_done > 1 / IAS15_SAFETY_FACTOR,
            dt_done / IAS15_SAFETY_FACTOR,
            dt_new,
        )

        x0, csx = add_cs(x0, csx, b.p6 / 72.0 * dt_done * dt_done)
        x0, csx = add_cs(x0, csx, b.p5 / 56.0 * dt_done * dt_done)
        x0, csx = add_cs(x0, csx, b.p4 / 42.0 * dt_done * dt_done)
        x0, csx = add_cs(x0, csx, b.p3 / 30.0 * dt_done * dt_done)
        x0, csx = add_cs(x0, csx, b.p2 / 20.0 * dt_done * dt_done)
        x0, csx = add_cs(x0, csx, b.p1 / 12.0 * dt_done * dt_done)
        x0, csx = add_cs(x0, csx, b.p0 / 6.0 * dt_done * dt_done)
        x0, csx = add_cs(x0, csx, a0 / 2.0 * dt_done * dt_done)
        x0, csx = add_cs(x0, csx, v0 * dt_done)
        v0, csv = add_cs(v0, csv, b.p6 / 8.0 * dt_done)
        v0, csv = add_cs(v0, csv, b.p5 / 7.0 * dt_done)
        v0, csv = add_cs(v0, csv, b.p4 / 6.0 * dt_done)
        v0, csv = add_cs(v0, csv, b.p3 / 5.0 * dt_done)
        v0, csv = add_cs(v0, csv, b.p2 / 4.0 * dt_done)
        v0, csv = add_cs(v0, csv, b.p1 / 3.0 * dt_done)
        v0, csv = add_cs(v0, csv, b.p0 / 2.0 * dt_done)
        v0, csv = add_cs(v0, csv, a0 * dt_done)

        return x0, v0, dt_done, dt_neww

    x0, v0, dt_done, dt_new = jax.lax.cond(
        jnp.abs(dt_new / dt_done) < IAS15_SAFETY_FACTOR,
        step_too_ambitious,
        step_was_good,
        x0,
        v0,
        csx,
        csv,
    )

    new_system_state = SystemState(
        massive_positions=x0[:M],
        massive_velocities=v0[:M],
        tracer_positions=x0[M:],
        tracer_velocities=v0[M:],
        log_gms=initial_system_state.log_gms,
        time=t_beginning + dt_done,
        acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
    )

    er = e
    br = b
    ratio = dt_new / dt
    # if we're rejecting the step, trick predict_next_step into not predicting
    ratio = jnp.where(dt_done == 0.0, 100.0, ratio)
    e, b = predict_next_step(ratio, er, br)

    new_integrator_state = IAS15IntegratorState(
        g=g,
        b=b,
        e=e,
        br=br,
        er=er,
        csx=csx,
        csv=csv,
        a0=acceleration_func(new_system_state),
        dt=dt_new,
        dt_last_done=dt_done,
    )

    return new_system_state, new_integrator_state


# @partial(jax.jit, static_argnames=("n_steps"))
# def ias15_evolve(
#     initial_system_state: SystemState,
#     acceleration_func: Callable[[SystemState], jnp.ndarray],
#     times: jnp.ndarray,
#     initial_integrator_state: IAS15IntegratorState,
#     n_steps: int = 1000,
# ):
#     def evolve(
#         initial_system_state, acceleration_func, final_time, initial_integrator_state
#     ):
#         def step_needed(system_state, acceleration_func, integrator_state):
#             system_state, integrator_state = ias15_step(
#                 system_state, acceleration_func, integrator_state
#             )
#             return system_state, integrator_state

#         def objective_reached(system_state, acceleration_func, integrator_state):
#             return system_state, integrator_state

#         def scan_func(carry, scan_over):
#             system_state, integrator_state, last_meaningful_dt = carry
#             t = system_state.time

#             step_length = jnp.sign(final_time - t) * jnp.min(
#                 jnp.array([jnp.abs(final_time - t), jnp.abs(integrator_state.dt)])
#             )
#             # jax.debug.print("step_length: {x}", x=step_length)
#             last_meaningful_dt = jnp.where(
#                 step_length == 0, last_meaningful_dt, step_length
#             )
#             integrator_state.dt = step_length
#             # jax.debug.print("integrator_state.dt: {x}", x=integrator_state.dt)

#             system_state, integrator_state = jax.lax.cond(
#                 step_length != 0,
#                 step_needed,
#                 objective_reached,
#                 *(system_state, acceleration_func, integrator_state),
#             )

#             return (system_state, integrator_state, last_meaningful_dt), None

#         (final_system_state, final_integrator_state, last_meaningful_dt), _ = (
#             jax.lax.scan(
#                 scan_func,
#                 (
#                     initial_system_state,
#                     initial_integrator_state,
#                     initial_integrator_state.dt,
#                 ),
#                 length=n_steps,
#             )
#         )
#         final_integrator_state.dt = last_meaningful_dt
#         final_integrator_state.dt_last_done = last_meaningful_dt

#         return (final_system_state, final_integrator_state)

#     def scan_func(carry, scan_over):
#         system_state, integrator_state = carry
#         final_time = scan_over
#         system_state, integrator_state = evolve(
#             system_state, acceleration_func, final_time, integrator_state
#         )
#         return (system_state, integrator_state), (
#             jnp.concatenate(
#                 (
#                     system_state.massive_positions,
#                     system_state.tracer_positions,
#                 )
#             ),
#             jnp.concatenate(
#                 (
#                     system_state.massive_velocities,
#                     system_state.tracer_velocities,
#                 )
#             ),
#         )

#     (final_system_state, final_integrator_state), (positions, velocities) = (
#         jax.lax.scan(scan_func, (initial_system_state, initial_integrator_state), times)
#     )
#     return positions, velocities, final_system_state, final_integrator_state


# @jax.jit
# def ias15_step_dynamic_predictor(
#     initial_system_state: SystemState,
#     acceleration_func: Callable[[SystemState], jnp.ndarray],
#     initial_integrator_state: IAS15IntegratorState,
# ) -> SystemState:

#     # for convenience, rename initial state
#     t_beginning = initial_system_state.time
#     M = initial_system_state.massive_positions.shape[0]
#     T = initial_system_state.tracer_positions.shape[0]
#     x0 = jnp.concatenate(
#         (initial_system_state.massive_positions, initial_system_state.tracer_positions)
#     )
#     v0 = jnp.concatenate(
#         (
#             initial_system_state.massive_velocities,
#             initial_system_state.tracer_velocities,
#         )
#     )
#     a0 = initial_integrator_state.a0

#     dt = initial_integrator_state.dt
#     csx = initial_integrator_state.csx
#     csv = initial_integrator_state.csv
#     g = initial_integrator_state.g
#     e = initial_integrator_state.e
#     b = initial_integrator_state.b
#     er = initial_integrator_state.er
#     br = initial_integrator_state.br

#     # always zero the compensation terms
#     csb = IAS15Helper(
#         p0=jnp.zeros_like(x0, dtype=jnp.float64),
#         p1=jnp.zeros_like(x0, dtype=jnp.float64),
#         p2=jnp.zeros_like(x0, dtype=jnp.float64),
#         p3=jnp.zeros_like(x0, dtype=jnp.float64),
#         p4=jnp.zeros_like(x0, dtype=jnp.float64),
#         p5=jnp.zeros_like(x0, dtype=jnp.float64),
#         p6=jnp.zeros_like(x0, dtype=jnp.float64),
#     )

#     # get the initial g terms from the b terms
#     g.p0 = (
#         b.p6 * IAS15_D[15]
#         + b.p5 * IAS15_D[10]
#         + b.p4 * IAS15_D[6]
#         + b.p3 * IAS15_D[3]
#         + b.p2 * IAS15_D[1]
#         + b.p1 * IAS15_D[0]
#         + b.p0
#     )
#     g.p1 = (
#         b.p6 * IAS15_D[16]
#         + b.p5 * IAS15_D[11]
#         + b.p4 * IAS15_D[7]
#         + b.p3 * IAS15_D[4]
#         + b.p2 * IAS15_D[2]
#         + b.p1
#     )
#     g.p2 = (
#         b.p6 * IAS15_D[17]
#         + b.p5 * IAS15_D[12]
#         + b.p4 * IAS15_D[8]
#         + b.p3 * IAS15_D[5]
#         + b.p2
#     )
#     g.p3 = b.p6 * IAS15_D[18] + b.p5 * IAS15_D[13] + b.p4 * IAS15_D[9] + b.p3
#     g.p4 = b.p6 * IAS15_D[19] + b.p5 * IAS15_D[14] + b.p4
#     g.p5 = b.p6 * IAS15_D[20] + b.p5
#     g.p6 = b.p6

#     def predictor_corrector_iteration(args):
#         (
#             b,
#             csb,
#             g,
#             predictor_corrector_error,
#             predictor_corrector_error_last,
#             iter_num,
#         ) = args
#         predictor_corrector_error_last = predictor_corrector_error
#         predictor_corrector_error = 0.0

#         # loop over each subinterval
#         ################################################################################
#         n = 1
#         step_time = t_beginning + dt * IAS15_H[n]
#         # get the new acceleration value at predicted position
#         # fmt: off
#         x = x0 - csx + ((((((((b.p6*7.*IAS15_H[n]/9. + b.p5)*3.*IAS15_H[n]/4. + b.p4)*5.*IAS15_H[n]/7. + b.p3)*2.*IAS15_H[n]/3. + b.p2)*3.*IAS15_H[n]/5. + b.p1)*IAS15_H[n]/2. + b.p0)*IAS15_H[n]/3. + a0)*dt*IAS15_H[n]/2. + v0)*dt*IAS15_H[n]
#         v = v0 - csv + (((((((b.p6*7.*IAS15_H[n]/8. + b.p5)*6.*IAS15_H[n]/7. + b.p4)*5.*IAS15_H[n]/6. + b.p3)*4.*IAS15_H[n]/5. + b.p2)*3.*IAS15_H[n]/4. + b.p1)*2.*IAS15_H[n]/3. + b.p0)*IAS15_H[n]/2. + a0)*dt*IAS15_H[n]
#         # fmt: on
#         acc_state = SystemState(
#             massive_positions=x[:M],
#             massive_velocities=v[:M],
#             tracer_positions=x[M:],
#             tracer_velocities=v[M:],
#             log_gms=initial_system_state.log_gms,
#             time=step_time,
#             acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
#         )
#         at = acceleration_func(acc_state)

#         tmp = g.p0
#         gk = at - a0
#         g.p0 = gk / IAS15_RR[0]
#         b.p0, csb.p0 = add_cs(b.p0, csb.p0, g.p0 - tmp)

#         ################################################################################
#         n = 2
#         step_time = t_beginning + dt * IAS15_H[n]
#         # fmt: off
#         x = x0 - csx + ((((((((b.p6*7.*IAS15_H[n]/9. + b.p5)*3.*IAS15_H[n]/4. + b.p4)*5.*IAS15_H[n]/7. + b.p3)*2.*IAS15_H[n]/3. + b.p2)*3.*IAS15_H[n]/5. + b.p1)*IAS15_H[n]/2. + b.p0)*IAS15_H[n]/3. + a0)*dt*IAS15_H[n]/2. + v0)*dt*IAS15_H[n]
#         v = v0 - csv + (((((((b.p6*7.*IAS15_H[n]/8. + b.p5)*6.*IAS15_H[n]/7. + b.p4)*5.*IAS15_H[n]/6. + b.p3)*4.*IAS15_H[n]/5. + b.p2)*3.*IAS15_H[n]/4. + b.p1)*2.*IAS15_H[n]/3. + b.p0)*IAS15_H[n]/2. + a0)*dt*IAS15_H[n]
#         # fmt: on
#         acc_state = SystemState(
#             massive_positions=x[:M],
#             massive_velocities=v[:M],
#             tracer_positions=x[M:],
#             tracer_velocities=v[M:],
#             log_gms=initial_system_state.log_gms,
#             time=step_time,
#             acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
#         )
#         at = acceleration_func(acc_state)

#         tmp = g.p1
#         gk = at - a0
#         g.p1 = (gk / IAS15_RR[1] - g.p0) / IAS15_RR[2]
#         tmp = g.p1 - tmp
#         b.p0, csb.p0 = add_cs(b.p0, csb.p0, tmp * IAS15_C[0])
#         b.p1, csb.p1 = add_cs(b.p1, csb.p1, tmp)

#         ################################################################################
#         n = 3
#         step_time = t_beginning + dt * IAS15_H[n]
#         # fmt: off
#         x = x0 - csx + ((((((((b.p6*7.*IAS15_H[n]/9. + b.p5)*3.*IAS15_H[n]/4. + b.p4)*5.*IAS15_H[n]/7. + b.p3)*2.*IAS15_H[n]/3. + b.p2)*3.*IAS15_H[n]/5. + b.p1)*IAS15_H[n]/2. + b.p0)*IAS15_H[n]/3. + a0)*dt*IAS15_H[n]/2. + v0)*dt*IAS15_H[n]
#         v = v0 - csv + (((((((b.p6*7.*IAS15_H[n]/8. + b.p5)*6.*IAS15_H[n]/7. + b.p4)*5.*IAS15_H[n]/6. + b.p3)*4.*IAS15_H[n]/5. + b.p2)*3.*IAS15_H[n]/4. + b.p1)*2.*IAS15_H[n]/3. + b.p0)*IAS15_H[n]/2. + a0)*dt*IAS15_H[n]
#         # fmt: on
#         acc_state = SystemState(
#             massive_positions=x[:M],
#             massive_velocities=v[:M],
#             tracer_positions=x[M:],
#             tracer_velocities=v[M:],
#             log_gms=initial_system_state.log_gms,
#             time=step_time,
#             acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
#         )
#         at = acceleration_func(acc_state)

#         tmp = g.p2
#         gk = at - a0
#         g.p2 = ((gk / IAS15_RR[3] - g.p0) / IAS15_RR[4] - g.p1) / IAS15_RR[5]
#         tmp = g.p2 - tmp
#         b.p0, csb.p0 = add_cs(b.p0, csb.p0, tmp * IAS15_C[1])
#         b.p1, csb.p1 = add_cs(b.p1, csb.p1, tmp * IAS15_C[2])
#         b.p2, csb.p2 = add_cs(b.p2, csb.p2, tmp)

#         ################################################################################
#         n = 4
#         step_time = t_beginning + dt * IAS15_H[n]
#         # fmt: off
#         x = x0 - csx + ((((((((b.p6*7.*IAS15_H[n]/9. + b.p5)*3.*IAS15_H[n]/4. + b.p4)*5.*IAS15_H[n]/7. + b.p3)*2.*IAS15_H[n]/3. + b.p2)*3.*IAS15_H[n]/5. + b.p1)*IAS15_H[n]/2. + b.p0)*IAS15_H[n]/3. + a0)*dt*IAS15_H[n]/2. + v0)*dt*IAS15_H[n]
#         v = v0 - csv + (((((((b.p6*7.*IAS15_H[n]/8. + b.p5)*6.*IAS15_H[n]/7. + b.p4)*5.*IAS15_H[n]/6. + b.p3)*4.*IAS15_H[n]/5. + b.p2)*3.*IAS15_H[n]/4. + b.p1)*2.*IAS15_H[n]/3. + b.p0)*IAS15_H[n]/2. + a0)*dt*IAS15_H[n]
#         # fmt: on
#         acc_state = SystemState(
#             massive_positions=x[:M],
#             massive_velocities=v[:M],
#             tracer_positions=x[M:],
#             tracer_velocities=v[M:],
#             log_gms=initial_system_state.log_gms,
#             time=step_time,
#             acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
#         )
#         at = acceleration_func(acc_state)

#         tmp = g.p3
#         gk = at - a0
#         g.p3 = (
#             ((gk / IAS15_RR[6] - g.p0) / IAS15_RR[7] - g.p1) / IAS15_RR[8] - g.p2
#         ) / IAS15_RR[9]
#         tmp = g.p3 - tmp
#         b.p0, csb.p0 = add_cs(b.p0, csb.p0, tmp * IAS15_C[3])
#         b.p1, csb.p1 = add_cs(b.p1, csb.p1, tmp * IAS15_C[4])
#         b.p2, csb.p2 = add_cs(b.p2, csb.p2, tmp * IAS15_C[5])
#         b.p3, csb.p3 = add_cs(b.p3, csb.p3, tmp)

#         ################################################################################
#         n = 5
#         step_time = t_beginning + dt * IAS15_H[n]
#         # fmt: off
#         x = x0 - csx + ((((((((b.p6*7.*IAS15_H[n]/9. + b.p5)*3.*IAS15_H[n]/4. + b.p4)*5.*IAS15_H[n]/7. + b.p3)*2.*IAS15_H[n]/3. + b.p2)*3.*IAS15_H[n]/5. + b.p1)*IAS15_H[n]/2. + b.p0)*IAS15_H[n]/3. + a0)*dt*IAS15_H[n]/2. + v0)*dt*IAS15_H[n]
#         v = v0 - csv + (((((((b.p6*7.*IAS15_H[n]/8. + b.p5)*6.*IAS15_H[n]/7. + b.p4)*5.*IAS15_H[n]/6. + b.p3)*4.*IAS15_H[n]/5. + b.p2)*3.*IAS15_H[n]/4. + b.p1)*2.*IAS15_H[n]/3. + b.p0)*IAS15_H[n]/2. + a0)*dt*IAS15_H[n]
#         # fmt: on
#         acc_state = SystemState(
#             massive_positions=x[:M],
#             massive_velocities=v[:M],
#             tracer_positions=x[M:],
#             tracer_velocities=v[M:],
#             log_gms=initial_system_state.log_gms,
#             time=step_time,
#             acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
#         )
#         at = acceleration_func(acc_state)

#         tmp = g.p4
#         gk = at - a0
#         g.p4 = (
#             (((gk / IAS15_RR[10] - g.p0) / IAS15_RR[11] - g.p1) / IAS15_RR[12] - g.p2)
#             / IAS15_RR[13]
#             - g.p3
#         ) / IAS15_RR[14]
#         tmp = g.p4 - tmp
#         b.p0, csb.p0 = add_cs(b.p0, csb.p0, tmp * IAS15_C[6])
#         b.p1, csb.p1 = add_cs(b.p1, csb.p1, tmp * IAS15_C[7])
#         b.p2, csb.p2 = add_cs(b.p2, csb.p2, tmp * IAS15_C[8])
#         b.p3, csb.p3 = add_cs(b.p3, csb.p3, tmp * IAS15_C[9])
#         b.p4, csb.p4 = add_cs(b.p4, csb.p4, tmp)

#         ################################################################################
#         n = 6
#         step_time = t_beginning + dt * IAS15_H[n]
#         # fmt: off
#         x = x0 - csx + ((((((((b.p6*7.*IAS15_H[n]/9. + b.p5)*3.*IAS15_H[n]/4. + b.p4)*5.*IAS15_H[n]/7. + b.p3)*2.*IAS15_H[n]/3. + b.p2)*3.*IAS15_H[n]/5. + b.p1)*IAS15_H[n]/2. + b.p0)*IAS15_H[n]/3. + a0)*dt*IAS15_H[n]/2. + v0)*dt*IAS15_H[n]
#         v = v0 - csv + (((((((b.p6*7.*IAS15_H[n]/8. + b.p5)*6.*IAS15_H[n]/7. + b.p4)*5.*IAS15_H[n]/6. + b.p3)*4.*IAS15_H[n]/5. + b.p2)*3.*IAS15_H[n]/4. + b.p1)*2.*IAS15_H[n]/3. + b.p0)*IAS15_H[n]/2. + a0)*dt*IAS15_H[n]
#         # fmt: on
#         acc_state = SystemState(
#             massive_positions=x[:M],
#             massive_velocities=v[:M],
#             tracer_positions=x[M:],
#             tracer_velocities=v[M:],
#             log_gms=initial_system_state.log_gms,
#             time=step_time,
#             acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
#         )
#         at = acceleration_func(acc_state)

#         tmp = g.p5
#         gk = at - a0
#         g.p5 = (
#             (
#                 (
#                     ((gk / IAS15_RR[15] - g.p0) / IAS15_RR[16] - g.p1) / IAS15_RR[17]
#                     - g.p2
#                 )
#                 / IAS15_RR[18]
#                 - g.p3
#             )
#             / IAS15_RR[19]
#             - g.p4
#         ) / IAS15_RR[20]
#         tmp = g.p5 - tmp
#         b.p0, csb.p0 = add_cs(b.p0, csb.p0, tmp * IAS15_C[10])
#         b.p1, csb.p1 = add_cs(b.p1, csb.p1, tmp * IAS15_C[11])
#         b.p2, csb.p2 = add_cs(b.p2, csb.p2, tmp * IAS15_C[12])
#         b.p3, csb.p3 = add_cs(b.p3, csb.p3, tmp * IAS15_C[13])
#         b.p4, csb.p4 = add_cs(b.p4, csb.p4, tmp * IAS15_C[14])
#         b.p5, csb.p5 = add_cs(b.p5, csb.p5, tmp)

#         ################################################################################
#         n = 7
#         step_time = t_beginning + dt * IAS15_H[n]
#         # fmt: off
#         x = x0 - csx + ((((((((b.p6*7.*IAS15_H[n]/9. + b.p5)*3.*IAS15_H[n]/4. + b.p4)*5.*IAS15_H[n]/7. + b.p3)*2.*IAS15_H[n]/3. + b.p2)*3.*IAS15_H[n]/5. + b.p1)*IAS15_H[n]/2. + b.p0)*IAS15_H[n]/3. + a0)*dt*IAS15_H[n]/2. + v0)*dt*IAS15_H[n]
#         v = v0 - csv + (((((((b.p6*7.*IAS15_H[n]/8. + b.p5)*6.*IAS15_H[n]/7. + b.p4)*5.*IAS15_H[n]/6. + b.p3)*4.*IAS15_H[n]/5. + b.p2)*3.*IAS15_H[n]/4. + b.p1)*2.*IAS15_H[n]/3. + b.p0)*IAS15_H[n]/2. + a0)*dt*IAS15_H[n]
#         # fmt: on
#         acc_state = SystemState(
#             massive_positions=x[:M],
#             massive_velocities=v[:M],
#             tracer_positions=x[M:],
#             tracer_velocities=v[M:],
#             log_gms=initial_system_state.log_gms,
#             time=step_time,
#             acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
#         )
#         at = acceleration_func(acc_state)

#         tmp = g.p6
#         gk = at - a0
#         g.p6 = (
#             (
#                 (
#                     (
#                         ((gk / IAS15_RR[21] - g.p0) / IAS15_RR[22] - g.p1)
#                         / IAS15_RR[23]
#                         - g.p2
#                     )
#                     / IAS15_RR[24]
#                     - g.p3
#                 )
#                 / IAS15_RR[25]
#                 - g.p4
#             )
#             / IAS15_RR[26]
#             - g.p5
#         ) / IAS15_RR[27]
#         tmp = g.p6 - tmp
#         b.p0, csb.p0 = add_cs(b.p0, csb.p0, tmp * IAS15_C[15])
#         b.p1, csb.p1 = add_cs(b.p1, csb.p1, tmp * IAS15_C[16])
#         b.p2, csb.p2 = add_cs(b.p2, csb.p2, tmp * IAS15_C[17])
#         b.p3, csb.p3 = add_cs(b.p3, csb.p3, tmp * IAS15_C[18])
#         b.p4, csb.p4 = add_cs(b.p4, csb.p4, tmp * IAS15_C[19])
#         b.p5, csb.p5 = add_cs(b.p5, csb.p5, tmp * IAS15_C[20])
#         b.p6, csb.p6 = add_cs(b.p6, csb.p6, tmp)

#         maxa = jnp.max(jnp.abs(at))
#         maxb6tmp = jnp.max(jnp.abs(tmp))

#         predictor_corrector_error = jnp.abs(maxb6tmp / maxa)

#         return (
#             b,
#             csb,
#             g,
#             predictor_corrector_error,
#             predictor_corrector_error_last,
#             iter_num + 1,
#         )

#     def cond_func(args):
#         (
#             b,
#             csb,
#             g,
#             predictor_corrector_error,
#             predictor_corrector_error_last,
#             iter_num,
#         ) = args
#         return (
#             (predictor_corrector_error < EPSILON)
#             | (
#                 (iter_num > 2)
#                 & (predictor_corrector_error > predictor_corrector_error_last)
#             )
#             | (iter_num > 10)
#         )

#     predictor_corrector_error = 1e300
#     predictor_corrector_error_last = 2.0
#     iter_num = 0
#     b, csb, g, predictor_corrector_error, predictor_corrector_error_last, iter_num = (
#         jax.lax.while_loop(
#             cond_func,
#             predictor_corrector_iteration,
#             (
#                 b,
#                 csb,
#                 g,
#                 predictor_corrector_error,
#                 predictor_corrector_error_last,
#                 iter_num,
#             ),
#         )
#     )

#     # check the validity of the step, estimate next timestep
#     dt_done = dt

#     a0i = jnp.sum(a0 * a0, axis=1)
#     tmp = a0 + b.p0 + b.p1 + b.p2 + b.p3 + b.p4 + b.p5 + b.p6
#     y2 = jnp.sum(tmp * tmp, axis=1)
#     tmp = (
#         b.p0
#         + 2.0 * b.p1
#         + 3.0 * b.p2
#         + 4.0 * b.p3
#         + 5.0 * b.p4
#         + 6.0 * b.p5
#         + 7.0 * b.p6
#     )
#     y3 = jnp.sum(tmp * tmp, axis=1)
#     tmp = (
#         2.0 * b.p1 + 6.0 * b.p2 + 12.0 * b.p3 + 20.0 * b.p4 + 30.0 * b.p5 + 42.0 * b.p6
#     )
#     y4 = jnp.sum(tmp * tmp, axis=1)
#     tmp = 6.0 * b.p2 + 24.0 * b.p3 + 60.0 * b.p4 + 120.0 * b.p5 + 210.0 * b.p6
#     y5 = jnp.sum(tmp * tmp, axis=1)

#     timescale2 = 2.0 * y2 / (y3 + jnp.sqrt(y4 * y2))  # PRS23
#     min_timescale2 = jnp.nanmin(timescale2)

#     dt_new = jnp.sqrt(min_timescale2) * dt_done * IAS15_EPS_Modified
#     # not checking for a min dt, since rebound default is 0.0 anyway
#     # and we're willing to let it get tiny

#     def step_too_ambitious(x0, v0, csx, csv):
#         dt_done = 0.0
#         return x0, v0, dt_done, dt_new

#     def step_was_good(x0, v0, csx, csv):
#         dt_neww = jnp.where(
#             dt_new / dt_done > 1 / IAS15_SAFETY_FACTOR,
#             dt_done / IAS15_SAFETY_FACTOR,
#             dt_new,
#         )

#         x0, csx = add_cs(x0, csx, b.p6 / 72.0 * dt_done * dt_done)
#         x0, csx = add_cs(x0, csx, b.p5 / 56.0 * dt_done * dt_done)
#         x0, csx = add_cs(x0, csx, b.p4 / 42.0 * dt_done * dt_done)
#         x0, csx = add_cs(x0, csx, b.p3 / 30.0 * dt_done * dt_done)
#         x0, csx = add_cs(x0, csx, b.p2 / 20.0 * dt_done * dt_done)
#         x0, csx = add_cs(x0, csx, b.p1 / 12.0 * dt_done * dt_done)
#         x0, csx = add_cs(x0, csx, b.p0 / 6.0 * dt_done * dt_done)
#         x0, csx = add_cs(x0, csx, a0 / 2.0 * dt_done * dt_done)
#         x0, csx = add_cs(x0, csx, v0 * dt_done)
#         v0, csv = add_cs(v0, csv, b.p6 / 8.0 * dt_done)
#         v0, csv = add_cs(v0, csv, b.p5 / 7.0 * dt_done)
#         v0, csv = add_cs(v0, csv, b.p4 / 6.0 * dt_done)
#         v0, csv = add_cs(v0, csv, b.p3 / 5.0 * dt_done)
#         v0, csv = add_cs(v0, csv, b.p2 / 4.0 * dt_done)
#         v0, csv = add_cs(v0, csv, b.p1 / 3.0 * dt_done)
#         v0, csv = add_cs(v0, csv, b.p0 / 2.0 * dt_done)
#         v0, csv = add_cs(v0, csv, a0 * dt_done)

#         return x0, v0, dt_done, dt_neww

#     x0, v0, dt_done, dt_new = jax.lax.cond(
#         jnp.abs(dt_new / dt_done) < IAS15_SAFETY_FACTOR,
#         step_too_ambitious,
#         step_was_good,
#         x0,
#         v0,
#         csx,
#         csv,
#     )

#     new_system_state = SystemState(
#         massive_positions=x0[:M],
#         massive_velocities=v0[:M],
#         tracer_positions=x0[M:],
#         tracer_velocities=v0[M:],
#         log_gms=initial_system_state.log_gms,
#         time=t_beginning + dt_done,
#         acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
#     )

#     er = e
#     br = b
#     ratio = dt_new / dt
#     # if we're rejecting the step, trick predict_next_step into not predicting
#     ratio = jnp.where(dt_done == 0.0, 100.0, ratio)
#     e, b = predict_next_step(ratio, er, br)

#     new_integrator_state = IAS15IntegratorState(
#         g=g,
#         b=b,
#         e=e,
#         br=br,
#         er=er,
#         csx=csx,
#         csv=csv,
#         a0=acceleration_func(new_system_state),
#         dt=dt_new,
#         dt_last_done=dt_done,
#     )

#     return new_system_state, new_integrator_state


@jax.jit
def ias15_evolve(
    initial_system_state: SystemState,
    acceleration_func: Callable[[SystemState], jnp.ndarray],
    times: jnp.ndarray,
    initial_integrator_state: IAS15IntegratorState,
):
    def evolve(
        initial_system_state, acceleration_func, final_time, initial_integrator_state
    ):
        def step_needed(args):
            system_state, integrator_state, last_meaningful_dt, iter_num = args

            t = system_state.time

            step_length = jnp.sign(final_time - t) * jnp.min(
                jnp.array([jnp.abs(final_time - t), jnp.abs(integrator_state.dt)])
            )

            last_meaningful_dt = jnp.where(
                step_length == 0, last_meaningful_dt, step_length
            )
            integrator_state.dt = step_length
            # system_state, integrator_state = ias15_step_dynamic_predictor(
            #     system_state, acceleration_func, integrator_state
            # )
            system_state, integrator_state = ias15_step(
                system_state, acceleration_func, integrator_state
            )
            return system_state, integrator_state, last_meaningful_dt, iter_num + 1

        def cond_func(args):
            system_state, integrator_state, last_meaningful_dt, iter_num = args
            t = system_state.time

            step_length = jnp.sign(final_time - t) * jnp.min(
                jnp.array([jnp.abs(final_time - t), jnp.abs(integrator_state.dt)])
            )
            return (step_length != 0) | (iter_num > 10_000)

        final_system_state, final_integrator_state, last_meaningful_dt, iter_num = (
            jax.lax.while_loop(
                cond_func,
                step_needed,
                (
                    initial_system_state,
                    initial_integrator_state,
                    initial_integrator_state.dt,
                    0,
                ),
            )
        )

        final_integrator_state.dt = last_meaningful_dt
        final_integrator_state.dt_last_done = last_meaningful_dt

        return (final_system_state, final_integrator_state)

    def scan_func(carry, scan_over):
        system_state, integrator_state = carry
        final_time = scan_over
        system_state, integrator_state = evolve(
            system_state, acceleration_func, final_time, integrator_state
        )
        return (system_state, integrator_state), (
            jnp.concatenate(
                (
                    system_state.massive_positions,
                    system_state.tracer_positions,
                )
            ),
            jnp.concatenate(
                (
                    system_state.massive_velocities,
                    system_state.tracer_velocities,
                )
            ),
        )

    (final_system_state, final_integrator_state), (positions, velocities) = (
        jax.lax.scan(scan_func, (initial_system_state, initial_integrator_state), times)
    )
    return positions, velocities, final_system_state, final_integrator_state

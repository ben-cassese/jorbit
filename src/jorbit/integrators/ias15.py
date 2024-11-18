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
    for iteration in range(10):
        if predictor_corrector_error < 1e-16:
            break
        elif (iteration > 2) and (
            predictor_corrector_error > predictor_corrector_error_last
        ):
            break
        predictor_corrector_error_last = predictor_corrector_error
        predictor_corrector_error = 0.0

        # loop over each subinterval
        for n in jnp.arange(1, 8):
            step_time = t_beginning + dt * IAS15_H[n]
            # get the new acceleration value at predicted position
            x = (
                -csx
                + (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (b.p6 * 7.0 * IAS15_H[n] / 9.0 + b.p5)
                                            * 3.0
                                            * IAS15_H[n]
                                            / 4.0
                                            + b.p4
                                        )
                                        * 5.0
                                        * IAS15_H[n]
                                        / 7.0
                                        + b.p3
                                    )
                                    * 2.0
                                    * IAS15_H[n]
                                    / 3.0
                                    + b.p2
                                )
                                * 3.0
                                * IAS15_H[n]
                                / 5.0
                                + b.p1
                            )
                            * IAS15_H[n]
                            / 2.0
                            + b.p0
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
            v = (
                -csv
                + (
                    (
                        (
                            (
                                (
                                    (
                                        (b.p6 * 7.0 * IAS15_H[n] / 8.0 + b.p5)
                                        * 6.0
                                        * IAS15_H[n]
                                        / 7.0
                                        + b.p4
                                    )
                                    * 5.0
                                    * IAS15_H[n]
                                    / 6.0
                                    + b.p3
                                )
                                * 4.0
                                * IAS15_H[n]
                                / 5.0
                                + b.p2
                            )
                            * 3.0
                            * IAS15_H[n]
                            / 4.0
                            + b.p1
                        )
                        * 2.0
                        * IAS15_H[n]
                        / 3.0
                        + b.p0
                    )
                    * IAS15_H[n]
                    / 2.0
                    + a0
                )
                * dt
                * IAS15_H[n]
            )
            acc_state = SystemState(
                positions=x,
                velocities=v,
                gms=initial_system_state.gms,
                time=step_time,
                acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
            )
            at = acceleration_func(acc_state)

        return x, v

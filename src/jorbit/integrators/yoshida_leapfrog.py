"""A JAX implementation of the leapfrog integrator from Yoshida (1990).

Please see/cite https://doi.org/10.1016/0375-9601(90)90092-3. Includes ability for a
4th, 6th, and 8th order integrator via appropriate choice of coefficients (all of which
are pre-computed using routines below and stored in jorbit.data.constants).
"""

import jax

jax.config.update("jax_enable_x64", True)
from typing import Callable

# import warnings
import jax.numpy as jnp

from jorbit.utils.states import LeapfrogIntegratorState, SystemState


@jax.jit
def leapfrog_step(
    initial_system_state: SystemState,
    acceleration_func: Callable[[SystemState], jnp.ndarray],
    initial_integrator_state: LeapfrogIntegratorState,
) -> SystemState:
    """Take a single step using a Yoshida leapfrog integrator.

    Args:
        initial_system_state (SystemState):
            The initial system state.
        acceleration_func (Callable[[SystemState], jnp.ndarray]):
            The acceleration function.
        initial_integrator_state (LeapfrogIntegratorState):
            The initial integrator state.

    Returns:
        SystemState:
            The new system state.
    """
    t0 = initial_system_state.time
    num_massive = initial_system_state.massive_positions.shape[0]
    x0 = jnp.concatenate(
        [initial_system_state.massive_positions, initial_system_state.tracer_positions],
        axis=0,
    )
    v0 = jnp.concatenate(
        [
            initial_system_state.massive_velocities,
            initial_system_state.tracer_velocities,
        ],
        axis=0,
    )

    dt = initial_integrator_state.dt
    C = initial_integrator_state.C
    D = initial_integrator_state.D

    def leapfrog_scan(X: tuple, mid_step_coeffs: tuple) -> tuple[tuple, None]:
        x, v, tau = X
        c, d = mid_step_coeffs
        x = x + c * v * dt
        tau = tau + c
        # potential to do: modify this to be more like ias15_static_step where you can
        # pass perturber positions and velocities at each substep. For now, just pass
        # empty arrays and hope the acceleration function isn't banking on them.
        acc = acceleration_func(
            SystemState(
                massive_positions=x[:num_massive],
                massive_velocities=v[:num_massive],
                tracer_positions=x[num_massive:],
                tracer_velocities=v[num_massive:],
                log_gms=initial_system_state.log_gms,
                acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
                fixed_perturber_positions=jnp.empty((0, 3)),
                fixed_perturber_velocities=jnp.empty((0, 3)),
                fixed_perturber_log_gms=jnp.empty((0,)),
                time=t0 + tau * dt,
            )
        )
        v = v + d * acc * dt
        return (x, v, tau), None

    (x, v, _tau), _ = jax.lax.scan(leapfrog_scan, (x0, v0, 0.0), (C[:-1], D))
    x = x + C[-1] * v * dt
    return SystemState(
        massive_positions=x[:num_massive],
        massive_velocities=v[:num_massive],
        tracer_positions=x[num_massive:],
        tracer_velocities=v[num_massive:],
        log_gms=initial_system_state.log_gms,
        acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
        time=t0 + dt,
        fixed_perturber_positions=jnp.empty((0, 3)),
        fixed_perturber_velocities=jnp.empty((0, 3)),
        fixed_perturber_log_gms=jnp.empty((0,)),
    )


def leapfrog_evolve(
    initial_system_state: SystemState,
    acceleration_func: Callable[[SystemState], jnp.ndarray],
    times: jnp.ndarray,
    initial_integrator_state: LeapfrogIntegratorState,
) -> tuple[jnp.ndarray, jnp.ndarray, SystemState, LeapfrogIntegratorState]:
    """Evolve a system to multiple different times using a Yoshida leapfrog integrator.

    .. warnging::
        This function does not accept an argument for the step size: it assumes that all
        jumps between neighboring timestamps can be done in a single step. Be sure to
        use :func:`create_leapfrog_times` to create a time array with appropriate
        intermediate steps if necessary.

    Args:
        initial_system_state (SystemState):
            The initial state of the system.
        acceleration_func (Callable[[SystemState], jnp.ndarray]):
            The acceleration function to use.
        times (jnp.ndarray):
            The times to evolve the system to.
        initial_integrator_state (LeapfrogIntegratorState):
            The initial state of the integrator.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, SystemState, LeapfrogIntegratorState]:
            The positions and velocities of the system at each timestep,
            the final state of the system, and the final state of the integrator.
    """

    def scan_func(carry: tuple, scan_over: float) -> tuple[tuple, tuple]:
        system_state, integrator_state = carry
        tf = scan_over
        dt = tf - system_state.time
        integrator_state.dt = jnp.abs(dt)
        new_state = leapfrog_step(system_state, acceleration_func, integrator_state)
        return (new_state, integrator_state), (
            jnp.concatenate(
                (
                    new_state.massive_positions,
                    new_state.tracer_positions,
                )
            ),
            jnp.concatenate(
                (
                    new_state.massive_velocities,
                    new_state.tracer_velocities,
                )
            ),
        )

    (final_system_state, final_integrator_state), (positions, velocities) = (
        jax.lax.scan(scan_func, (initial_system_state, initial_integrator_state), times)
    )
    return positions, velocities, final_system_state, final_integrator_state


def create_leapfrog_times(
    t0: float, times: jnp.ndarray, biggest_allowed_dt: float
) -> jnp.ndarray:
    """Create an expanded array of times for leapfrog integration.

    Since :func:`leapfrog_evolve` assumes that all jumps between neighboring
    timestamps can be done in a single step, this function creates an expanded array of
    times by inserting intermediate steps as necessary to ensure that no step is
    larger than `biggest_allowed_dt`. Also returns the indices of the original times in
    the expanded array.

    Args:
        t0 (float):
            The initial time.
        times (jnp.ndarray):
            The desired output times.
        biggest_allowed_dt (float):
            The maximum allowed step size.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]:
            The expanded array of times, and the indices of the original times in the
            expanded array.
    """
    time_deltas = jnp.diff(times, prepend=t0)
    step_times = jnp.array([])
    inds = jnp.array([])
    for jump in time_deltas:
        if jump == 0:
            step_times = jnp.concatenate([step_times, jnp.array([t0])])
            inds = jnp.concatenate([inds, jnp.array([0])])
            continue
        step_size = jnp.sign(jump) * jnp.min(
            jnp.abs(jnp.array([jump, biggest_allowed_dt]))
        )
        steps_needed = jnp.ceil(jump / step_size).astype(jnp.int32)
        intermediate_times = jnp.linspace(t0, t0 + jump, steps_needed + 1)[1:]
        step_times = jnp.concatenate([step_times, intermediate_times])
        inds = jnp.concatenate([inds, jnp.array([step_times.shape[0] - 1])])
        t0 += jump
    # if len(jnp.unique(jnp.diff(step_times))) > 1:
    #     warnings.warn(
    #         "Step sizes are not uniform for this time sequence, which may lead to reduced accuracy in the integrator.",
    #         RuntimeWarning,
    #     )
    return step_times, inds.astype(jnp.int32)


def _create_yoshida_coeffs(Ws: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    w0 = 1 - 2 * (jnp.sum(Ws))
    w = jnp.concatenate((jnp.array([w0]), Ws))

    Ds = jnp.zeros(2 * len(Ws) + 1)
    Ds = Ds.at[: len(Ws)].set(Ws[::-1])
    Ds = Ds.at[len(Ws)].set(w0)
    Ds = Ds.at[len(Ws) + 1 :].set(Ws)

    Cs = jnp.zeros(2 * len(Ws) + 2)
    for i in range(len(w) - 1):
        Cs = Cs.at[i + 1].set(0.5 * (w[len(w) - 1 - i] + w[len(w) - 2 - i]))

    Cs = Cs.at[int(len(Cs) / 2) :].set(Cs[: int(len(Cs) / 2)][::-1])
    Cs = Cs.at[0].set(0.5 * w[-1])
    Cs = Cs.at[-1].set(0.5 * w[-1])

    # to do it at extended precision, use Decimal package:
    # tmp = 0
    # for i in Ws:
    #     tmp += i
    # w0 = 1 - 2 * tmp
    # w = [w0] + Ws

    # Ds = [0]*(2 * len(Ws) + 1)
    # Ds[:len(Ws)] = Ws[::-1]
    # Ds[len(Ws)] = w0
    # Ds[len(Ws) + 1:] = Ws

    # Cs = [0]*(2 * len(Ws) + 2)
    # for i in range(len(w) - 1):
    #     Cs[i + 1] = Decimal(0.5) * (w[len(w) - 1 - i] + w[len(w) - 2 - i])
    # Cs[int(len(Cs) / 2):] = Cs[: int(len(Cs) / 2)][::-1]
    # Cs[0] = Decimal(0.5) * w[-1]
    # Cs[-1] = Decimal(0.5) * w[-1]

    return jnp.array(Cs), jnp.array(Ds)

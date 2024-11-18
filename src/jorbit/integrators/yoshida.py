import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from typing import Callable
from functools import partial

from jorbit.utils.states import SystemState
from jorbit.integrators import LeapfrogIntegratorState


@jax.jit
def yoshida_step(
    initial_system_state: SystemState,
    acceleration_func: Callable[[SystemState], jnp.ndarray],
    initial_integrator_state: LeapfrogIntegratorState,
) -> SystemState:
    def leapfrog_scan(carry, scan_over):
        x, v = carry
        c, d = scan_over

        x = x + c * v * initial_integrator_state.dt
        _acc_state = SystemState(
            positions=x,
            velocities=v,
            gms=initial_system_state.gms,
            time=initial_system_state.time,
            acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
        )
        acc = acceleration_func(_acc_state)
        v = v + d * acc * initial_integrator_state.dt

        return (x, v), None

    x0 = initial_system_state.positions
    v0 = initial_system_state.velocities
    C = initial_integrator_state.c_coeff
    D = initial_integrator_state.d_coeff

    q = jax.lax.scan(leapfrog_scan, (x0, v0), (C[:-1], D))[0]
    x, v = q
    x = x + C[-1] * v * initial_integrator_state.dt

    new_system_state = SystemState(
        positions=x,
        velocities=v,
        gms=initial_system_state.gms,
        time=initial_system_state.time + initial_integrator_state.dt,
        acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
    )

    new_integrator_state = initial_integrator_state

    return new_system_state, new_integrator_state


@partial(jax.jit, static_argnames=("n_steps"))
def leapfrog_evolve(
    initial_system_state: SystemState,
    acceleration_func: Callable[[SystemState], jnp.ndarray],
    final_time: float,
    initial_integrator_state: LeapfrogIntegratorState,
    n_steps: int = 1000,
):
    _dt = initial_integrator_state.dt

    def step_needed(system_state, acceleration_func, integrator_state):
        system_state, integrator_state = yoshida_step(
            system_state, acceleration_func, integrator_state
        )
        return system_state, integrator_state

    def objective_reached(system_state, acceleration_func, integrator_state):
        return system_state, integrator_state

    def scan_func(carry, scan_over):
        system_state, integrator_state = carry
        t = system_state.time
        step_length = jnp.min(
            jnp.array([jnp.abs(final_time - t), jnp.abs(integrator_state.dt)])
        )

        step_length = jnp.sign(final_time - t) * jnp.min(jnp.array([step_length, _dt]))

        integrator_state.dt = step_length

        system_state, integrator_state = jax.lax.cond(
            step_length != 0,
            step_needed,
            objective_reached,
            *(system_state, acceleration_func, integrator_state),
        )

        return (system_state, integrator_state), None

    (final_system_state, final_integrator_state), _ = jax.lax.scan(
        scan_func, (initial_system_state, initial_integrator_state), length=n_steps
    )

    final_integrator_state.dt = _dt

    return (final_system_state, final_integrator_state)

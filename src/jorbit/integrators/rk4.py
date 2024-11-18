import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from typing import Callable
from functools import partial

from jorbit.utils.states import SystemState
from jorbit.integrators import RK4IntegratorState


@jax.jit
def rk4_step(
    initial_system_state: SystemState,
    acceleration_func: Callable[[SystemState], jnp.ndarray],
    initial_integrator_state: RK4IntegratorState,
) -> SystemState:
    dt = initial_integrator_state.dt
    x = initial_system_state.positions
    v = initial_system_state.velocities
    a = acceleration_func(initial_system_state)

    k1_x = v
    k1_v = a

    k2_x = v + k1_v * dt / 2
    _k2v_state = SystemState(
        positions=x + k1_x * dt / 2,
        velocities=v + k1_v * dt / 2,
        gms=initial_system_state.gms,
        time=initial_system_state.time + dt / 2,
        acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
    )
    k2_v = acceleration_func(_k2v_state)

    k3_x = v + k2_v * dt / 2
    _k3v_state = SystemState(
        positions=x + k2_x * dt / 2,
        velocities=v + k2_v * dt / 2,
        gms=initial_system_state.gms,
        time=initial_system_state.time + dt / 2,
        acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
    )
    k3_v = acceleration_func(_k3v_state)

    k4_x = v + k3_v * dt
    _k4v_state = SystemState(
        positions=x + k3_x * dt,
        velocities=v + k3_v * dt,
        gms=initial_system_state.gms,
        time=initial_system_state.time + dt,
        acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
    )
    k4_v = acceleration_func(_k4v_state)

    x_new = x + dt * (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
    v_new = v + dt * (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6

    new_state = SystemState(
        positions=x_new,
        velocities=v_new,
        gms=initial_system_state.gms,
        time=initial_system_state.time + dt,
        acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
    )

    new_integrator_state = initial_integrator_state

    return new_state, new_integrator_state


@partial(jax.jit, static_argnames=("n_steps"))
def rk4_evolve(
    initial_system_state: SystemState,
    acceleration_func: Callable[[SystemState], jnp.ndarray],
    final_time: float,
    initial_integrator_state: RK4IntegratorState,
    n_steps: int = 1000,
) -> SystemState:
    _dt = initial_integrator_state.dt

    def step_needed(system_state, acceleration_func, integrator_state):
        system_state, integrator_state = rk4_step(
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

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from typing import Callable
from functools import partial

from jorbit.utils.system_state import SystemState
from jorbit.integrators import IntegratorState


@jax.jit
def yoshida_step(
    initial_system_state: SystemState,
    acceleration_func: Callable[[SystemState], jnp.ndarray],
    initial_integrator_state: IntegratorState,
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
    C = initial_integrator_state.meta["leapfrog_coeffs"]["C"]
    D = initial_integrator_state.meta["leapfrog_coeffs"]["D"]

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

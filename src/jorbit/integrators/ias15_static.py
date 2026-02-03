"""A version of the standard-precision IAS15 integrator with all logic/adaptivity removed."""

import jax

jax.config.update("jax_enable_x64", True)
from collections.abc import Callable

import jax.numpy as jnp

from jorbit.data.constants import (
    IAS15_D_MATRIX,
    IAS15_H,
    IAS15_sub_cs,
    IAS15_sub_rs,
)
from jorbit.integrators.ias15 import (
    _estimate_x_v_from_b,
    _refine_sub_g,
    _update_bs,
    add_cs,
)
from jorbit.utils.states import IAS15IntegratorState, SystemState


@jax.jit
def ias15_static_step(
    initial_system_state: SystemState,
    acceleration_func: Callable[[SystemState], jnp.ndarray],
    initial_integrator_state: IAS15IntegratorState,
    fixed_perturber_positions: jnp.ndarray = jnp.empty((8, 0, 3)),
    fixed_perturber_velocities: jnp.ndarray = jnp.empty((8, 0, 3)),
    fixed_perturber_log_gms: jnp.ndarray = jnp.empty((0,)),
) -> SystemState:
    """Take a single step using the IAS15 integraton with no checks for convergence or adaptivity.

    Like ias15_step, but with a fixed number of predictor-corrector iterations, no
    checks for convergence, no checks for step size appropriateness, and no next step
    predictions. Right now hard-coded to 4 predictor-corrector iterations. Blindly
    trusts what it's given in exchange for no logic branches and better JIT compilation.

    Args:
        initial_system_state (SystemState):
            The initial system state.
        acceleration_func (Callable[[SystemState], jnp.ndarray]):
            The acceleration function.
        initial_integrator_state (IAS15IntegratorState):
            The initial integrator state.
        fixed_perturber_positions (jnp.ndarray):
            The fixed perturber positions for each substep. Shape
            (n_substeps, n_perturbers, 3). Default is empty array of shape (8, 0, 3).
        fixed_perturber_velocities (jnp.ndarray):
            Same as above, but for velocities.
        fixed_perturber_log_gms (jnp.ndarray):
            The log GM values for the fixed perturbers. Shape (n_perturbers,).
            Default is empty array of shape (0,).

    Returns:
        tuple[SystemState, IAS15IntegratorState]:
            The new system state and new integrator state.
    """
    n_pc_iterations = 3
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

    dt = initial_integrator_state.dt
    a0 = initial_integrator_state.a0
    csx = initial_integrator_state.csx
    csv = initial_integrator_state.csv
    e = initial_integrator_state.e
    b = initial_integrator_state.b

    csb = jnp.zeros_like(b)
    g = jnp.einsum("ij,jnk->ink", IAS15_D_MATRIX, b)

    def _predictor_corrector_iteration(
        b: jnp.ndarray,
        csb: jnp.ndarray,
        g: jnp.ndarray,
    ) -> tuple:
        # leaving in for loops instead of scans since they give simpler jaxprs, but
        # should actually go benchmark both ways
        for n, h, c, r in zip(range(1, 8), IAS15_H[1:], IAS15_sub_cs, IAS15_sub_rs):
            step_time = t_beginning + dt * h
            x, v = _estimate_x_v_from_b(
                a0=a0,
                v0=v0,
                x0=x0,
                h=h,
                dt=dt,
                bp=b[::-1],
            )
            acc_state = SystemState(
                massive_positions=x[:M],
                massive_velocities=v[:M],
                tracer_positions=x[M:],
                tracer_velocities=v[M:],
                log_gms=initial_system_state.log_gms,
                time=step_time,
                fixed_perturber_positions=fixed_perturber_positions[n],
                fixed_perturber_velocities=fixed_perturber_velocities[n],
                fixed_perturber_log_gms=fixed_perturber_log_gms,
                acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
            )
            at = acceleration_func(acc_state)
            g_old = g[n - 1]
            g_new = _refine_sub_g(at, a0, g[: n - 1], r)
            g_diff = g_new - g_old
            new_bs, new_csbs = _update_bs(b[:n], csb[:n], g_diff, c)
            g = g.at[n - 1].set(g_new)
            b = b.at[:n].set(new_bs)
            csb = csb.at[:n].set(new_csbs)

        return b, csb, g

    for _ in range(n_pc_iterations):
        (
            b,
            csb,
            g,
        ) = _predictor_corrector_iteration(b, csb, g)

    dt_done = dt
    x0, csx = add_cs(x0, csx, b[6] / 72.0 * dt_done * dt_done)
    x0, csx = add_cs(x0, csx, b[5] / 56.0 * dt_done * dt_done)
    x0, csx = add_cs(x0, csx, b[4] / 42.0 * dt_done * dt_done)
    x0, csx = add_cs(x0, csx, b[3] / 30.0 * dt_done * dt_done)
    x0, csx = add_cs(x0, csx, b[2] / 20.0 * dt_done * dt_done)
    x0, csx = add_cs(x0, csx, b[1] / 12.0 * dt_done * dt_done)
    x0, csx = add_cs(x0, csx, b[0] / 6.0 * dt_done * dt_done)
    x0, csx = add_cs(x0, csx, a0 / 2.0 * dt_done * dt_done)
    x0, csx = add_cs(x0, csx, v0 * dt_done)
    v0, csv = add_cs(v0, csv, b[6] / 8.0 * dt_done)
    v0, csv = add_cs(v0, csv, b[5] / 7.0 * dt_done)
    v0, csv = add_cs(v0, csv, b[4] / 6.0 * dt_done)
    v0, csv = add_cs(v0, csv, b[3] / 5.0 * dt_done)
    v0, csv = add_cs(v0, csv, b[2] / 4.0 * dt_done)
    v0, csv = add_cs(v0, csv, b[1] / 3.0 * dt_done)
    v0, csv = add_cs(v0, csv, b[0] / 2.0 * dt_done)
    v0, csv = add_cs(v0, csv, a0 * dt_done)

    # Zeroing out the perturber positions so that hopefully it's obvious something
    # has gone wrong if these are reused for the next time step. These are only valid
    # for the time given at the start of the step
    new_system_state = SystemState(
        massive_positions=x0[:M],
        massive_velocities=v0[:M],
        tracer_positions=x0[M:],
        tracer_velocities=v0[M:],
        log_gms=initial_system_state.log_gms,
        time=t_beginning + dt_done,
        fixed_perturber_positions=fixed_perturber_positions[-1],
        fixed_perturber_velocities=fixed_perturber_velocities[-1],
        fixed_perturber_log_gms=fixed_perturber_log_gms,
        acceleration_func_kwargs=initial_system_state.acceleration_func_kwargs,
    )

    new_integrator_state = IAS15IntegratorState(
        g=g,
        b=b,
        e=e,
        csx=csx,
        csv=csv,
        a0=acceleration_func(new_system_state),
        dt=dt_done,
        dt_last_done=dt_done,
    )

    return new_system_state, new_integrator_state


@jax.jit
def ias15_static_evolve(
    initial_system_state: SystemState,
    acceleration_func: Callable[[SystemState], jnp.ndarray],
    dts: jnp.ndarray,
    perturber_positions: jnp.ndarray,
    perturber_velocities: jnp.ndarray,
    perturber_log_gms: jnp.ndarray,
    initial_integrator_state: IAS15IntegratorState,
) -> tuple[jnp.ndarray, jnp.ndarray, SystemState, IAS15IntegratorState]:
    """Take multiple steps using ias15_static_step to evolve without safety checks.

    This blindly evolves the system using the provided time steps, without any checks
    for convergence within each step or appropriateness of each step size.

    Args:
        initial_system_state (SystemState):
            The initial state of the system.
        acceleration_func (Callable[[SystemState], jnp.ndarray]):
            The acceleration function to use.
        dts (jnp.ndarray):
            The time steps to evolve the system by.
        initial_integrator_state (IAS15IntegratorState):
            The initial state of the integrator.
        perturber_positions (jnp.ndarray):
            The positions of the perturbers. Shape (n_dts, 8, n_perturbers, 3).
        perturber_velocities (jnp.ndarray):
            The velocities of the perturbers. Same shape as perturber_positions.
        perturber_log_gms (jnp.ndarray):
            The log gravitational parameters of the perturbers. Shape (n_perturbers,).

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, SystemState, IAS15IntegratorState]:
            The positions and velocities of the system at each timestep,
            the final state of the system, and the final state of the integrator.
    """

    def scan_func(carry: tuple, scan_over: float) -> tuple:
        system_state, integrator_state = carry
        dt, perturber_positions, perturber_velocities = scan_over
        integrator_state.dt = dt
        new_system_state, new_integrator_state = ias15_static_step(
            system_state,
            acceleration_func,
            integrator_state,
            perturber_positions,
            perturber_velocities,
            perturber_log_gms,
        )
        return (new_system_state, new_integrator_state), (
            jnp.concatenate(
                (
                    new_system_state.massive_positions,
                    new_system_state.tracer_positions,
                )
            ),
            jnp.concatenate(
                (
                    new_system_state.massive_velocities,
                    new_system_state.tracer_velocities,
                )
            ),
        )

    initial_system_state.fixed_perturber_log_gms = perturber_log_gms

    (final_system_state, final_integrator_state), (positions, velocities) = (
        jax.lax.scan(
            scan_func,
            (initial_system_state, initial_integrator_state),
            (dts, perturber_positions, perturber_velocities),
        )
    )
    return positions, velocities, final_system_state, final_integrator_state

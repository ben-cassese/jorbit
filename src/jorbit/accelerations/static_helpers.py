"""Helper functions for pre-computing static perturber positions for use in accelerations."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from astropy.time import Time

from jorbit.data.constants import IAS15_H
from jorbit.ephemeris.ephemeris import Ephemeris
from jorbit.integrators import ias15_step
from jorbit.utils.states import IAS15IntegratorState


def precompute_perturber_positions(
    t0: Time, dts: jnp.ndarray, de_ephemeris_version: str = "440"
) -> tuple:
    """Precompute the states of perturbers for use in repeated integrations.

    Args:
        t0 (Time): Initial time to start an integration
        dts (jnp.ndarray):
            Array of all step sizes dt used in the integration, including inserted
            intermediate steps.
        de_ephemeris_version (str, optional):
            Version of the JPL DE ephemeris to use. Defaults to "440", accepts "430"

    Returns:
        tuple:
            planet_pos (jnp.ndarray):
                Array of shape (N_steps, N_substeps (8), 11, 3) containing the positions
                of the 11 major solar system bodies at each step and substep.
            planet_vel (jnp.ndarray):
                Like planet_pos, but for velocities.
            asteroid_pos (jnp.ndarray):
                Array of shape (N_steps, N_substeps (8), N_asteroids (16), 3) containing
                the positions of the asteroid perturbers at each step and substep.
            asteroid_vel (jnp.ndarray):
                Like asteroid_pos, but for velocities.
    """
    # this whole function could absolutely be done without loops/appends, but it only
    # runs once so I don't really care

    t0 = t0.tdb.jd
    all_times = jnp.cumsum(jnp.concatenate([jnp.array([0.0]), dts])) + t0

    eph = Ephemeris(
        ssos="default solar system",
        earliest_time=Time(min(all_times), format="jd", scale="tdb"),
        latest_time=Time(max(all_times), format="jd", scale="tdb"),
        de_ephemeris_version=de_ephemeris_version,
    )

    planet_pos = []
    planet_vel = []
    asteroid_pos = []
    asteroid_vel = []
    for i in range(len(all_times) - 1):
        step_size = all_times[i + 1] - all_times[i]
        subtimes = all_times[i] + step_size * jnp.concatenate(
            [IAS15_H, jnp.array([1.0])]
        )
        perturber_states = jax.vmap(eph.processor.state)(subtimes)
        planet_x = perturber_states[0][:, :11, :]
        planet_v = perturber_states[1][:, :11, :]
        asteroid_x = perturber_states[0][:, 11:, :]
        asteroid_v = perturber_states[1][:, 11:, :]
        planet_pos.append(planet_x)
        planet_vel.append(planet_v)
        asteroid_pos.append(asteroid_x)
        asteroid_vel.append(asteroid_v)

    planet_pos = jnp.array(planet_pos)
    planet_vel = jnp.array(planet_vel)
    asteroid_pos = jnp.array(asteroid_pos)
    asteroid_vel = jnp.array(asteroid_vel)

    gms = eph.processor.log_gms

    return (
        planet_pos,
        planet_vel,
        asteroid_pos,
        asteroid_vel,
        gms,
    )


def get_dynamic_intermediate_dts(
    initial_system_state: IAS15IntegratorState,
    acceleration_func: jax.tree_util.Partial,
    final_time: float,
    initial_integrator_state: IAS15IntegratorState,
) -> jnp.ndarray:
    """Use the adaptive IAS15 stepper to compute the required steps between two times.

    Args:
        initial_system_state (IAS15IntegratorState):
            The initial state of the system at the start of the integration. Contains
            the initial time.
        acceleration_func (jax.tree_util.Partial):
            The acceleration function to use for the integration.
        final_time (float):
            The final time to integrate to.
        initial_integrator_state (IAS15IntegratorState):
            The initial state of the integrator.

    Returns:
        jnp.ndarray:
            Array of the intermediate step sizes that would be taken by the IAS15
            integrator between the initial time and final time.
    """

    def step_needed(args: tuple) -> tuple:
        system_state, integrator_state, last_meaningful_dt, iter_num = args

        t = system_state.time

        diff = final_time - t
        step_length = jnp.sign(diff) * jnp.min(
            jnp.array([jnp.abs(diff), jnp.abs(integrator_state.dt)])
        )

        integrator_state.dt = step_length

        system_state, integrator_state = ias15_step(
            system_state, acceleration_func, integrator_state
        )
        return system_state, integrator_state, last_meaningful_dt, iter_num + 1

    def cond_func(args: tuple) -> bool:
        system_state, integrator_state, _last_meaningful_dt, iter_num = args
        t = system_state.time

        step_length = jnp.sign(final_time - t) * jnp.min(
            jnp.array([jnp.abs(final_time - t), jnp.abs(integrator_state.dt)])
        )
        return (step_length != 0) & (iter_num < 10_000)

    args = (
        initial_system_state,
        initial_integrator_state,
        initial_integrator_state.dt,
        0,
    )
    dts = []
    while cond_func(args):
        args = step_needed(args)
        if args[1].dt_last_done != 0:
            dts.append(args[1].dt_last_done)

    if len(dts) == 0:
        dts = jnp.array([0.0])
    return jnp.array(dts), len(dts), args[0], args[1]


def get_all_dynamic_intermediate_dts(
    initial_system_state: IAS15IntegratorState,
    acceleration_func: jax.tree_util.Partial,
    times: Time,
    initial_integrator_state: IAS15IntegratorState,
) -> jnp.ndarray:
    """Get all intermediate step sizes needed to integrate over a series of times.

    Just a wrapper around get_dynamic_intermediate_dts to handle multiple times.

    Args:
        initial_system_state (IAS15IntegratorState):
            The initial state of the system at the start of the integration. Contains
            the initial time.
        acceleration_func (jax.tree_util.Partial):
            The acceleration function to use for the integration.
        times (Time):
            The final times to integrate to.
        initial_integrator_state (IAS15IntegratorState):
            The initial state of the integrator.

    Returns:
        tuple:
            all_dts (jnp.ndarray):
                Array of all intermediate step sizes that would be taken by the IAS15
                integrator between the initial time and each of the final times. The
                array is flattened, 1D.
            obs_indices (jnp.ndarray):
                Array of indices into all_dts that correspond to the original
                observation times.
    """
    times = times.tdb.jd
    if times.shape == ():
        times = jnp.array([times])

    all_dts = []
    obs_inds = []
    system_state = initial_system_state
    integrator_state = initial_integrator_state

    for final_time in times:
        dts, num_steps, system_state, integrator_state = get_dynamic_intermediate_dts(
            system_state,
            acceleration_func,
            final_time,
            integrator_state,
        )
        all_dts.append(dts)
        obs_inds.append(num_steps)

    return jnp.concatenate(all_dts), jnp.cumsum(jnp.array(obs_inds)) - 1


def get_fixed_intermediate_dts(t0: Time, times: Time, max_step_size: float) -> tuple:
    """Generate fixed intermediate step times between observation times.

    Given a set of observation times, generate a set of step times that includes
    intermediate steps such that no step is larger than max_step_size.

    Args:
        t0 (Time):
            Initial time to start an integration
        times (Time):
            Times at which observations are made
        max_step_size (float):
            Maximum step size to use during an integration. This function will insert
            intermediate steps between the observation times as needed to keep the step
            size below this value.

    Returns:
        tuple:
            all_dts (jnp.ndarray):
                Array of all step sizes dt used in the integration, including inserted
                intermediate steps.
            obs_indices (jnp.ndarray):
                Array of indices into all_dts that correspond to the original
                observation times.
    """
    times = times.tdb.jd
    if times.shape == ():
        times = jnp.array([times])
    t0 = t0.tdb.jd

    times = jnp.concatenate([jnp.array([t0]), times])
    diffs = jnp.diff(times)

    obs_indices = []
    all_dts = []
    for i in range(len(diffs)):
        n_steps = int(jnp.ceil(diffs[i] / max_step_size))
        step_size = diffs[i] / n_steps
        if n_steps == 0:
            n_steps = 1
            step_size = 0.0
        all_dts.extend([step_size] * n_steps)
        obs_indices.append(len(all_dts))

    obs_indices = jnp.array(obs_indices)

    return jnp.array(all_dts), obs_indices - 1

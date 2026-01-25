"""Helper functions for pre-computing static perturber positions for use in accelerations."""

import jax

jax.config.update("jax_enable_x64", True)
import astropy.units as u
import jax.numpy as jnp
from astropy.time import Time

from jorbit.data.constants import IAS15_H
from jorbit.ephemeris.ephemeris import Ephemeris


def precompute_perturber_positions(
    t0: Time, times: Time, max_step_size: u.Quantity, de_ephemeris_version: str = "440"
) -> tuple:
    """Precompute the states of perturbers for use in repeated integrations.

    Args:
        t0 (Time): Initial time to start an integration
        times (Time): Times at which observations are made
        max_step_size (u.Quantity):
            Maximum step size to use during an integration. This function will insert
            intermediate steps between the observation times as needed to keep the step
            size below this value.
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
            all_times (jnp.ndarray):
                Array of all step times dt used in the integration, including inserted
                intermediate steps. Does not include substeps.
            obs_indices (jnp.ndarray):
                Array of indices into all_times that correspond to the original
                observation
    """
    # this whole function could absolutely be done without loops/appends, but it only
    # runs once so I don't really care
    times = times.tdb.jd
    t0 = t0.tdb.jd

    times = jnp.concatenate([jnp.array([t0]), times])

    diffs = jnp.diff(times)

    all_times = []
    obs_indices = []
    current_index = 0

    obs_indices.append(current_index)
    for i in range(len(diffs)):
        n_steps = int(jnp.ceil(diffs[i] / max_step_size))
        step_size = diffs[i] / n_steps

        for j in range(n_steps):
            all_times.append(times[i] + j * step_size)
            current_index += 1
        obs_indices.append(current_index)

    all_times.append(times[-1])

    all_times = jnp.array(all_times)
    obs_indices = jnp.array(obs_indices)

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
        subtimes = all_times[i] + step_size * IAS15_H
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

    return planet_pos, planet_vel, asteroid_pos, asteroid_vel, all_times, obs_indices

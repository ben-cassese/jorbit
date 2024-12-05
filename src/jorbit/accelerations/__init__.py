import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.utils.states import SystemState
from jorbit.accelerations.newtonian import newtonian_gravity


def create_newtonian_ephemeris_acceleration_func(ephem_processor):

    def func(inputs: SystemState) -> jnp.ndarray:
        perturber_xs, _, _ = ephem_processor.state(inputs.time)
        perturber_log_gms = ephem_processor.log_gms

        new_state = SystemState(
            positions=jnp.concatenate([inputs.positions, perturber_xs]),
            velocities=inputs.velocities,
            log_gms=jnp.concatenate([inputs.log_gms, perturber_log_gms]),
            time=inputs.time,
            acceleration_func_kwargs=inputs.acceleration_func_kwargs,
        )

        accs = newtonian_gravity(new_state)

        num_particles = inputs.positions.shape[0]
        return accs[:num_particles]

    return jax.tree_util.Partial(func)

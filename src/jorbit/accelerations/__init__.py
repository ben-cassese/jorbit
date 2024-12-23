import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.utils.states import SystemState
from jorbit.accelerations.newtonian import newtonian_gravity
from jorbit.accelerations.gr import ppn_gravity


def create_newtonian_ephemeris_acceleration_func(ephem_processor):

    def func(inputs: SystemState) -> jnp.ndarray:
        perturber_xs, _, _ = ephem_processor.state(inputs.time)
        perturber_log_gms = ephem_processor.log_gms

        new_state = SystemState(
            massive_positions=jnp.concatenate([perturber_xs, inputs.massive_positions]),
            massive_velocities=jnp.concatenate(
                [jnp.zeros_like(perturber_xs), inputs.massive_velocities]
            ),
            tracer_positions=inputs.tracer_positions,
            tracer_velocities=inputs.tracer_velocities,
            log_gms=jnp.concatenate([perturber_log_gms, inputs.log_gms]),
            time=inputs.time,
            acceleration_func_kwargs=inputs.acceleration_func_kwargs,
        )

        accs = newtonian_gravity(new_state)

        num_perturbers = perturber_xs.shape[0]
        return accs[num_perturbers:]

    return jax.tree_util.Partial(func)


def create_gr_ephemeris_acceleration_func(ephem_processor):

    def func(inputs: SystemState) -> jnp.ndarray:
        perturber_xs, perturber_vs, _ = ephem_processor.state(inputs.time)
        perturber_log_gms = ephem_processor.log_gms

        new_state = SystemState(
            massive_positions=jnp.concatenate([perturber_xs, inputs.massive_positions]),
            massive_velocities=jnp.concatenate(
                [perturber_vs, inputs.massive_velocities]
            ),
            tracer_positions=inputs.tracer_positions,
            tracer_velocities=inputs.tracer_velocities,
            log_gms=jnp.concatenate([perturber_log_gms, inputs.log_gms]),
            time=inputs.time,
            acceleration_func_kwargs=inputs.acceleration_func_kwargs,
        )

        accs = ppn_gravity(new_state)

        num_perturbers = perturber_xs.shape[0]
        return accs[num_perturbers:]

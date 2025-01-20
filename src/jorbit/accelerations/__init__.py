import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.accelerations.gr import ppn_gravity
from jorbit.accelerations.newtonian import newtonian_gravity
from jorbit.utils.states import SystemState


def create_newtonian_ephemeris_acceleration_func(ephem_processor):

    def func(inputs: SystemState) -> jnp.ndarray:
        perturber_xs, perturber_vs = ephem_processor.state(inputs.time)
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

        accs = newtonian_gravity(new_state)

        num_perturbers = perturber_xs.shape[0]
        return accs[num_perturbers:]

    return jax.tree_util.Partial(func)


def create_gr_ephemeris_acceleration_func(ephem_processor):

    def func(inputs: SystemState) -> jnp.ndarray:
        perturber_xs, perturber_vs = ephem_processor.state(inputs.time)
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

    return jax.tree_util.Partial(func)


def create_default_ephemeris_acceleration_func(ephem_processor):

    def func(inputs: SystemState) -> jnp.ndarray:
        num_gr_perturbers = 10  # the "planets"
        num_newtonian_perturbers = 16  # the asteroids

        perturber_xs, perturber_vs = ephem_processor.state(inputs.time)
        perturber_log_gms = ephem_processor.log_gms

        gr_state = SystemState(
            massive_positions=jnp.concatenate(
                [perturber_xs[:num_gr_perturbers], inputs.massive_positions]
            ),
            massive_velocities=jnp.concatenate(
                [perturber_vs[:num_gr_perturbers], inputs.massive_velocities]
            ),
            tracer_positions=inputs.tracer_positions,
            tracer_velocities=inputs.tracer_velocities,
            log_gms=jnp.concatenate(
                [perturber_log_gms[:num_gr_perturbers], inputs.log_gms]
            ),
            time=inputs.time,
            acceleration_func_kwargs=inputs.acceleration_func_kwargs,
        )
        gr_acc = ppn_gravity(gr_state)[num_gr_perturbers:]

        newtonian_state = SystemState(
            massive_positions=jnp.concatenate(
                [perturber_xs[num_gr_perturbers:], inputs.massive_positions]
            ),
            massive_velocities=jnp.concatenate(
                [perturber_vs[num_gr_perturbers:], inputs.massive_velocities]
            ),
            tracer_positions=inputs.tracer_positions,
            tracer_velocities=inputs.tracer_velocities,
            log_gms=jnp.concatenate(
                [perturber_log_gms[num_gr_perturbers:], inputs.log_gms]
            ),
            time=inputs.time,
            acceleration_func_kwargs=inputs.acceleration_func_kwargs,
        )
        newtonian_acc = newtonian_gravity(newtonian_state)[num_newtonian_perturbers:]

        return gr_acc + newtonian_acc

    return jax.tree_util.Partial(func)

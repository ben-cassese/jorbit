import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.data.constants import IAS15_INIT
from jorbit.engine.ias15_integrator import ias15_integrate_multiple
from jorbit.engine.accelerations import acceleration_at_time
from jorbit.particle import Particle
from jorbit.system.base_system import BaseSystem


class IAS15System(BaseSystem):
    def __init__(self, particles, **kwargs):
        super().__init__(particles, **kwargs)

        self._particles, self._epoch = IAS15System._initialize_particles(
            particles=self._particles,
            infer_epoch=self._infer_epoch,
            verbose=self._verbose,
            planet_params=self._planet_params,
            asteroid_params=self._asteroid_params,
            planet_gms=self._planet_gms,
            asteroid_gms=self._asteroid_gms,
        )
        (
            self._tracer_xs,
            self._tracer_vs,
            self._massive_xs,
            self._massive_vs,
            self._massive_gms,
            self._free_tracer_x_inds,
            self._free_tracer_v_inds,
            self._free_massive_x_inds,
            self._free_massive_v_inds,
            self._free_massive_gm_inds,
            self._free_planet_gm_inds,
            self._free_asteroid_gm_inds,
        ) = IAS15System._setup_free_params(
            particles=self._particles,
            fit_planet_gms=self._fit_planet_gms,
            fit_asteroid_gms=self._fit_asteroid_gms,
        )

        (
            self._leading_tracer_ra,
            self._leading_tracer_dec,
            self._leading_tracer_uncertainty,
            self._leading_tracer_times,
            self._leading_tracer_observer_positions,
            self._trailing_tracer_ra,
            self._trailing_tracer_dec,
            self._trailing_tracer_uncertainty,
            self._trailing_tracer_times,
            self._trailing_tracer_observer_positions,
            self._leading_massive_ra,
            self._leading_massive_dec,
            self._leading_massive_uncertainty,
            self._leading_massive_times,
            self._leading_massive_observer_positions,
            self._trailing_massive_ra,
            self._trailing_massive_dec,
            self._trailing_massive_uncertainty,
            self._trailing_massive_times,
            self._trailing_massive_observer_positions,
        ) = IAS15System._setup_observations(
            particles=self._particles, epoch=self._epoch
        )

    ####################################################################################
    @staticmethod
    def _initialize_particles(
        particles,
        infer_epoch,
        verbose,
        planet_params,
        asteroid_params,
        planet_gms,
        asteroid_gms,
    ):
        if infer_epoch:
            # find the mean epoch of all observations, weighted by the precision of those
            # observations. There might be a better way to do this.
            # Also, right now I'm forcing the whole system to have a common epoch. In theory
            # the tracer particles could all have their own optimial epochs
            times = []
            weights = []
            for p in particles:
                if p.observations != None:
                    times.append(p.observations.times)
                    weights.append(1 / p.observations.astrometric_uncertainties**2)
            if len(times) > 0:
                times = jnp.concatenate(times)
                weights = jnp.concatenate(weights)
                epoch = jnp.sum(times * weights) / jnp.sum(weights)

                # TODO remove this, just testing issues with trailing observations
                # epoch = jnp.min(times)
                # epoch = jnp.min(times) - 1e-2
                # epoch = jnp.max(times) + 1e-2

            else:
                epoch = particles[0].time

        # _input_checks already verified all particles have same epoch
        else:
            epoch = particles[0].time

        # move each particles to the mean epoch. To do this, we neglect the influence of
        # massive particles and evolve each particle only under the influence of fixed
        # solar system perturbers
        modified_particles = []
        obs_mask = []
        if verbose and infer_epoch:
            print("Moving particles to common epoch")
        j = jax.jit(ias15_integrate_multiple)
        for i, p in enumerate(particles):
            if p.time != epoch:
                a0 = acceleration_at_time(
                    jnp.array([p.x]),
                    jnp.array([p.v]),
                    jnp.array([0.0]),
                    p.time,
                    planet_params,
                    asteroid_params,
                    planet_gms,
                    asteroid_gms,
                )
                x, v, t = j(
                    x0=jnp.array([p.x]),
                    v0=jnp.array([p.v]),
                    a0=a0,
                    acc=jax.tree_util.Partial(acceleration_at_time),
                    acc_fixed_kwargs={
                        "gm": jnp.array([0.0]),
                        "planet_params": planet_params,
                        "asteroid_params": asteroid_params,
                        "planet_gms": planet_gms,
                        "asteroid_gms": asteroid_gms,
                    },
                    acc_free_kwargs={},
                    t0=p.time,
                    tfs=jnp.array([epoch]),
                    **IAS15_INIT,
                )
                x = x[0, 0]
                v = v[0, 0]
            else:
                x = p.x
                v = p.v

            new_particle = Particle(
                x=x,
                v=v,
                elements=None,
                gm=p.gm,
                time=epoch,
                observations=p.observations,
                earliest_time=p.earliest_time,
                latest_time=p.latest_time,
                name=p.name,
                free_orbit=p.free_orbit,
                free_gm=p.free_gm,
            )
            modified_particles.append(new_particle)

        return modified_particles, epoch

    @staticmethod
    def _setup_free_params(particles, fit_planet_gms, fit_asteroid_gms):
        ntracers = 0
        nmassive = 0

        for p in particles:
            if p.gm == 0.0 and not p.free_gm:
                ntracers += 1
            else:
                nmassive += 1

        tracer_xs = jnp.ones((ntracers, 3)) * 999.0
        tracer_vs = jnp.ones((ntracers, 3)) * 999.0
        massive_xs = jnp.ones((nmassive, 3)) * 999.0
        massive_vs = jnp.ones((nmassive, 3)) * 999.0
        massive_gms = jnp.ones((nmassive)) * 999.0

        free_tracer_x_inds = []
        free_tracer_v_inds = []
        free_massive_x_inds = []
        free_massive_v_inds = []
        free_massive_gm_inds = []
        seen_tracers = 0
        seen_massive = 0
        for p in particles:
            if p.gm == 0.0 and not p.free_gm:
                tracer_xs = tracer_xs.at[seen_tracers].set(p.x)
                tracer_vs = tracer_vs.at[seen_tracers].set(p.v)
                if p.free_orbit:
                    free_tracer_x_inds.append(seen_tracers)
                    free_tracer_v_inds.append(seen_tracers)
                seen_tracers += 1
            else:
                massive_xs = massive_xs.at[seen_massive].set(p.x)
                massive_vs = massive_vs.at[seen_massive].set(p.v)
                massive_gms = massive_gms.at[seen_massive].set(p.gm)
                if p.free_orbit:
                    free_massive_x_inds.append(seen_massive)
                    free_massive_v_inds.append(seen_massive)
                if p.free_gm:
                    free_massive_gm_inds.append(seen_massive)
                seen_massive += 1

        free_planet_gm_inds = []
        for i, p in enumerate(fit_planet_gms):
            if p:
                free_planet_gm_inds.append(i)
        free_asteroid_gm_inds = []
        for i, p in enumerate(fit_asteroid_gms):
            if p:
                free_asteroid_gm_inds.append(i)

        return (
            tracer_xs,
            tracer_vs,
            massive_xs,
            massive_vs,
            massive_gms,
            jnp.array(free_tracer_x_inds),
            jnp.array(free_tracer_v_inds),
            jnp.array(free_massive_x_inds),
            jnp.array(free_massive_v_inds),
            jnp.array(free_massive_gm_inds),
            jnp.array(free_planet_gm_inds),
            jnp.array(free_asteroid_gm_inds),
        )

    @staticmethod
    def _setup_observations(particles, epoch):
        most_leading_obs = 0
        most_trailing_obs = 0
        ntracers = 0
        for p in particles:
            if p.gm == 0.0 and not p.free_gm:
                ntracers += 1
                if p.observations is not None:
                    leading = jnp.sum(p.observations.times > epoch)
                    most_leading_obs = max(most_leading_obs, leading)
                    trailing = jnp.sum(p.observations.times < epoch)
                    most_trailing_obs = max(most_trailing_obs, trailing)

        print(f"most leading obs: {most_leading_obs}")
        print(f"most trailing obs: {most_trailing_obs}")
        trailing_tracer_ra = jnp.zeros((ntracers, most_trailing_obs))
        trailing_tracer_dec = jnp.zeros((ntracers, most_trailing_obs))
        trailing_tracer_uncertainty = jnp.ones((ntracers, most_trailing_obs)) * 999.0
        trailing_tracer_times = jnp.ones((ntracers, most_trailing_obs)) * 999.0
        trailing_tracer_observer_positions = (
            jnp.ones((ntracers, most_trailing_obs, 3)) * 999.0
        )
        leading_tracer_ra = jnp.zeros((ntracers, most_leading_obs))
        leading_tracer_dec = jnp.zeros((ntracers, most_leading_obs))
        leading_tracer_uncertainty = jnp.ones((ntracers, most_leading_obs)) * 999.0
        leading_tracer_times = jnp.ones((ntracers, most_leading_obs)) * 999.0
        leading_tracer_observer_positions = (
            jnp.ones((ntracers, most_leading_obs, 3)) * 999.0
        )

        print(f"leading tracer ra shape: {leading_tracer_ra.shape}")
        i = 0
        for p in particles:
            if p.gm == 0.0 and not p.free_gm:
                if p.observations is not None:
                    leading = jnp.sum(p.observations.times > epoch)
                    trailing = jnp.sum(p.observations.times < epoch)
                    print(f"leading: {leading}")
                    print(f"trailing: {trailing}")
                    if leading > 0:
                        mask = p.observations.times > epoch
                        leading_tracer_ra = leading_tracer_ra.at[i, :leading].set(
                            p.observations.ra[mask]
                        )
                        leading_tracer_dec = leading_tracer_dec.at[i, :leading].set(
                            p.observations.dec[mask]
                        )
                        leading_tracer_uncertainty = leading_tracer_uncertainty.at[
                            i, :leading
                        ].set(p.observations.astrometric_uncertainties[mask])
                        leading_tracer_times = leading_tracer_times.at[i, :leading].set(
                            p.observations.times[mask]
                        )
                        leading_tracer_observer_positions = (
                            leading_tracer_observer_positions.at[i, :leading].set(
                                p.observations.observer_positions[mask]
                            )
                        )
                    if trailing > 0:
                        mask = p.observations.times < epoch
                        trailing_tracer_ra = trailing_tracer_ra.at[i, :trailing].set(
                            p.observations.ra[mask]
                        )
                        trailing_tracer_dec = trailing_tracer_dec.at[i, :trailing].set(
                            p.observations.dec[mask]
                        )
                        trailing_tracer_uncertainty = trailing_tracer_uncertainty.at[
                            i, :trailing
                        ].set(p.observations.astrometric_uncertainties[mask])
                        trailing_tracer_times = trailing_tracer_times.at[
                            i, :trailing
                        ].set(p.observations.times[mask])
                        trailing_tracer_observer_positions = (
                            trailing_tracer_observer_positions.at[i, :trailing].set(
                                p.observations.observer_positions[mask]
                            )
                        )
                    i += 1
        print(f"leading tracer ra shape: {leading_tracer_ra.shape}")

        most_leading_obs = 0
        most_trailing_obs = 0
        nmassive = 0
        for p in particles:
            if p.gm > 0.0 or p.free_gm:
                nmassive += 1
                if p.observations is not None:
                    leading = jnp.sum(p.observations.times > epoch)
                    most_leading_obs = max(most_leading_obs, leading)
                    trailing = jnp.sum(p.observations.times < epoch)
                    most_trailing_obs = max(most_trailing_obs, trailing)
        print(f"most leading obs: {most_leading_obs}")
        print(f"most trailing obs: {most_trailing_obs}")
        trailing_massive_ra = jnp.zeros((nmassive, most_trailing_obs))
        trailing_massive_dec = jnp.zeros((nmassive, most_trailing_obs))
        trailing_massive_uncertainty = jnp.ones((nmassive, most_trailing_obs)) * 999.0
        trailing_massive_times = jnp.ones((nmassive, most_trailing_obs)) * 999.0
        trailing_massive_observer_positions = (
            jnp.ones((nmassive, most_trailing_obs, 3)) * 999.0
        )
        leading_massive_ra = jnp.zeros((nmassive, most_leading_obs))
        leading_massive_dec = jnp.zeros((nmassive, most_leading_obs))
        leading_massive_uncertainty = jnp.ones((nmassive, most_leading_obs)) * 999.0
        leading_massive_times = jnp.ones((nmassive, most_leading_obs)) * 999.0
        leading_massive_observer_positions = (
            jnp.ones((nmassive, most_leading_obs, 3)) * 999.0
        )

        i = 0
        for p in particles:
            if p.gm > 0.0 or p.free_gm:
                if p.observations is not None:
                    leading = jnp.sum(p.observations.times > epoch)
                    trailing = jnp.sum(p.observations.times < epoch)
                    if leading > 0:
                        mask = p.observations.times > epoch
                        leading_massive_ra = leading_massive_ra.at[i, :trailing].set(
                            p.observations.ra[mask]
                        )
                        leading_massive_dec = leading_massive_dec.at[i, :trailing].set(
                            p.observations.dec[mask]
                        )
                        leading_massive_uncertainty = leading_massive_uncertainty.at[
                            i, :trailing
                        ].set(p.observations.astrometric_uncertainties[mask])
                        leading_massive_times = leading_massive_times.at[
                            i, :trailing
                        ].set(p.observations.times[mask])
                        leading_massive_observer_positions = (
                            leading_massive_observer_positions.at[i, :trailing].set(
                                p.observations.observer_positions[mask]
                            )
                        )
                    if trailing > 0:
                        mask = p.observations.times < epoch
                        trailing_massive_ra = trailing_massive_ra.at[i, :trailing].set(
                            p.observations.ra[mask]
                        )
                        trailing_massive_dec = trailing_massive_dec.at[
                            i, :trailing
                        ].set(p.observations.dec[mask])
                        trailing_massive_uncertainty = trailing_massive_uncertainty.at[
                            i, :trailing
                        ].set(p.observations.astrometric_uncertainties[mask])
                        trailing_massive_times = trailing_massive_times.at[
                            i, :trailing
                        ].set(p.observations.times[mask])
                        trailing_massive_observer_positions = (
                            trailing_massive_observer_positions.at[i, :trailing].set(
                                p.observations.observer_positions[mask]
                            )
                        )
                    i += 1

        return (
            leading_tracer_ra,
            leading_tracer_dec,
            leading_tracer_uncertainty,
            leading_tracer_times,
            leading_tracer_observer_positions,
            trailing_tracer_ra,
            trailing_tracer_dec,
            trailing_tracer_uncertainty,
            trailing_tracer_times,
            trailing_tracer_observer_positions,
            leading_massive_ra,
            leading_massive_dec,
            leading_massive_uncertainty,
            leading_massive_times,
            leading_massive_observer_positions,
            trailing_massive_ra,
            trailing_massive_dec,
            trailing_massive_uncertainty,
            trailing_massive_times,
            trailing_massive_observer_positions,
        )

    @property
    def particles(self):
        pass

    @property
    def residuals(self):
        pass

    def _generate_likelihood_funcs(self):
        pass

    def propagate(self, t):
        pass

    # def parse_1d(x,
    #     extras_sculpting_func,
    #     tracer_fx_rm__x=jnp.array([]),
    #     tracer_fx_rm__v=jnp.array([]),
    #     n_free_tracer_fx_rm=0,
    #     tracer_rx_rm__x=jnp.array([]),
    #     tracer_rx_rm__v=jnp.array([]),
    #     massive_fx_rm__x=jnp.array([]),
    #     massive_fx_rm__v=jnp.array([]),
    #     massive_fx_rm__gm=jnp.array([]),
    #     n_free_massive_fx_rm=0,
    #     massive_rx_fm__x=jnp.array([]),
    #     massive_rx_fm__v=jnp.array([]),
    #     massive_rx_fm__gm=jnp.array([]),
    #     n_free_massive_rx_fm=0,
    #     massive_fx_fm__x=jnp.array([]),
    #     massive_fx_fm__v=jnp.array([]),
    #     massive_fx_fm__gm=jnp.array([]),
    #     n_free_massive_fx_fm=0,
    #     massive_rx_rm__x=jnp.array([]),
    #     massive_rx_rm__v=jnp.array([]),
    #     massive_rx_rm__gm=jnp.array([]),
    #     planet_gms=jnp.array([]),
    #     n_free_planet_gms=0,
    #     asteroid_gms=jnp.array([]),
    #     n_free_asteroid_gms=0,
    # ):
    #     # x is a 1D array of free parameters, need to parse it into the right combination
    #     # of free and fixed parameters. It's laid out as: positions, velocities, GMs, planet_gms, asteroid_gms, then extras.

    #     # the appends here are a little misleading- ideally it'd be a conditional, but
    #     # that was annoying to implement in jax since jnp.where and jax.lax.cond both
    #     # need the output shapes of each branch to match

    #     n_tracers = tracer_fx_rm__x.shape[0] + n_free_tracer_fx_rm + tracer_rx_rm__x.shape[0]
    #     n_massive = massive_fx_rm__x.shape[0] + n_free_massive_fx_rm + \
    #                 massive_rx_fm__x.shape[0] + n_free_massive_rx_fm + \
    #                 massive_fx_fm__x.shape[0] + n_free_massive_fx_fm + \
    #                 massive_rx_rm__x.shape[0]

    #     # positions
    #     lb = 0
    #     ub = lb + n_free_tracer_fx_rm*3
    #     tracer_fx_rm__x = jnp.append(x[lb:ub], tracer_fx_rm__x)

    #     lb = ub
    #     ub = lb + n_free_massive_fx_rm*3
    #     massive_fx_rm__x = jnp.append(x[lb:ub], massive_fx_rm__x)

    #     lb = ub
    #     ub = lb + n_free_massive_fx_fm*3
    #     massive_fx_fm__x = jnp.append(x[lb:ub], massive_fx_fm__x)

    #     # velocities
    #     lb = ub
    #     ub = lb + n_free_tracer_fx_rm*3
    #     tracer_fx_rm__v = jnp.append(x[lb:ub], tracer_fx_rm__v)

    #     lb = ub
    #     ub = lb + n_free_massive_fx_rm*3
    #     massive_fx_rm__v = jnp.append(x[lb:ub], massive_fx_rm__v)

    #     lb = ub
    #     ub = lb + n_free_massive_fx_fm*3
    #     massive_fx_fm__v = jnp.append(x[lb:ub], massive_fx_fm__v)

    #     # GMs
    #     lb = ub
    #     ub = lb + n_free_massive_rx_fm
    #     massive_rx_fm__gm = jnp.append(x[lb:ub], massive_rx_fm__gm)

    #     lb = ub
    #     ub = lb + n_free_massive_fx_fm
    #     massive_fx_fm__gm = jnp.append(x[lb:ub], massive_fx_fm__gm)

    #     # planet GMs
    #     lb = ub
    #     ub = lb + n_free_planet_gms
    #     planet_gms = jnp.append(x[lb:ub], planet_gms)

    #     # asteroid GMs
    #     lb = ub
    #     ub = lb + n_free_asteroid_gms
    #     asteroid_gms = jnp.append(x[lb:ub], asteroid_gms)

    #     # extras
    #     lb = ub
    #     extras = extras_sculpting_func(x[lb:])

    #     # # combine everything:
    #     tracer_x0 = jnp.concatenate([tracer_fx_rm__x, tracer_rx_rm__x.flatten()])
    #     tracer_v0 = jnp.concatenate([tracer_fx_rm__v, tracer_rx_rm__v.flatten()])

    #     massive_x0 = jnp.concatenate([massive_fx_rm__x, massive_rx_fm__x.flatten(), massive_fx_fm__x, massive_rx_rm__x.flatten()])
    #     massive_v0 = jnp.concatenate([massive_fx_rm__v, massive_rx_fm__v.flatten(), massive_fx_fm__v, massive_rx_rm__v.flatten()])
    #     massive_gms = jnp.concatenate([massive_fx_rm__gm, massive_rx_fm__gm, massive_fx_fm__gm, massive_rx_rm__gm.flatten()])

    #     return tracer_x0.reshape(-1,3), tracer_v0.reshape(-1,3), \
    #             massive_x0.reshape(-1,3), massive_v0.reshape(-1,3), massive_gms, planet_gms, asteroid_gms, extras

    # def free_dict_to_vec(free_params):
    #     x = jnp.array([])
    #     if 'tracer_fx_rm__x' in free_params:
    #         x = jnp.append(x, free_params['tracer_fx_rm__x'])
    #     if 'tracer_fx_rm__v' in free_params:
    #         x = jnp.append(x, free_params['tracer_fx_rm__v'])

    #     if 'massive_fx_rm__x' in free_params:
    #         x = jnp.append(x, free_params['massive_fx_rm__x'])
    #     if 'massive_fx_rm__v' in free_params:
    #         x = jnp.append(x, free_params['massive_fx_rm__v'])

    #     if 'massive_rx_fm__gm' in free_params:
    #         x = jnp.append(x, free_params['massive_rx_fm__gm'])

    #     if 'massive_fx_fm__x' in free_params:
    #         x = jnp.append(x, free_params['massive_fx_fm__x'])
    #     if 'massive_fx_fm__v' in free_params:
    #         x = jnp.append(x, free_params['massive_fx_fm__v'])
    #     if 'massive_fx_fm__gm' in free_params:
    #         x = jnp.append(x, free_params['massive_fx_fm__gm'])

    #     if 'planet_gms' in free_params:
    #         x = jnp.append(x, free_params['planet_gms'])

    #     if 'asteroid_gms' in free_params:
    #         x = jnp.append(x, free_params['asteroid_gms'])

    #     return x

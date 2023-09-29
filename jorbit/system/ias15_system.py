import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.engine.ias15_integrator import (
    ias15_integrate_multiple,
    ias15_initial_params,
)
from jorbit.engine.ias15_likelihood import ias15_likelihood
from jorbit.engine import pad_to_parallelize
from jorbit.engine.accelerations import acceleration_at_time
from jorbit.engine.ephemeris import perturber_positions
from jorbit.particle import Particle
from jorbit.system.base_system import BaseSystem


class IAS15System(BaseSystem):
    def __init__(self, particles, **kwargs):
        #################
        # initializations
        #################
        super().__init__(particles, **kwargs)

        if self._verbose:
            print("Initializing particles...")
        self._particles, self._epoch = IAS15System._initialize_particles(
            particles=self._particles,
            infer_epoch=self._infer_epoch,
            verbose=self._verbose,
            planet_params=self._planet_params,
            asteroid_params=self._asteroid_params,
            planet_gms=self._planet_gms,
            asteroid_gms=self._asteroid_gms,
        )

        if self._verbose:
            print("Sorting free/fixed parameters...")
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
            self._tracer_key,
            self._massive_key,
        ) = IAS15System._setup_free_params(
            particles=self._particles,
            fit_planet_gms=self._fit_planet_gms,
            fit_asteroid_gms=self._fit_asteroid_gms,
        )

        if self._verbose:
            print("Caching observations metadata...")
        (
            self._leading_tracer_ra,
            self._leading_tracer_dec,
            self._leading_tracer_astrometric_uncertainties,
            self._leading_tracer_times,
            self._leading_tracer_observer_positions,
            self._leading_tracer_planet_xs_at_obs,
            self._leading_tracer_asteroid_xs_at_obs,
            self._trailing_tracer_ra,
            self._trailing_tracer_dec,
            self._trailing_tracer_astrometric_uncertainties,
            self._trailing_tracer_times,
            self._trailing_tracer_observer_positions,
            self._trailing_tracer_planet_xs_at_obs,
            self._trailing_tracer_asteroid_xs_at_obs,
            self._leading_massive_ra,
            self._leading_massive_dec,
            self._leading_massive_astrometric_uncertainties,
            self._leading_massive_times,
            self._leading_massive_observer_positions,
            self._leading_massive_planet_xs_at_obs,
            self._leading_massive_asteroid_xs_at_obs,
            self._trailing_massive_ra,
            self._trailing_massive_dec,
            self._trailing_massive_astrometric_uncertainties,
            self._trailing_massive_times,
            self._trailing_massive_observer_positions,
            self._trailing_massive_planet_xs_at_obs,
            self._trailing_massive_asteroid_xs_at_obs,
        ) = IAS15System._setup_observations(
            particles=self._particles,
            epoch=self._epoch,
            planet_params=self._planet_params,
            asteroid_params=self._asteroid_params,
        )

        ##################
        # likelihood funcs
        ##################
        # self._extras = jnp.array([], dtype=jnp.int64)
        # self._free_extras_inds = jnp.array([], dtype=jnp.int64)

        self._x = IAS15System.generate_initial_free_params(
            tracer_xs=self._tracer_xs,
            tracer_vs=self._tracer_vs,
            massive_xs=self._massive_xs,
            massive_vs=self._massive_vs,
            massive_gms=self._massive_gms,
            planet_gms=self._planet_gms,
            asteroid_gms=self._asteroid_gms,
            free_tracer_x_inds=self._free_tracer_x_inds,
            free_tracer_v_inds=self._free_tracer_v_inds,
            free_massive_x_inds=self._free_massive_x_inds,
            free_massive_v_inds=self._free_massive_v_inds,
            free_massive_gm_inds=self._free_massive_gm_inds,
            free_planet_gm_inds=self._free_planet_gm_inds,
            free_asteroid_gm_inds=self._free_asteroid_gm_inds,
        )

        if self._verbose:
            print("Compiling likelihood functions...")
        self._full_resids, self.loglike = self._generate_likelihood_funcs()
        _ = self._full_resids(self._x)
        _ = self.loglike(self._x)

        if self._verbose:
            print(
                "Compiling gradient of likelihood functions (longest and last step)..."
            )
        # _ = jax.jacfwd(self.loglike)(self._x)
        _ = jax.grad(self.loglike)(self._x)

        if self._verbose:
            print("System initialized!")
            print(self.__repr__())

    ####################################################################################
    # initialization methods
    ####################################################################################
    @staticmethod  # there's not really a reason to make these static, I'm just in the
    # jax headspace of making sure nothing has side effects
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
            print("Moving particles to common epoch...")
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
                    acc=jax.tree_util.Partial(
                        acceleration_at_time,
                        **{
                            "gm": jnp.array([0.0]),
                            "planet_params": planet_params,
                            "asteroid_params": asteroid_params,
                            "planet_gms": planet_gms,
                            "asteroid_gms": asteroid_gms,
                        },
                    ),
                    acc_free_kwargs={},
                    t0=p.time,
                    tfs=jnp.array([epoch]),
                    **ias15_initial_params(1),
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
        tracer_key = []
        massive_key = []
        for i, p in enumerate(particles):
            if p.gm == 0.0 and not p.free_gm:
                tracer_xs = tracer_xs.at[seen_tracers].set(p.x)
                tracer_vs = tracer_vs.at[seen_tracers].set(p.v)
                if p.free_orbit:
                    free_tracer_x_inds.append(seen_tracers)
                    free_tracer_v_inds.append(seen_tracers)
                seen_tracers += 1
                tracer_key.append(i)
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
                massive_key.append(i)

        free_planet_gm_inds = []
        for i, p in enumerate(fit_planet_gms):
            if p:
                free_planet_gm_inds.append(i)
        free_asteroid_gm_inds = []
        for i, p in enumerate(fit_asteroid_gms):
            if p:
                free_asteroid_gm_inds.append(i)

        # ugly fix for the parser in ias15_likelihood not being able to handle empty
        # arrays, come back to this later
        if tracer_xs.shape[0] == 0:
            tracer_xs = jnp.ones((1, 3)) * 999.0
            tracer_vs = jnp.ones((1, 3)) * 999.0
        if massive_xs.shape[0] == 0:
            massive_xs = jnp.ones((1, 3)) * 999.0
            massive_vs = jnp.ones((1, 3)) * 999.0
            massive_gms = jnp.ones((1)) * 0.0

        return (
            tracer_xs,
            tracer_vs,
            massive_xs,
            massive_vs,
            massive_gms,
            jnp.array(free_tracer_x_inds, dtype=jnp.int64),
            jnp.array(free_tracer_v_inds, dtype=jnp.int64),
            jnp.array(free_massive_x_inds, dtype=jnp.int64),
            jnp.array(free_massive_v_inds, dtype=jnp.int64),
            jnp.array(free_massive_gm_inds, dtype=jnp.int64),
            jnp.array(free_planet_gm_inds, dtype=jnp.int64),
            jnp.array(free_asteroid_gm_inds, dtype=jnp.int64),
            tracer_key,
            massive_key,
        )

    @staticmethod
    def _setup_observations(particles, epoch, planet_params, asteroid_params):
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
        if ntracers == 0:
            ntracers = 1
        if most_leading_obs == 0:
            most_leading_obs = 1
        if most_trailing_obs == 0:
            most_trailing_obs = 1

        trailing_tracer_ra = jnp.zeros((ntracers, most_trailing_obs))
        trailing_tracer_dec = jnp.zeros((ntracers, most_trailing_obs))
        trailing_tracer_astrometric_uncertainties = (
            jnp.ones((ntracers, most_trailing_obs)) * jnp.inf
        )
        trailing_tracer_times = jnp.ones((ntracers, most_trailing_obs)) * 999.0
        trailing_tracer_observer_positions = (
            jnp.ones((ntracers, most_trailing_obs, 3)) * 999.0
        )
        leading_tracer_ra = jnp.zeros((ntracers, most_leading_obs))
        leading_tracer_dec = jnp.zeros((ntracers, most_leading_obs))
        leading_tracer_astrometric_uncertainties = (
            jnp.ones((ntracers, most_leading_obs)) * jnp.inf
        )
        leading_tracer_times = jnp.ones((ntracers, most_leading_obs)) * 999.0
        leading_tracer_observer_positions = (
            jnp.ones((ntracers, most_leading_obs, 3)) * 999.0
        )

        i = 0
        for p in particles:
            if p.gm == 0.0 and not p.free_gm:
                if p.observations is not None:
                    leading = jnp.sum(p.observations.times > epoch)
                    trailing = jnp.sum(p.observations.times < epoch)
                    if leading > 0:
                        mask = p.observations.times > epoch
                        leading_tracer_ra = leading_tracer_ra.at[i, :leading].set(
                            p.observations.ra[mask]
                        )
                        leading_tracer_dec = leading_tracer_dec.at[i, :leading].set(
                            p.observations.dec[mask]
                        )
                        leading_tracer_astrometric_uncertainties = (
                            leading_tracer_astrometric_uncertainties.at[
                                i, :leading
                            ].set(p.observations.astrometric_uncertainties[mask])
                        )
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
                        trailing_tracer_astrometric_uncertainties = (
                            trailing_tracer_astrometric_uncertainties.at[
                                i, :trailing
                            ].set(p.observations.astrometric_uncertainties[mask])
                        )
                        trailing_tracer_times = trailing_tracer_times.at[
                            i, :trailing
                        ].set(p.observations.times[mask])
                        trailing_tracer_observer_positions = (
                            trailing_tracer_observer_positions.at[i, :trailing].set(
                                p.observations.observer_positions[mask]
                            )
                        )
                    i += 1

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
        if nmassive == 0:
            nmassive = 1
        if most_leading_obs == 0:
            most_leading_obs = 1
        if most_trailing_obs == 0:
            most_trailing_obs = 1

        trailing_massive_ra = jnp.zeros((nmassive, most_trailing_obs))
        trailing_massive_dec = jnp.zeros((nmassive, most_trailing_obs))
        trailing_massive_astrometric_uncertainties = (
            jnp.ones((nmassive, most_trailing_obs)) * jnp.inf
        )
        trailing_massive_times = jnp.ones((nmassive, most_trailing_obs)) * 999.0
        trailing_massive_observer_positions = (
            jnp.ones((nmassive, most_trailing_obs, 3)) * 999.0
        )
        leading_massive_ra = jnp.zeros((nmassive, most_leading_obs))
        leading_massive_dec = jnp.zeros((nmassive, most_leading_obs))
        leading_massive_astrometric_uncertainties = (
            jnp.ones((nmassive, most_leading_obs)) * jnp.inf
        )
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
                        leading_massive_ra = leading_massive_ra.at[i, :leading].set(
                            p.observations.ra[mask]
                        )
                        leading_massive_dec = leading_massive_dec.at[i, :leading].set(
                            p.observations.dec[mask]
                        )
                        leading_massive_astrometric_uncertainties = (
                            leading_massive_astrometric_uncertainties.at[
                                i, :leading
                            ].set(p.observations.astrometric_uncertainties[mask])
                        )
                        leading_massive_times = leading_massive_times.at[
                            i, :leading
                        ].set(p.observations.times[mask])
                        leading_massive_observer_positions = (
                            leading_massive_observer_positions.at[i, :leading].set(
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
                        trailing_massive_astrometric_uncertainties = (
                            trailing_massive_astrometric_uncertainties.at[
                                i, :trailing
                            ].set(p.observations.astrometric_uncertainties[mask])
                        )
                        trailing_massive_times = trailing_massive_times.at[
                            i, :trailing
                        ].set(p.observations.times[mask])
                        trailing_massive_observer_positions = (
                            trailing_massive_observer_positions.at[i, :trailing].set(
                                p.observations.observer_positions[mask]
                            )
                        )
                    i += 1

        leading_tracer_planet_xs_at_obs = jax.vmap(
            perturber_positions, in_axes=(None, 0)
        )(planet_params, leading_tracer_times)
        leading_tracer_asteroid_xs_at_obs = jax.vmap(
            perturber_positions, in_axes=(None, 0)
        )(asteroid_params, leading_tracer_times)
        trailing_tracer_planet_xs_at_obs = jax.vmap(
            perturber_positions, in_axes=(None, 0)
        )(planet_params, trailing_tracer_times)
        trailing_tracer_asteroid_xs_at_obs = jax.vmap(
            perturber_positions, in_axes=(None, 0)
        )(asteroid_params, trailing_tracer_times)
        leading_massive_planet_xs_at_obs = jax.vmap(
            perturber_positions, in_axes=(None, 0)
        )(planet_params, leading_massive_times)
        leading_massive_asteroid_xs_at_obs = jax.vmap(
            perturber_positions, in_axes=(None, 0)
        )(asteroid_params, leading_massive_times)
        trailing_massive_planet_xs_at_obs = jax.vmap(
            perturber_positions, in_axes=(None, 0)
        )(planet_params, trailing_massive_times)
        trailing_massive_asteroid_xs_at_obs = jax.vmap(
            perturber_positions, in_axes=(None, 0)
        )(asteroid_params, trailing_massive_times)

        return (
            pad_to_parallelize(leading_tracer_ra, 0.0),
            pad_to_parallelize(leading_tracer_dec, 0.0),
            pad_to_parallelize(leading_tracer_astrometric_uncertainties, jnp.inf),
            pad_to_parallelize(leading_tracer_times, 999.0),
            pad_to_parallelize(leading_tracer_observer_positions, 999.0),
            pad_to_parallelize(leading_tracer_planet_xs_at_obs, 999.0),
            pad_to_parallelize(leading_tracer_asteroid_xs_at_obs, 999.0),
            pad_to_parallelize(trailing_tracer_ra, 0.0),
            pad_to_parallelize(trailing_tracer_dec, 0.0),
            pad_to_parallelize(trailing_tracer_astrometric_uncertainties, jnp.inf),
            pad_to_parallelize(trailing_tracer_times, 999.0),
            pad_to_parallelize(trailing_tracer_observer_positions, 999.0),
            pad_to_parallelize(trailing_tracer_planet_xs_at_obs, 999.0),
            pad_to_parallelize(trailing_tracer_asteroid_xs_at_obs, 999.0),
            pad_to_parallelize(leading_massive_ra, 0.0),
            pad_to_parallelize(leading_massive_dec, 0.0),
            pad_to_parallelize(leading_massive_astrometric_uncertainties, jnp.inf),
            pad_to_parallelize(leading_massive_times, 999.0),
            pad_to_parallelize(leading_massive_observer_positions, 999.0),
            pad_to_parallelize(leading_massive_planet_xs_at_obs, 999.0),
            pad_to_parallelize(leading_massive_asteroid_xs_at_obs, 999.0),
            pad_to_parallelize(trailing_massive_ra, 0.0),
            pad_to_parallelize(trailing_massive_dec, 0.0),
            pad_to_parallelize(trailing_massive_astrometric_uncertainties, jnp.inf),
            pad_to_parallelize(trailing_massive_times, 999.0),
            pad_to_parallelize(trailing_massive_observer_positions, 999.0),
            pad_to_parallelize(trailing_massive_planet_xs_at_obs, 999.0),
            pad_to_parallelize(trailing_massive_asteroid_xs_at_obs, 999.0),
        )

    ####################################################################################
    # likelihood methods
    ####################################################################################
    @staticmethod
    def generate_initial_free_params(
        tracer_xs,
        tracer_vs,
        massive_xs,
        massive_vs,
        massive_gms,
        planet_gms,
        asteroid_gms,
        free_tracer_x_inds,
        free_tracer_v_inds,
        free_massive_x_inds,
        free_massive_v_inds,
        free_massive_gm_inds,
        free_planet_gm_inds,
        free_asteroid_gm_inds,
    ):
        x = jnp.concatenate(
            [
                tracer_xs[free_tracer_x_inds].flatten(),
                tracer_vs[free_tracer_v_inds].flatten(),
                massive_xs[free_massive_x_inds].flatten(),
                massive_vs[free_massive_v_inds].flatten(),
                massive_gms[free_massive_gm_inds].flatten(),
                planet_gms[free_planet_gm_inds].flatten(),
                asteroid_gms[free_asteroid_gm_inds].flatten(),
            ]
        )
        return x

    @property
    def particles(self):
        pass

    @property
    def residuals(self):
        pass

    def _generate_likelihood_funcs(self):
        full_resids = jax.tree_util.Partial(
            ias15_likelihood,
            tracer_xs=self._tracer_xs,
            tracer_vs=self._tracer_vs,
            massive_xs=self._massive_xs,
            massive_vs=self._massive_vs,
            massive_gms=self._massive_gms,
            planet_gms=self._planet_gms,
            asteroid_gms=self._asteroid_gms,
            free_tracer_x_inds=self._free_tracer_x_inds,
            free_tracer_v_inds=self._free_tracer_v_inds,
            free_massive_x_inds=self._free_massive_x_inds,
            free_massive_v_inds=self._free_massive_v_inds,
            free_massive_gm_inds=self._free_massive_gm_inds,
            free_planet_gm_inds=self._free_planet_gm_inds,
            free_asteroid_gm_inds=self._free_asteroid_gm_inds,
            epoch=self._epoch,
            planet_params=self._planet_params,
            asteroid_params=self._asteroid_params,
            leading_tracer_times=self._leading_tracer_times,
            leading_tracer_observer_positions=self._leading_tracer_observer_positions,
            leading_tracer_planet_xs_at_obs=self._leading_tracer_planet_xs_at_obs,
            leading_tracer_asteroid_xs_at_obs=self._leading_tracer_asteroid_xs_at_obs,
            leading_tracer_ra=self._leading_tracer_ra,
            leading_tracer_dec=self._leading_tracer_dec,
            leading_tracer_astrometric_uncertainties=self._leading_tracer_astrometric_uncertainties,
            trailing_tracer_times=self._trailing_tracer_times,
            trailing_tracer_observer_positions=self._trailing_tracer_observer_positions,
            trailing_tracer_planet_xs_at_obs=self._trailing_tracer_planet_xs_at_obs,
            trailing_tracer_asteroid_xs_at_obs=self._trailing_tracer_asteroid_xs_at_obs,
            trailing_tracer_ra=self._trailing_tracer_ra,
            trailing_tracer_dec=self._trailing_tracer_dec,
            trailing_tracer_astrometric_uncertainties=self._trailing_tracer_astrometric_uncertainties,
            leading_massive_times=self._leading_massive_times,
            leading_massive_observer_positions=self._leading_massive_observer_positions,
            leading_massive_planet_xs_at_obs=self._leading_massive_planet_xs_at_obs,
            leading_massive_asteroid_xs_at_obs=self._leading_massive_asteroid_xs_at_obs,
            leading_massive_ra=self._leading_massive_ra,
            leading_massive_dec=self._leading_massive_dec,
            leading_massive_astrometric_uncertainties=self._leading_massive_astrometric_uncertainties,
            trailing_massive_times=self._trailing_massive_times,
            trailing_massive_observer_positions=self._trailing_massive_observer_positions,
            trailing_massive_planet_xs_at_obs=self._trailing_massive_planet_xs_at_obs,
            trailing_massive_asteroid_xs_at_obs=self._trailing_massive_asteroid_xs_at_obs,
            trailing_massive_ra=self._trailing_massive_ra,
            trailing_massive_dec=self._trailing_massive_dec,
            trailing_massive_astrometric_uncertainties=self._trailing_massive_astrometric_uncertainties,
        )

        _loglike = lambda x: full_resids(x)[0]

        @jax.custom_jvp
        def loglike(x):
            return _loglike(x)

        @loglike.defjvp
        def fwd(primals, tangents):
            (x,) = primals
            (x_dot,) = tangents
            ans = _loglike(x)
            ans_dot = jax.jacfwd(_loglike)(x)
            return ans, jnp.sum(ans_dot * x_dot)

        return full_resids, loglike

    def propagate(self, t):
        pass

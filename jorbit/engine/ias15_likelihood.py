import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.engine.ias15_integrator import (
    ias15_integrate_multiple,
    ias15_initial_params,
)
from jorbit.engine.sky_projection import on_sky, sky_error
from jorbit.engine.accelerations import acceleration_at_time


def scanned_array_update(x, indecies, vals):
    def scan_fun(carry, scan_over):
        ind, val = scan_over
        return (carry.at[ind].set(val), None)

    return jax.lax.scan(scan_fun, x, (indecies, vals))[0]


def parser(
    x,
    tracer_xs_template,
    tracer_vs_template,
    massive_xs_template,
    massive_vs_template,
    massive_gms_template,
    planet_gms_template,
    asteroid_gms_template,
    free_tracer_x_inds,
    free_tracer_v_inds,
    free_massive_x_inds,
    free_massive_v_inds,
    free_massive_gm_inds,
    free_planet_gm_inds,
    free_asteroid_gm_inds,
):
    lb = 0
    ub = free_tracer_x_inds.size * 3
    tracer_xs = scanned_array_update(
        tracer_xs_template, free_tracer_x_inds, x[:ub].reshape((-1, 3))
    )

    lb = ub
    ub += free_tracer_v_inds.size * 3
    tracer_vs = scanned_array_update(
        tracer_vs_template, free_tracer_v_inds, x[lb:ub].reshape((-1, 3))
    )

    lb = ub
    ub += free_massive_x_inds.size * 3
    massive_xs = scanned_array_update(
        massive_xs_template, free_massive_x_inds, x[lb:ub].reshape((-1, 3))
    )

    lb = ub
    ub += free_massive_v_inds.size * 3
    massive_vs = scanned_array_update(
        massive_vs_template, free_massive_v_inds, x[lb:ub].reshape((-1, 3))
    )

    lb = ub
    ub += free_massive_gm_inds.size
    massive_gms = scanned_array_update(
        massive_gms_template, free_massive_gm_inds, x[lb:ub]
    )

    lb = ub
    ub += free_planet_gm_inds.size
    planet_gms = scanned_array_update(
        planet_gms_template, free_planet_gm_inds, x[lb:ub]
    )

    lb = ub
    ub += free_asteroid_gm_inds.size
    asteroid_gms = scanned_array_update(
        asteroid_gms_template, free_asteroid_gm_inds, x[lb:ub]
    )

    return (
        tracer_xs,
        tracer_vs,
        massive_xs,
        massive_vs,
        massive_gms,
        planet_gms,
        asteroid_gms,
    )


def single_tracer_likelihood(
    t0,
    massive_x0,
    massive_v0,
    massive_gms,
    planet_gms,
    asteroid_gms,
    planet_params,
    asteroid_params,
    indiv_tracer_x,
    indiv_tracer_v,
    indiv_tracer_obs_times,
    indiv_tracer_observer_positions,
    indiv_tracer_planet_xs_at_obs,
    indiv_tracer_asteroid_xs_at_obs,
    indiv_tracer_ra,
    indiv_tracer_dec,
    indiv_tracer_astrometric_uncertainties,
):
    def true_func():
        x = jnp.concatenate((massive_x0, indiv_tracer_x[None, :]))
        v = jnp.concatenate((massive_v0, indiv_tracer_v[None, :]))
        gms = jnp.concatenate((massive_gms, jnp.array([0.0])))

        a0 = acceleration_at_time(
            x,
            v,
            gms,
            t0,
            planet_params,
            asteroid_params,
            planet_gms,
            asteroid_gms,
        )

        xs, vs, ts = ias15_integrate_multiple(
            x0=x,
            v0=v,
            a0=a0,
            acc=jax.tree_util.Partial(acceleration_at_time),
            acc_fixed_kwargs={
                "gm": gms,
                "planet_params": planet_params,
                "asteroid_params": asteroid_params,
                "planet_gms": planet_gms,
                "asteroid_gms": asteroid_gms,
            },
            acc_free_kwargs={},
            t0=t0,
            tfs=indiv_tracer_obs_times,
            **ias15_initial_params(x.shape[0]),
        )

        # cut to just the particle of interest
        xs = xs[:, -1, :]  # (n_times, 3)
        vs = vs[:, -1, :]  # (n_times, 3)

        calc_ra, calc_dec = on_sky(
            xs=xs,
            vs=vs,
            gms=jnp.zeros(xs.shape[0]),
            observer_positions=indiv_tracer_observer_positions,
            planet_xs=indiv_tracer_planet_xs_at_obs,
            asteroid_xs=indiv_tracer_asteroid_xs_at_obs,
            planet_gms=planet_gms,
            asteroid_gms=asteroid_gms,
        )

        resids = sky_error(
            calc_ra=calc_ra,
            calc_dec=calc_dec,
            true_ra=indiv_tracer_ra,
            true_dec=indiv_tracer_dec,
        )

        sigma2 = indiv_tracer_astrometric_uncertainties**2
        loglike = -0.5 * jnp.sum(resids**2 / sigma2)

        return (xs, vs, calc_ra, calc_dec, resids, loglike)

    def false_func():
        return (
            jnp.ones((indiv_tracer_obs_times.shape[0], 3)) * 999.0,
            jnp.ones((indiv_tracer_obs_times.shape[0], 3)) * 999.0,
            jnp.zeros(indiv_tracer_obs_times.shape[0]),
            jnp.zeros(indiv_tracer_obs_times.shape[0]),
            jnp.ones(indiv_tracer_obs_times.shape[0]) * jnp.inf,
            0.0,
        )

    return jax.lax.cond(
        indiv_tracer_astrometric_uncertainties[0] != jnp.inf, true_func, false_func
    )


def likelihood(
    x,
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
    planet_params,
    asteroid_params,
    tracer_ras,
    tracer_decs,
    tracer_obs_times,
    tracer_obs_uncertainties,
    tracer_oberver_positions,
    massive_ras,
    massive_decs,
    massive_obs_times,
    massive_obs_uncertainties,
    massive_oberver_positions,
):
    (
        tracer_xs,
        tracer_vs,
        massive_xs,
        massive_vs,
        massive_gms,
        planet_gms,
        asteroid_gms,
    ) = parser(
        x,
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
    )

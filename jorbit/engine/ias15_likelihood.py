import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.engine.ias15_integrator import (
    ias15_integrate_multiple,
    ias15_initial_params,
)

from jorbit.engine import pad_to_parallelize
from jorbit.engine.sky_projection import on_sky, sky_error
from jorbit.engine.accelerations import acceleration_at_time

########################################################################################
# Helpers
########################################################################################


def _scanned_array_update_inner(carry, scan_over):
    ind, val = scan_over
    return (carry.at[ind].set(val), None)


@jax.jit
def scanned_array_update(x, indecies, vals):
    return jax.lax.scan(_scanned_array_update_inner, x, (indecies, vals))[0]


@jax.jit
def positions_to_residuals(
    xs,
    vs,
    observer_positions,
    planet_xs,
    asteroid_xs,
    planet_gms,
    asteroid_gms,
    ras,
    decs,
    astrometric_uncertainties,
):
    """
    Convert cartesian positions to on-sky positions, residuals, and loglikelihoods
    """
    calc_ra, calc_dec = on_sky(
        xs=xs,
        vs=vs,
        gms=jnp.zeros(xs.shape[0]),
        observer_positions=observer_positions,
        planet_xs=planet_xs,
        asteroid_xs=asteroid_xs,
        planet_gms=planet_gms,
        asteroid_gms=asteroid_gms,
    )

    resids = sky_error(
        calc_ra=calc_ra,
        calc_dec=calc_dec,
        true_ra=ras,
        true_dec=decs,
    )

    sigma2 = astrometric_uncertainties**2
    loglike = -0.5 * jnp.sum(resids**2 / sigma2)

    return calc_ra, calc_dec, resids, loglike


@jax.jit
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


########################################################################################
# Tracer particles
########################################################################################


def _single_tracer_true(
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
    """
    Feeds into single_tracer_likelihood, the branch to compute
    if the particle isn't entirely masked
    """
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
        acc=jax.tree_util.Partial(
            acceleration_at_time,
            **{
                "gm": gms,
                "planet_params": planet_params,
                "asteroid_params": asteroid_params,
                "planet_gms": planet_gms,
                "asteroid_gms": asteroid_gms,
            },
        ),
        acc_free_kwargs={},
        t0=t0,
        tfs=indiv_tracer_obs_times,
        **ias15_initial_params(x.shape[0]),
    )

    # cut to just the particle of interest
    xs = xs[:, -1, :]  # (n_times, 3)
    vs = vs[:, -1, :]  # (n_times, 3)

    calc_ra, calc_dec, resids, loglike = positions_to_residuals(
        xs=xs,
        vs=vs,
        observer_positions=indiv_tracer_observer_positions,
        planet_xs=indiv_tracer_planet_xs_at_obs,
        asteroid_xs=indiv_tracer_asteroid_xs_at_obs,
        planet_gms=planet_gms,
        asteroid_gms=asteroid_gms,
        ras=indiv_tracer_ra,
        decs=indiv_tracer_dec,
        astrometric_uncertainties=indiv_tracer_astrometric_uncertainties,
    )

    return (xs, vs, calc_ra, calc_dec, resids, loglike)


def _single_tracer_false(
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
    """
    Feeds into single_tracer_likelihood, the branch to compute
    if the particle is entirely masked
    """
    return (
        jnp.ones((indiv_tracer_obs_times.shape[0], 3)) * 999.0,
        jnp.ones((indiv_tracer_obs_times.shape[0], 3)) * 999.0,
        jnp.zeros(indiv_tracer_obs_times.shape[0]),
        jnp.zeros(indiv_tracer_obs_times.shape[0]),
        jnp.ones(indiv_tracer_obs_times.shape[0]) * jnp.inf,
        0.0,
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
    """
    Computes the residuals and loglikelihood for a single tracer particle
    """
    return jax.lax.cond(
        indiv_tracer_astrometric_uncertainties[0] != jnp.inf,
        _single_tracer_true,
        _single_tracer_false,
        *(
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
        ),
    )


def _tracer_likelihood_batch_scan(carry, scan_over):
    """
    Feeds into tracer_likelihood_batch, which computes the residuals and log likelihood
    for a batch of tracer particles
    """
    res = single_tracer_likelihood(*carry, *scan_over)
    return carry, res


@jax.jit
def tracer_likelihood_batch(
    t0,
    massive_x0,
    massive_v0,
    massive_gms,
    planet_gms,
    asteroid_gms,
    planet_params,
    asteroid_params,
    batch_tracer_x,
    batch_tracer_v,
    batch_tracer_obs_times,
    batch_tracer_observer_positions,
    batch_tracer_planet_xs_at_obs,
    batch_tracer_asteroid_xs_at_obs,
    batch_tracer_ra,
    batch_tracer_dec,
    batch_tracer_astrometric_uncertainties,
):
    """
    Compute the residuals and log likelihood for a batch of tracer particles
    """
    return jax.lax.scan(
        _tracer_likelihood_batch_scan,
        (
            t0,
            massive_x0,
            massive_v0,
            massive_gms,
            planet_gms,
            asteroid_gms,
            planet_params,
            asteroid_params,
        ),
        (
            batch_tracer_x,
            batch_tracer_v,
            batch_tracer_obs_times,
            batch_tracer_observer_positions,
            batch_tracer_planet_xs_at_obs,
            batch_tracer_asteroid_xs_at_obs,
            batch_tracer_ra,
            batch_tracer_dec,
            batch_tracer_astrometric_uncertainties,
        ),
    )[1]


########################################################################################
# Massive particles
########################################################################################
def _single_massive_true(
    t0,
    massive_x0,
    massive_v0,
    massive_gms,
    planet_gms,
    asteroid_gms,
    planet_params,
    asteroid_params,
    indiv_massive_ind,
    indiv_massive_obs_times,
    indiv_massive_observer_positions,
    indiv_massive_planet_xs_at_obs,
    indiv_massive_asteroid_xs_at_obs,
    indiv_massive_ra,
    indiv_massive_dec,
    indiv_massive_astrometric_uncertainties,
):
    """
    Feeds into single_massive_likelihood, the branch to compute
    if the particle isn't entirely masked
    """
    a0 = acceleration_at_time(
        massive_x0,
        massive_v0,
        massive_gms,
        t0,
        planet_params,
        asteroid_params,
        planet_gms,
        asteroid_gms,
    )

    xs, vs, ts = ias15_integrate_multiple(
        x0=massive_x0,
        v0=massive_v0,
        a0=a0,
        acc=jax.tree_util.Partial(
            acceleration_at_time,
            **{
                "gm": massive_gms,
                "planet_params": planet_params,
                "asteroid_params": asteroid_params,
                "planet_gms": planet_gms,
                "asteroid_gms": asteroid_gms,
            },
        ),
        acc_free_kwargs={},
        t0=t0,
        tfs=indiv_massive_obs_times,
        **ias15_initial_params(massive_x0.shape[0]),
    )

    # cut to just the particle of interest
    xs = xs[:, indiv_massive_ind, :]  # (n_times, 3)
    vs = vs[:, indiv_massive_ind, :]  # (n_times, 3)

    calc_ra, calc_dec, resids, loglike = positions_to_residuals(
        xs=xs,
        vs=vs,
        observer_positions=indiv_massive_observer_positions,
        planet_xs=indiv_massive_planet_xs_at_obs,
        asteroid_xs=indiv_massive_asteroid_xs_at_obs,
        planet_gms=planet_gms,
        asteroid_gms=asteroid_gms,
        ras=indiv_massive_ra,
        decs=indiv_massive_dec,
        astrometric_uncertainties=indiv_massive_astrometric_uncertainties,
    )

    return (xs, vs, calc_ra, calc_dec, resids, loglike)


def _single_massive_false(
    t0,
    massive_x0,
    massive_v0,
    massive_gms,
    planet_gms,
    asteroid_gms,
    planet_params,
    asteroid_params,
    indiv_massive_ind,
    indiv_massive_obs_times,
    indiv_massive_observer_positions,
    indiv_massive_planet_xs_at_obs,
    indiv_massive_asteroid_xs_at_obs,
    indiv_massive_ra,
    indiv_massive_dec,
    indiv_massive_astrometric_uncertainties,
):
    """
    Feeds into single_massive_likelihood, the branch to compute
    if the particle is entirely masked
    """
    return (
        jnp.ones((indiv_massive_obs_times.shape[0], 3)) * 999.0,
        jnp.ones((indiv_massive_obs_times.shape[0], 3)) * 999.0,
        jnp.zeros(indiv_massive_obs_times.shape[0]),
        jnp.zeros(indiv_massive_obs_times.shape[0]),
        jnp.ones(indiv_massive_obs_times.shape[0]) * jnp.inf,
        0.0,
    )


def single_massive_likelihood(
    t0,
    massive_x0,
    massive_v0,
    massive_gms,
    planet_gms,
    asteroid_gms,
    planet_params,
    asteroid_params,
    indiv_massive_ind,
    indiv_massive_obs_times,
    indiv_massive_observer_positions,
    indiv_massive_planet_xs_at_obs,
    indiv_massive_asteroid_xs_at_obs,
    indiv_massive_ra,
    indiv_massive_dec,
    indiv_massive_astrometric_uncertainties,
):
    """
    Computes the residuals and loglikelihood for a single massive particle
    """
    return jax.lax.cond(
        indiv_massive_astrometric_uncertainties[0] != jnp.inf,
        _single_massive_true,
        _single_massive_false,
        *(
            t0,
            massive_x0,
            massive_v0,
            massive_gms,
            planet_gms,
            asteroid_gms,
            planet_params,
            asteroid_params,
            indiv_massive_ind,
            indiv_massive_obs_times,
            indiv_massive_observer_positions,
            indiv_massive_planet_xs_at_obs,
            indiv_massive_asteroid_xs_at_obs,
            indiv_massive_ra,
            indiv_massive_dec,
            indiv_massive_astrometric_uncertainties,
        ),
    )


def _massive_likelihood_batch_scan(carry, scan_over):
    """
    Feeds into tracer_likelihood_batch, which computes the residuals and log likelihood
    for a batch of massive particles
    """
    res = single_massive_likelihood(*carry, *scan_over)
    return carry, res


@jax.jit
def massive_likelihood_batch(
    t0,
    massive_x0,
    massive_v0,
    massive_gms,
    planet_gms,
    asteroid_gms,
    planet_params,
    asteroid_params,
    batch_massive_ind,
    batch_massive_obs_times,
    batch_massive_observer_positions,
    batch_massive_planet_xs_at_obs,
    batch_massive_asteroid_xs_at_obs,
    batch_massive_ra,
    batch_massive_dec,
    batch_massive_astrometric_uncertainties,
):
    return jax.lax.scan(
        _massive_likelihood_batch_scan,
        (
            t0,
            massive_x0,
            massive_v0,
            massive_gms,
            planet_gms,
            asteroid_gms,
            planet_params,
            asteroid_params,
        ),
        (
            batch_massive_ind,
            batch_massive_obs_times,
            batch_massive_observer_positions,
            batch_massive_planet_xs_at_obs,
            batch_massive_asteroid_xs_at_obs,
            batch_massive_ra,
            batch_massive_dec,
            batch_massive_astrometric_uncertainties,
        ),
    )[1]


########################################################################################
# Assembling everything
########################################################################################
tracer_parallel = jax.pmap(tracer_likelihood_batch, in_axes=((None,) * 8 + (0,) * 9))
massive_parallel = jax.pmap(massive_likelihood_batch, in_axes=((None,) * 8 + (0,) * 8))


@jax.jit
def generate_batch_inds(arr):
    a = arr.shape[0]
    b = arr.shape[1]
    return jnp.repeat(jnp.arange(b), a).reshape(a, b)


@jax.jit
def combine_loglikes(
    leading_tracer_loglikes,
    trailing_tracer_loglikes,
    leading_massive_loglikes,
    trailing_massive_loglikes,
):
    return (
        jnp.sum(leading_tracer_loglikes)
        + jnp.sum(trailing_tracer_loglikes)
        + jnp.sum(leading_massive_loglikes)
        + jnp.sum(trailing_massive_loglikes)
    )


def ias15_likelihood(
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
    epoch,
    planet_params,
    asteroid_params,
    leading_tracer_times,
    leading_tracer_observer_positions,
    leading_tracer_planet_xs_at_obs,
    leading_tracer_asteroid_xs_at_obs,
    leading_tracer_ra,
    leading_tracer_dec,
    leading_tracer_astrometric_uncertainties,
    trailing_tracer_times,
    trailing_tracer_observer_positions,
    trailing_tracer_planet_xs_at_obs,
    trailing_tracer_asteroid_xs_at_obs,
    trailing_tracer_ra,
    trailing_tracer_dec,
    trailing_tracer_astrometric_uncertainties,
    leading_massive_times,
    leading_massive_observer_positions,
    leading_massive_planet_xs_at_obs,
    leading_massive_asteroid_xs_at_obs,
    leading_massive_ra,
    leading_massive_dec,
    leading_massive_astrometric_uncertainties,
    trailing_massive_times,
    trailing_massive_observer_positions,
    trailing_massive_planet_xs_at_obs,
    trailing_massive_asteroid_xs_at_obs,
    trailing_massive_ra,
    trailing_massive_dec,
    trailing_massive_astrometric_uncertainties,
):
    # setup
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

    # contribution from the tracer particles
    tracer_xs = pad_to_parallelize(tracer_xs, 999.0)
    tracer_vs = pad_to_parallelize(tracer_vs, 999.0)

    (
        leading_tracer_xs,
        leading_tracer_vs,
        leading_tracer_ras,
        leading_tracer_decs,
        leading_tracer_resids,
        leading_tracer_loglikes,
    ) = tracer_parallel(
        epoch,
        massive_xs,
        massive_vs,
        massive_gms,
        planet_gms,
        asteroid_gms,
        planet_params,
        asteroid_params,
        tracer_xs,
        tracer_vs,
        leading_tracer_times,
        leading_tracer_observer_positions,
        leading_tracer_planet_xs_at_obs,
        leading_tracer_asteroid_xs_at_obs,
        leading_tracer_ra,
        leading_tracer_dec,
        leading_tracer_astrometric_uncertainties,
    )

    (
        trailing_tracer_xs,
        trailing_tracer_vs,
        trailing_tracer_ras,
        trailing_tracer_decs,
        trailing_tracer_resids,
        trailing_tracer_loglikes,
    ) = tracer_parallel(
        epoch,
        massive_xs,
        massive_vs,
        massive_gms,
        planet_gms,
        asteroid_gms,
        planet_params,
        asteroid_params,
        tracer_xs,
        tracer_vs,
        trailing_tracer_times,
        trailing_tracer_observer_positions,
        trailing_tracer_planet_xs_at_obs,
        trailing_tracer_asteroid_xs_at_obs,
        trailing_tracer_ra,
        trailing_tracer_dec,
        trailing_tracer_astrometric_uncertainties,
    )

    # contribution from the massive particles

    batch_inds = generate_batch_inds(leading_massive_times)

    (
        leading_massive_xs,
        leading_massive_vs,
        leading_massive_ras,
        leading_massive_decs,
        leading_massive_resids,
        leading_massive_loglikes,
    ) = massive_parallel(
        epoch,
        massive_xs,
        massive_vs,
        massive_gms,
        planet_gms,
        asteroid_gms,
        planet_params,
        asteroid_params,
        batch_inds,
        leading_massive_times,
        leading_massive_observer_positions,
        leading_massive_planet_xs_at_obs,
        leading_massive_asteroid_xs_at_obs,
        leading_massive_ra,
        leading_massive_dec,
        leading_massive_astrometric_uncertainties,
    )

    (
        trailing_massive_xs,
        trailing_massive_vs,
        trailing_massive_ras,
        trailing_massive_decs,
        trailing_massive_resids,
        trailing_massive_loglikes,
    ) = massive_parallel(
        epoch,
        massive_xs,
        massive_vs,
        massive_gms,
        planet_gms,
        asteroid_gms,
        planet_params,
        asteroid_params,
        batch_inds,
        trailing_massive_times,
        trailing_massive_observer_positions,
        trailing_massive_planet_xs_at_obs,
        trailing_massive_asteroid_xs_at_obs,
        trailing_massive_ra,
        trailing_massive_dec,
        trailing_massive_astrometric_uncertainties,
    )

    loglike = combine_loglikes(
        leading_tracer_loglikes,
        trailing_tracer_loglikes,
        leading_massive_loglikes,
        trailing_massive_loglikes,
    )
    return loglike, (
        (
            trailing_tracer_xs,
            trailing_tracer_vs,
            trailing_tracer_ras,
            trailing_tracer_decs,
            trailing_tracer_resids,
            trailing_tracer_loglikes,
        ),
        (
            leading_tracer_xs,
            leading_tracer_vs,
            leading_tracer_ras,
            leading_tracer_decs,
            leading_tracer_resids,
            leading_tracer_loglikes,
        ),
        (
            trailing_massive_xs,
            trailing_massive_vs,
            trailing_massive_ras,
            trailing_massive_decs,
            trailing_massive_resids,
            trailing_massive_loglikes,
        ),
        (
            leading_massive_xs,
            leading_massive_vs,
            leading_massive_ras,
            leading_massive_decs,
            leading_massive_resids,
            leading_massive_loglikes,
        ),
    )

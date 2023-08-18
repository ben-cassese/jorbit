import jax
from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.engine.gauss_jackson_integrator import gj_integrate_multiple
from jorbit.engine.sky_projection import on_sky, sky_error


def tracer_likelihoods(
    tracer_x0s,
    tracer_v0s,
    tracer_Init_Times,
    tracer_Jump_Times,
    tracer_Valid_Steps,
    tracer_Planet_Xs,
    tracer_Planet_Vs,
    tracer_Planet_As,
    tracer_Asteroid_Xs,
    tracer_Planet_Xs_Warmup,
    tracer_Asteroid_Xs_Warmup,
    tracer_Dts_Warmup,
    tracer_Observer_Positions,
    tracer_RAs,
    tracer_Decs,
    tracer_Observed_Planet_Xs,
    tracer_Observed_Asteroid_Xs,
    tracer_Astrometric_Uncertainties,
    massive_x0s,
    massive_v0s,
    massive_gms,
    planet_gms,
    asteroid_gms,
    bjk,
    ajk,
    yc,
    yd,
):
    """
    Wrapper for a function that scans over multiple particles, meant to
    operate on pmapped chunks of particles
    """

    def _tracer_scan_func(carry, scan_over):
        """
        This scans over multiple particles
        """

        def true_func(scan_over):
            (
                x0,
                v0,
                init_time,
                jump_times,
                valid_steps,
                planet_xs,
                planet_vs,
                planet_as,
                asteroid_xs,
                planet_xs_warmup,
                asteroid_xs_warmup,
                dts_warmup,
                observer_positions,
                ras,
                decs,
                observed_planet_xs,
                observed_asteroid_xs,
                astrometric_uncertainties,
            ) = scan_over
            x0 = jnp.concatenate((jnp.array([x0]), massive_x0s))
            v0 = jnp.concatenate((jnp.array([v0]), massive_v0s))
            gms = jnp.concatenate((jnp.array([0.0]), massive_gms))

            x, v = gj_integrate_multiple(
                x0=x0,
                v0=v0,
                gms=gms,
                valid_steps=valid_steps,
                b_jk=bjk,
                a_jk=ajk,
                t0=init_time,
                times=jump_times,
                planet_xs=planet_xs,
                planet_vs=planet_vs,
                planet_as=planet_as,
                asteroid_xs=asteroid_xs,
                planet_xs_warmup=planet_xs_warmup,
                asteroid_xs_warmup=asteroid_xs_warmup,
                dts_warmup=dts_warmup,
                warmup_C=yc,
                warmup_D=yd,
                planet_gms=planet_gms,
                asteroid_gms=asteroid_gms,
                use_GR=True,
            )

            # we only care about the first particle, the tracer
            # add the first position back in
            x = jnp.concatenate((x0[0][None], x[0, :, :]))
            v = jnp.concatenate((v0[0][None], v[0, :, :]))

            calc_ra, calc_dec = on_sky(
                xs=x,
                vs=v,
                gms=jnp.zeros(x.shape[0]),
                observer_positions=observer_positions,
                planet_xs=observed_planet_xs,
                asteroid_xs=observed_asteroid_xs,
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

            return None, (x, v, calc_ra, calc_dec, resids, loglike)

        def false_func(scan_over):
            (
                x0,
                v0,
                init_time,
                jump_times,
                valid_steps,
                planet_xs,
                planet_vs,
                planet_as,
                asteroid_xs,
                planet_xs_warmup,
                asteroid_xs_warmup,
                dts_warmup,
                observer_positions,
                ras,
                decs,
                observed_planet_xs,
                observed_asteroid_xs,
                astrometric_uncertainties,
            ) = scan_over
            return None, (
                jnp.ones((jump_times.shape[0] + 1, 3)) * 999.0,
                jnp.ones((jump_times.shape[0] + 1, 3)) * 999.0,
                jnp.zeros(jump_times.shape[0] + 1),
                jnp.zeros(jump_times.shape[0] + 1),
                jnp.ones(jump_times.shape[0] + 1) * 999.0,
                0.0,
            )

        a = scan_over[5]
        return jax.lax.cond(
            a[tuple([0] * (len(a.shape)))] != 999, true_func, false_func, scan_over
        )

    tracer_xs, tracer_vs, tracer_ras, tracer_decs, tracer_resids, tracer_loglike = (
        jax.lax.scan(
            _tracer_scan_func,
            None,
            (
                tracer_x0s,
                tracer_v0s,
                tracer_Init_Times,
                tracer_Jump_Times,
                tracer_Valid_Steps,
                tracer_Planet_Xs,
                tracer_Planet_Vs,
                tracer_Planet_As,
                tracer_Asteroid_Xs,
                tracer_Planet_Xs_Warmup,
                tracer_Asteroid_Xs_Warmup,
                tracer_Dts_Warmup,
                tracer_Observer_Positions,
                tracer_RAs,
                tracer_Decs,
                tracer_Observed_Planet_Xs,
                tracer_Observed_Asteroid_Xs,
                tracer_Astrometric_Uncertainties,
            ),
        )[1]
    )

    # These aren't necessary and don't affect the likelihood, but makes it easier to
    # interpret
    tracer_ras = jnp.where(tracer_Astrometric_Uncertainties != jnp.inf, tracer_ras, 0.0)
    tracer_decs = jnp.where(
        tracer_Astrometric_Uncertainties != jnp.inf, tracer_decs, 0.0
    )
    tracer_resids = jnp.where(
        tracer_Astrometric_Uncertainties != jnp.inf, tracer_resids, 999.0
    )

    tracer_loglike = jnp.sum(tracer_loglike)
    return tracer_xs, tracer_vs, tracer_ras, tracer_decs, tracer_resids, tracer_loglike


def massive_likelihoods(
    scan_inds,
    massive_x0s,
    massive_v0s,
    massive_gms,
    massive_Valid_Steps,
    massive_Init_Times,
    massive_Jump_Times,
    massive_Planet_Xs,
    massive_Planet_Vs,
    massive_Planet_As,
    massive_Asteroid_Xs,
    massive_Planet_Xs_Warmup,
    massive_Asteroid_Xs_Warmup,
    massive_Dts_Warmup,
    massive_Observer_Positions,
    massive_Observed_Planet_Xs,
    massive_Observed_Asteroid_Xs,
    massive_RAs,
    massive_Decs,
    massive_Astrometric_Uncertainties,
    bjk,
    ajk,
    yc,
    yd,
    planet_gms,
    asteroid_gms,
):
    """
    Wrapper for a function that scans over multiple particles, meant to
    operate on pmapped chunks of particles
    """

    def _massive_scan_func(carry, scan_over):
        """
        This scans over multiple particles, but only by passing the index
        since all particles are needed when they have possibly non-zero GMs
        """

        def true_func(scan_over):
            ind = scan_over
            x, v = gj_integrate_multiple(
                x0=massive_x0s,
                v0=massive_v0s,
                gms=massive_gms,
                valid_steps=massive_Valid_Steps[ind],
                b_jk=bjk,
                a_jk=ajk,
                t0=massive_Init_Times[ind],
                times=massive_Jump_Times[ind],
                planet_xs=massive_Planet_Xs[ind],
                planet_vs=massive_Planet_Vs[ind],
                planet_as=massive_Planet_As[ind],
                asteroid_xs=massive_Asteroid_Xs[ind],
                planet_xs_warmup=massive_Planet_Xs_Warmup[ind],
                asteroid_xs_warmup=massive_Asteroid_Xs_Warmup[ind],
                dts_warmup=massive_Dts_Warmup[ind],
                warmup_C=yc,
                warmup_D=yd,
                planet_gms=planet_gms,
                asteroid_gms=asteroid_gms,
                use_GR=True,
            )

            # add the first position back in
            x = jnp.concatenate((massive_x0s[ind][None, :], x[ind, :, :]))
            v = jnp.concatenate((massive_v0s[ind][None, :], v[ind, :, :]))

            calc_ra, calc_dec = on_sky(
                xs=x,
                vs=v,
                gms=jnp.ones(x.shape[0]) * massive_gms[ind],
                observer_positions=massive_Observer_Positions[ind],
                planet_xs=massive_Observed_Planet_Xs[ind],
                asteroid_xs=massive_Observed_Asteroid_Xs[ind],
                planet_gms=planet_gms,
                asteroid_gms=asteroid_gms,
            )

            resids = sky_error(
                calc_ra=calc_ra,
                calc_dec=calc_dec,
                true_ra=massive_RAs[ind],
                true_dec=massive_Decs[ind],
            )

            sigma2 = massive_Astrometric_Uncertainties[ind] ** 2
            loglike = -0.5 * jnp.sum(resids**2 / sigma2)

            return None, (x, v, calc_ra, calc_dec, resids, loglike)

        def false_func(scan_over):
            j = massive_Jump_Times[scan_over].shape[0]
            return None, (
                jnp.ones((j + 1, 3)) * 999.0,
                jnp.ones((j + 1, 3)) * 999.0,
                jnp.zeros(j + 1),
                jnp.zeros(j + 1),
                jnp.ones(j + 1) * 999.0,
                0.0,
            )

        a = massive_Planet_Xs[scan_over]
        return jax.lax.cond(
            a[tuple([0] * (len(a.shape)))] != 999, true_func, false_func, scan_over
        )

    (
        massive_xs,
        massive_vs,
        massive_calc_RAs,
        massive_calc_Decs,
        massive_resids,
        massive_loglike,
    ) = jax.lax.scan(_massive_scan_func, None, scan_inds,)[1]

    # These aren't necessary and don't affect the likelihood, but makes it easier to
    # interpret
    massive_calc_RAs = jnp.where(
        massive_Astrometric_Uncertainties != jnp.inf, massive_calc_RAs, 0.0
    )
    massive_calc_Decs = jnp.where(
        massive_Astrometric_Uncertainties != jnp.inf, massive_calc_Decs, 0.0
    )
    massive_resids = jnp.where(
        massive_Astrometric_Uncertainties != jnp.inf, massive_resids, 999.0
    )

    massive_loglike = jnp.sum(massive_loglike)
    return (
        massive_xs,
        massive_vs,
        massive_calc_RAs,
        massive_calc_Decs,
        massive_resids,
        massive_loglike,
    )

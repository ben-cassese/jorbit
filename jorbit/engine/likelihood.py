import jax
from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, lax
import pickle

from jorbit.data import (
    STANDARD_PLANET_PARAMS,
    STANDARD_ASTEROID_PARAMS,
    STANDARD_PLANET_GMS,
    STANDARD_ASTEROID_GMS,
)


from jorbit.engine.slapshot_integrator import integrate_multiple
from jorbit.engine.utils import on_sky, sky_error


def negative_loglike_single(
    X,
    times,
    planet_params,
    asteroid_params,
    planet_gms,
    asteroid_gms,
    observer_positions,
    ra,
    dec,
    position_uncertainties,
    max_steps=jnp.arange(100),
):
    x = X[:3]
    v = X[3:6]

    x = x[None, :]
    v = v[None, :]

    xs, vs, final_times, success = integrate_multiple(
        xs=x,
        vs=v,
        gms=jnp.array([0]),
        initial_time=times[0],
        final_times=times[1:],
        planet_params=planet_params,
        asteroid_params=asteroid_params,
        planet_gms=planet_gms,
        asteroid_gms=asteroid_gms,
        max_steps=max_steps,
        use_GR=True,
    )

    xs = jnp.concatenate((x[:, None, :], xs), axis=1)
    vs = jnp.concatenate((v[:, None, :], vs), axis=1)

    calc_RAs, calc_Decs = on_sky(
        xs=xs[0],
        vs=vs[0],
        gms=jnp.zeros(len(xs[0])),
        times=times,
        observer_positions=observer_positions,
        planet_params=planet_params,
        asteroid_params=asteroid_params,
        planet_gms=planet_gms,
        asteroid_gms=asteroid_gms,
    )

    err = sky_error(calc_ra=calc_RAs, calc_dec=calc_Decs, true_ra=ra, true_dec=dec)

    sigma2 = position_uncertainties**2
    # p = jnp.log(2 * jnp.pi * sigma2)
    q = -0.5 * jnp.sum(((err**2 / sigma2)))  # + p))
    return -q


def negative_loglike_single_grad(
    X,
    times,
    planet_params,
    asteroid_params,
    planet_gms,
    asteroid_gms,
    observer_positions,
    ra,
    dec,
    position_uncertainties,
    max_steps=jnp.arange(100),
):
    return jax.grad(negative_loglike_single)(
        X,
        times,
        planet_params,
        asteroid_params,
        planet_gms,
        asteroid_gms,
        observer_positions,
        ra,
        dec,
        position_uncertainties,
        max_steps=max_steps,
    )


def weave_free_and_fixed(free_mask, free, fixed):
    """
    Combine free and fixed parameters into a single array.

    This is a helper for prepareloglike_helper_input. Combines the free and fixed parameters
    into a single array, making sure they stay in the correct order so that the
    states and GMs of each particle stay together. These "free" and "fixed" parameters
    can be either positions, velocities, or GMs.

    Parameters:
        free_mask (jnp.ndarray(shape=(N,), dtype=bool)):
            A mask marking parameters are free to vary in a fit/likelihood evaluation.
            True means free, False means fixed.
        free (jnp.ndarray(shape=(M,))):
            The current value of the free parameters.
        fixed (jnp.ndarray(shape=(N-M,))):
            The current value of the fixed parameters.

    Returns:
        jnp.ndarray(shape=(N,)):
            The combined free and fixed parameters, with the free parameters in the correct
            order.

    Examples:

        >>> from jorbit.engine import weave_free_and_fixed
        >>> import jax.numpy as jnp
        >>> weave_free_and_fixed(
        ...     free_mask=jnp.array([True, True, False, True, False]),
        ...     free=jnp.array([1.0, 2.0, 3.0]),
        ...     fixed=jnp.array([4.0, 5.0]),
        ... )
    """
    if len(free_mask) == 0:
        return jnp.empty(free.shape)
    if len(fixed) == 0:
        return free
    if len(free) == 0:
        return fixed

    def scan_func(carry, scan_over):
        free_placed = carry
        i = scan_over
        add_free = free_mask[i]
        x = jnp.where(add_free, free[free_placed], fixed[i - free_placed])
        return free_placed + add_free * 1, x

    return jax.lax.scan(scan_func, 0, jnp.arange(len(free_mask)))[1]


def prepare_loglike_input(
    free_tracer_state_mask,
    free_tracer_xs,
    fixed_tracer_xs,
    free_tracer_vs,
    fixed_tracer_vs,
    free_massive_state_mask,
    free_massive_xs,
    fixed_massive_xs,
    free_massive_vs,
    fixed_massive_vs,
    free_massive_gm_mask,
    free_massive_gms,
    fixed_massive_gms,
    free_planet_gm_mask,
    free_planet_gms,
    fixed_planet_gms,
    free_asteroid_gm_mask,
    free_asteroid_gms,
    fixed_asteroid_gms,
    tracer_particle_times,
    tracer_particle_ras,
    tracer_particle_decs,
    tracer_particle_observer_positions,
    tracer_particle_astrometric_uncertainties,
    massive_particle_times,
    massive_particle_ras,
    massive_particle_decs,
    massive_particle_observer_positions,
    massive_particle_astrometric_uncertainties,
    planet_params,
    asteroid_params,
    max_steps,
    use_GR,
):
    """
    Convert the attributes of a System into a form that can be used by loglike.

    We ultimately want to perform inferences using loglike to evaluate the likelihood of
    our system given some data. However, loglike takes in the full state of the system,
    including bits that we want to remain fixed to certain values and therefore cannot
    influence the likelihood. For examples, we likely do not want to vary the masses of
    the planets included in the ephemeris, or we might want to fix the initial
    state vector to some value and fit for a mass. In these cases, we don't want to
    differentiate w.r.t. these values, so we need a buffer layer to that combines things
    we want to vary with those we want to fix before passing them all off to loglike.
    That's the purpose of this function: to combine all of the parameters contained
    within a System object into a form that's useful for loglike.

    Parameters:
        free_tracer_state_mask (jnp.ndarray(shape=(N,), dtype=bool)):
            A mask marking which of the N tracer particles parameters have state vectors
            which are allowed to vary.
        free_tracer_xs (jnp.ndarray(shape=(M, 3))):
            The current values of the M free tracer particle positions in AU.
        fixed_tracer_xs (jnp.ndarray(shape=(N-M, 3))):
            The current values of the N-M fixed tracer particle positions in AU.
        free_tracer_vs (jnp.ndarray(shape=(M, 3))):
            The current values of the M free tracer particle velocities in AU/day.
        fixed_tracer_vs (jnp.ndarray(shape=(N-M, 3))):
            The current values of the N-M fixed tracer particle velocities in AU/day.
        free_massive_state_mask (jnp.ndarray(shape=(P,), dtype=bool)):
            A mask marking which of the P massive particles parameters have state vectors
            which are allowed to vary.
        free_massive_xs (jnp.ndarray(shape=(Q, 3))):
            The current values of the Q free massive particle positions in AU.
        fixed_massive_xs (jnp.ndarray(shape=(P-Q, 3))):
            The current values of the P-Q fixed massive particle positions in AU.
        free_massive_vs (jnp.ndarray(shape=(Q, 3))):
            The current values of the Q free massive particle velocities in AU/day.
        fixed_massive_vs (jnp.ndarray(shape=(P-Q, 3))):
            The current values of the P-Q fixed massive particle velocities in AU/day.
        free_massive_gm_mask (jnp.ndarray(shape=(P,), dtype=bool)):
            A mask marking which of the P massive particles parameters have GMs which are
            allowed to vary.
        free_massive_gms (jnp.ndarray(shape=(R,))):
            The current values of the R free massive particle GMs in AU^3/day^2.
        fixed_massive_gms (jnp.ndarray(shape=(P-R,))):
            The current values of the P-R fixed massive particle GMs in AU^3/day^2.
        free_planet_gm_mask (jnp.ndarray(shape=(Y,), dtype=bool)):
            A mask marking which of the Y planets from an ephemeris have GMs which are
            allowed to vary
        free_planet_gms (jnp.ndarray(shape=(Z,))):
            The current values of the Z free planet GMs in AU^3/day^2.
        fixed_planet_gms (jnp.ndarray(shape=(Y-Z,))):
            The current values of the Y-Z fixed planet GMs in AU^3/day^2.
        free_asteroid_gm_mask (jnp.ndarray(shape=(W,), dtype=bool)):
            A mask marking which of the W asteroids from an ephemeris have GMs which are
            allowed to vary
        free_asteroid_gms (jnp.ndarray(shape=(X,))):
            The current values of the X free asteroid GMs in AU^3/day^2.
        fixed_asteroid_gms (jnp.ndarray(shape=(W-X,))):
            The current values of the W-X fixed asteroid GMs in AU^3/day^2.
        tracer_particle_times (jnp.ndarray(shape=(N,T))):
            The T times the N tracer particles were observed in TDB JD. Will be a sequence
            of T repetitions of the value "2458849" for particles that were not observed-
            see System for more info.
        tracer_particle_ras (jnp.ndarray(shape=(N,T))):
            The T RAs of the N tracer particles in radians. Will be a sequence of zeros for
            particles that were not observed- see System for more info.
        tracer_particle_decs (jnp.ndarray(shape=(N,T))):
            The T Decs of the N tracer particles in radians. Will be a sequence of zeros for
            particles that were not observed- see System for more info.
        tracer_particle_observer_positions (jnp.ndarray(shape=(N,T,3))):
            The T 3D positions of the observers at each observation time in AU. Will be a
            sequence of 999s for particles that were not observed- see System for more info.
        tracer_particle_astrometric_uncertainties (jnp.ndarray(shape=(N,T))):
            The T astrometric uncertainties of the N tracer particles in arcsec. Will be a
            sequence of infs for particles that were not observed- see System for more info.
        massive_particle_times: (jnp.ndarray(shape=(P,D))):
            Analog of free_particle times for the P massive particles each observed D times.
        massive_particle_ras (jnp.ndarray(shape=(P,D))):
            Analog of free_particle ras for the P massive particles each observed D times.
        massive_particle_decs (jnp.ndarray(shape=(P,D))):
            Analog of free_particle decs for the P massive particles each observed D times.
        massive_particle_observer_positions (jnp.ndarray(shape=(P,D,3))):
            Analog of free_particle observer_positions for the P massive particles each
            observed D times.
        massive_particle_astrometric_uncertainties (jnp.ndarray(shape=(P,D))):
            Analog of free_particle astrometric_uncertainties for the P massive particles
            each observed D times.
        planet_params (Tuple[jnp.ndarray(shape=(P,)), jnp.ndarray(shape=(P,)), jnp.ndarray(shape=(P,Q,3,R))], default=STANDARD_PLANET_PARAMS from jorbit.construct_perturbers):
            The ephemeris describing P massive objects in the solar system. The first
            element is the initial time of the ephemeris in seconds since J2000 TDB. The
            second element is the length of the interval covered by each piecewise chunk of
            the ephemeris in seconds (for DE44x planets, this is 16 days, and for
            asteroids, it's 32 days). The third element contains the Q coefficients of the
            R piecewise chunks of Chebyshev polynomials that make up the ephemeris, in 3
            x,y,z dimensions.
        asteroid_params (Tuple[jnp.ndarray(shape=(W,)), jnp.ndarray(shape=(W,)), jnp.ndarray(shape=(W,Z,3,K))], default=STANDARD_ASTEROID_PARAMS from jorbit.construct_perturbers):
            Same as planet_params but for W asteroids. They are separated only in case
            use_GR=True, in which case the planet perturbations are calculated using the
            PPN formalism while the asteroids are still just Newtonian.
        max_steps (jnp.ndarray(shape=(Z,)), default=jnp.arange(100)):
            Any array of length Z, the maximum number of calls to single_step.
        use_GR (bool, default=False):
            Whether to use the PPN formalism to calculate the gravitational influence of
            the planets. If False, the planets are treated as Newtonian point masses. The
            asteroids are always treated as Newtonian point masses regardless of this flag.


    Returns:
        dict:
            A dictionary containing all of the inputs needed for loglike.


    Examples:

        See the source code for System.loglike

    """
    tracer_xs = weave_free_and_fixed(
        free_tracer_state_mask, free_tracer_xs, fixed_tracer_xs
    )
    tracer_vs = weave_free_and_fixed(
        free_tracer_state_mask, free_tracer_vs, fixed_tracer_vs
    )
    massive_xs = weave_free_and_fixed(
        free_massive_state_mask, free_massive_xs, fixed_massive_xs
    )
    massive_vs = weave_free_and_fixed(
        free_massive_state_mask, free_massive_vs, fixed_massive_vs
    )
    massive_gms = weave_free_and_fixed(
        free_massive_gm_mask, free_massive_gms, fixed_massive_gms
    )
    planet_gms = weave_free_and_fixed(
        free_planet_gm_mask, free_planet_gms, fixed_planet_gms
    )
    asteroid_gms = weave_free_and_fixed(
        free_asteroid_gm_mask, free_asteroid_gms, fixed_asteroid_gms
    )

    return {
        "tracer_particle_xs": tracer_xs,
        "tracer_particle_vs": tracer_vs,
        "tracer_particle_times": tracer_particle_times,
        "tracer_particle_ras": tracer_particle_ras,
        "tracer_particle_decs": tracer_particle_decs,
        "tracer_particle_observer_positions": tracer_particle_observer_positions,
        "tracer_particle_astrometric_uncertainties": (
            tracer_particle_astrometric_uncertainties
        ),
        "massive_particle_xs": massive_xs,
        "massive_particle_vs": massive_vs,
        "massive_particle_gms": massive_gms,
        "massive_particle_times": massive_particle_times,
        "massive_particle_ras": massive_particle_ras,
        "massive_particle_decs": massive_particle_decs,
        "massive_particle_observer_positions": massive_particle_observer_positions,
        "massive_particle_astrometric_uncertainties": (
            massive_particle_astrometric_uncertainties
        ),
        "planet_params": planet_params,
        "asteroid_params": asteroid_params,
        "planet_gms": planet_gms,
        "asteroid_gms": asteroid_gms,
        "max_steps": max_steps,
        "use_GR": use_GR,
    }


def residuals(
    tracer_particle_xs=jnp.empty((0, 3)),
    tracer_particle_vs=jnp.empty((0, 3)),
    tracer_particle_times=jnp.empty((0, 1)),
    tracer_particle_ras=jnp.empty((0, 1)),
    tracer_particle_decs=jnp.empty((0, 1)),
    tracer_particle_observer_positions=jnp.empty((0, 1, 3)),
    tracer_particle_astrometric_uncertainties=jnp.empty((0, 1)),
    massive_particle_xs=jnp.empty((0, 3)),
    massive_particle_vs=jnp.empty((0, 3)),
    massive_particle_gms=jnp.empty((0)),
    massive_particle_times=jnp.empty((0, 1)),
    massive_particle_ras=jnp.empty((0, 1)),
    massive_particle_decs=jnp.empty((0, 1)),
    massive_particle_observer_positions=jnp.empty((0, 1, 3)),
    massive_particle_astrometric_uncertainties=jnp.empty((0, 1)),
    planet_params=STANDARD_PLANET_PARAMS,
    asteroid_params=STANDARD_ASTEROID_PARAMS,
    planet_gms=STANDARD_PLANET_GMS,
    asteroid_gms=STANDARD_ASTEROID_GMS,
    max_steps=jnp.arange(100),
    use_GR=True,
):
    def tracer_false_func(C):
        x_, v_, times, observer_pos, ra, dec, pos_uncertainty = C
        return jnp.zeros(len(times), dtype=jnp.float64)

    def massive_false_func(C):
        ind = C
        return jnp.zeros(len(massive_particle_times[ind]), dtype=jnp.float64)

    def tracer_true_func(C):
        x_, v_, times, observer_pos, ra, dec, pos_uncertainty = C
        x = jnp.concatenate((massive_particle_xs, x_[None, :]))
        v = jnp.concatenate((massive_particle_vs, v_[None, :]))
        xs, vs, final_times, success = integrate_multiple(
            xs=x,
            vs=v,
            gms=jnp.concatenate((massive_particle_gms, jnp.array([0]))),
            initial_time=times[0],
            final_times=times[1:],
            planet_params=planet_params,
            asteroid_params=asteroid_params,
            planet_gms=planet_gms,
            asteroid_gms=asteroid_gms,
            max_steps=max_steps,
            use_GR=use_GR,
        )

        xs = jnp.concatenate((x_[None, None, :], xs[-1][None, :]), axis=1)
        vs = jnp.concatenate((v_[None, None, :], vs[-1][None, :]), axis=1)

        calc_RAs, calc_Decs = on_sky(
            xs=xs[0],
            vs=vs[0],
            gms=jnp.zeros(len(xs[0]), dtype=jnp.float64),
            times=times,
            observer_positions=observer_pos,
            planet_params=planet_params,
            asteroid_params=asteroid_params,
            planet_gms=planet_gms,
            asteroid_gms=asteroid_gms,
        )

        err = sky_error(calc_ra=calc_RAs, calc_dec=calc_Decs, true_ra=ra, true_dec=dec)
        return err

    def tracer_scan_func(carry, scan_over):
        # x_, v_, times, observer_positions, ras, decs, astrometric_uncertainties = scan_over

        # This is a little gross right now- if the very first entry in the list is inf,
        # then there were no observations of this particle and it's not worth integrating/checking error.
        # But, just because the first entry is not zero doesn't mean all of them are-
        # to avoid a ragged array, these are all padded out to the length of whichever particle had the most
        # observations. So, you'll be integrating out to a bunch of dummy times- you only get saved
        # in the final loglike calc since 1/inf = 0, so the bogus times don't contribute to the total loglike.
        # Possibly lots of wasted computation, but hopefully not terrible since a) all the bogus times are the same,
        # so integrating should be instant and b) hopefully all particles have similar number of observations.
        # Worst case scenario would be i.e. one particle has 1,000 observations, 50 particles have 1.
        q = jax.lax.cond(
            scan_over[-1][0] != jnp.inf, tracer_true_func, tracer_false_func, scan_over
        )
        return None, q

    def massive_true_func(C):
        ind = C
        xs, vs, final_times, success = integrate_multiple(
            xs=massive_particle_xs,
            vs=massive_particle_vs,
            gms=massive_particle_gms,
            initial_time=massive_particle_times[ind][0],
            final_times=massive_particle_times[ind][1:],
            planet_params=planet_params,
            asteroid_params=asteroid_params,
            planet_gms=planet_gms,
            asteroid_gms=asteroid_gms,
            max_steps=max_steps,
        )

        xs = jnp.concatenate((massive_particle_xs[:, None, :], xs), axis=1)
        vs = jnp.concatenate((massive_particle_vs[:, None, :], vs), axis=1)

        # Note: here, when correcting for light travel time, we are setting the GMs
        # of the massive particles to zero. Assuming that self-interaction is
        # negligible on this short timescale
        calc_RAs, calc_Decs = on_sky(
            xs=xs[ind],
            vs=vs[ind],
            gms=jnp.zeros(len(xs[ind]), dtype=jnp.float64),
            times=massive_particle_times[ind],
            observer_positions=massive_particle_observer_positions[ind],
            planet_params=planet_params,
            asteroid_params=asteroid_params,
            planet_gms=planet_gms,
            asteroid_gms=asteroid_gms,
        )

        err = sky_error(
            calc_ra=calc_RAs,
            calc_dec=calc_Decs,
            true_ra=massive_particle_ras[ind],
            true_dec=massive_particle_decs[ind],
        )

        return err

    def massive_scan_func(carry, scan_over):
        q = jax.lax.cond(
            massive_particle_astrometric_uncertainties[scan_over][0] != 999.0,
            massive_true_func,
            massive_false_func,
            scan_over,
        )
        return None, q

    # Eventually look into pmapping instead of scanning these
    if len(tracer_particle_astrometric_uncertainties) > 0:
        tracer_contribution = jax.lax.scan(
            tracer_scan_func,
            None,
            (
                tracer_particle_xs,
                tracer_particle_vs,
                tracer_particle_times,
                tracer_particle_observer_positions,
                tracer_particle_ras,
                tracer_particle_decs,
                tracer_particle_astrometric_uncertainties,
            ),
        )[1]
    else:
        tracer_contribution = 0.0

    if len(massive_particle_astrometric_uncertainties) > 0:
        massive_contribution = jax.lax.scan(
            massive_scan_func, None, jnp.arange(len(massive_particle_xs))
        )[1]
    else:
        massive_contribution = 0.0

    return tracer_contribution, massive_contribution


def loglike(
    tracer_particle_xs=jnp.empty((0, 3)),
    tracer_particle_vs=jnp.empty((0, 3)),
    tracer_particle_times=jnp.empty((0, 1)),
    tracer_particle_ras=jnp.empty((0, 1)),
    tracer_particle_decs=jnp.empty((0, 1)),
    tracer_particle_observer_positions=jnp.empty((0, 1, 3)),
    tracer_particle_astrometric_uncertainties=jnp.empty((0, 1)),
    massive_particle_xs=jnp.empty((0, 3)),
    massive_particle_vs=jnp.empty((0, 3)),
    massive_particle_gms=jnp.empty((0)),
    massive_particle_times=jnp.empty((0, 1)),
    massive_particle_ras=jnp.empty((0, 1)),
    massive_particle_decs=jnp.empty((0, 1)),
    massive_particle_observer_positions=jnp.empty((0, 1, 3)),
    massive_particle_astrometric_uncertainties=jnp.empty((0, 1)),
    planet_params=STANDARD_PLANET_PARAMS,
    asteroid_params=STANDARD_ASTEROID_PARAMS,
    planet_gms=STANDARD_PLANET_GMS,
    asteroid_gms=STANDARD_ASTEROID_GMS,
    max_steps=jnp.arange(100),
    use_GR=True,
):
    tracer_residuals, massive_residuals = residuals(
        tracer_particle_xs=tracer_particle_xs,
        tracer_particle_vs=tracer_particle_vs,
        tracer_particle_times=tracer_particle_times,
        tracer_particle_ras=tracer_particle_ras,
        tracer_particle_decs=tracer_particle_decs,
        tracer_particle_observer_positions=tracer_particle_observer_positions,
        tracer_particle_astrometric_uncertainties=tracer_particle_astrometric_uncertainties,
        massive_particle_xs=massive_particle_xs,
        massive_particle_vs=massive_particle_vs,
        massive_particle_gms=massive_particle_gms,
        massive_particle_times=massive_particle_times,
        massive_particle_ras=massive_particle_ras,
        massive_particle_decs=massive_particle_decs,
        massive_particle_observer_positions=massive_particle_observer_positions,
        massive_particle_astrometric_uncertainties=massive_particle_astrometric_uncertainties,
        planet_params=planet_params,
        asteroid_params=asteroid_params,
        planet_gms=planet_gms,
        asteroid_gms=asteroid_gms,
        max_steps=max_steps,
        use_GR=use_GR,
    )

    l = 0

    sigma2 = tracer_particle_astrometric_uncertainties**2
    # p = jnp.log(2 * jnp.pi * sigma2)
    l += -0.5 * jnp.sum(((tracer_residuals**2 / sigma2)))  # + p))

    sigma2 = massive_particle_astrometric_uncertainties**2
    # p = jnp.log(2 * jnp.pi * sigma2)
    l += -0.5 * jnp.sum(((massive_residuals**2 / sigma2)))  # + p))

    return l


def system_negative_loglike(
    free_params,
    fixed_params,
    max_steps,
    use_GR,
):
    params = prepare_loglike_input(
        **free_params, **fixed_params, use_GR=use_GR, max_steps=max_steps
    )
    return -loglike(**params)


# def loglike_helper(
#     tracer_particle_xs=jnp.empty((0, 3)),
#     tracer_particle_vs=jnp.empty((0, 3)),
#     tracer_particle_times=jnp.empty((0, 1)),
#     tracer_particle_ras=jnp.empty((0, 1)),
#     tracer_particle_decs=jnp.empty((0, 1)),
#     tracer_particle_observer_positions=jnp.empty((0, 1, 3)),
#     tracer_particle_astrometric_uncertainties=jnp.empty((0, 1)),
#     massive_particle_xs=jnp.empty((0, 3)),
#     massive_particle_vs=jnp.empty((0, 3)),
#     massive_particle_gms=jnp.empty((0)),
#     massive_particle_times=jnp.empty((0, 1)),
#     massive_particle_ras=jnp.empty((0, 1)),
#     massive_particle_decs=jnp.empty((0, 1)),
#     massive_particle_observer_positions=jnp.empty((0, 1, 3)),
#     massive_particle_astrometric_uncertainties=jnp.empty((0, 1)),
#     planet_params=STANDARD_PLANET_PARAMS,
#     asteroid_params=STANDARD_ASTEROID_PARAMS,
#     planet_gms=STANDARD_PLANET_GMS,
#     asteroid_gms=STANDARD_ASTEROID_GMS,
#     max_steps=jnp.arange(100),
#     use_GR=True,
# ):
#     """loglike_helper(tracer_particle_xs=jnp.empty((0, 3)),tracer_particle_vs=jnp.empty((0, 3)), tracer_particle_times=jnp.empty((0, 1)), tracer_particle_ras=jnp.empty((0, 1)), tracer_particle_decs=jnp.empty((0, 1)), tracer_particle_observer_positions=jnp.empty((0, 1, 3)), tracer_particle_astrometric_uncertainties=jnp.empty((0, 1)), massive_particle_xs=jnp.empty((0, 3)), massive_particle_vs=jnp.empty((0, 3)), massive_particle_gms=jnp.empty((0)), massive_particle_times=jnp.empty((0, 1)), massive_particle_ras=jnp.empty((0, 1)), massive_particle_decs=jnp.empty((0, 1)), massive_particle_observer_positions=jnp.empty((0, 1, 3)), massive_particle_astrometric_uncertainties=jnp.empty((0, 1)), planet_params=STANDARD_PLANET_PARAMS, asteroid_params=STANDARD_ASTEROID_PARAMS, planet_gms=STANDARD_PLANET_GMS, asteroid_gms=STANDARD_ASTEROID_GMS, max_steps=jnp.arange(100), use_GR=True)
#     Calculate the log likelihood of a System given some data.

#     Parameters:
#         tracer_particle_xs (jnp.ndarray(shape=(N, 3), default=jnp.empty((0, 3))):
#             The current values of the N tracer particle positions in AU.
#         tracer_particle_vs (jnp.ndarray(shape=(N, 3), default=jnp.empty((0, 3))):
#             The current values of the N tracer particle velocities in AU/day.
#         tracer_particle_times (jnp.ndarray(shape=(N,T), default=jnp.empty((0, 1))):
#             The T times the N tracer particles were observed in TDB JD. Every particle must
#             have the same number of observations, but padding entries with inf is fine and
#             will not affect the log likelihood.
#         tracer_particle_ras (jnp.ndarray(shape=(N,T), default=jnp.empty((0, 1))):
#             The T RAs of the N tracer particles in radians.
#         tracer_particle_decs (jnp.ndarray(shape=(N,T), default=jnp.empty((0, 1))):
#             The T Decs of the N tracer particles in radians.
#         tracer_particle_observer_positions (jnp.ndarray(shape=(N,T,3), default=jnp.empty((0, 1, 3))):
#             The T 3D positions of the observers at each observation time in AU.
#         tracer_particle_astrometric_uncertainties (jnp.ndarray(shape=(N,T), default=jnp.empty((0, 1))):
#             The T astrometric uncertainties of the N tracer particles in arcsec. To make
#             sure padded observations are ignored, dummy values should be set to "inf". This
#             is so that division by sigma^2 gives zero, which will not contribute to the sum.
#         massive_particle_xs (jnp.ndarray(shape=(P, 3), default=jnp.empty((0, 3))):
#             The current values of the P massive particle positions in AU.
#         massive_particle_vs (jnp.ndarray(shape=(P, 3), default=jnp.empty((0, 3))):
#             The current values of the P massive particle velocities in AU/day.
#         massive_particle_gms (jnp.ndarray(shape=(P,), default=jnp.empty((0,))):
#             The current values of the P massive particle GMs in AU^3/day^2.
#         massive_particle_times (jnp.ndarray(shape=(P,D), default=jnp.empty((0, 1))):
#             The D times the P massive particles were observed in TDB JD. Similar to the
#             tracer particles, every massive particle must have the same number of
#             observations, but padding entries with inf is fine and will not affect the log
#             likelihood.
#         massive_particle_ras (jnp.ndarray(shape=(P,D), default=jnp.empty((0, 1))):
#             The D RAs of the P massive particles in radians.
#         massive_particle_decs (jnp.ndarray(shape=(P,D), default=jnp.empty((0, 1))):
#             The D Decs of the P massive particles in radians.
#         massive_particle_observer_positions (jnp.ndarray(shape=(P,D,3), default=jnp.empty((0, 1, 3))):
#             The D 3D positions of the observers at each observation time in AU.
#         massive_particle_astrometric_uncertainties (jnp.ndarray(shape=(P,D)jnp.empty((0, 1))
#             The D astrometric uncertainties of the P massive particles in arcsec. Similar
#             again to the tracers, use inf to pad out observations that don't exist.
#         planet_params (Tuple[jnp.ndarray(shape=(P,)), jnp.ndarray(shape=(P,)), jnp.ndarray(shape=(P,Q,3,R))], default=STANDARD_PLANET_PARAMS from jorbit.construct_perturbers):
#             The ephemeris describing P massive objects in the solar system. The first
#             element is the initial time of the ephemeris in seconds since J2000 TDB. The
#             second element is the length of the interval covered by each piecewise chunk of
#             the ephemeris in seconds (for DE44x planets, this is 16 days, and for
#             asteroids, it's 32 days). The third element contains the Q coefficients of the
#             R piecewise chunks of Chebyshev polynomials that make up the ephemeris, in 3
#             x,y,z dimensions.
#         asteroid_params (Tuple[jnp.ndarray(shape=(W,)), jnp.ndarray(shape=(W,)), jnp.ndarray(shape=(W,Z,3,K))], default=STANDARD_ASTEROID_PARAMS from jorbit.construct_perturbers):
#             Same as planet_params but for W asteroids. They are separated only in case
#             use_GR=True, in which case the planet perturbations are calculated using the
#             PPN formalism while the asteroids are still just Newtonian.
#         planet_gms (jnp.ndarray(shape=(G,)), default=STANDARD_PLANET_GMS from jorbit.construct_perturbers):
#             The GM values of the included planets in AU^3/day^2. If sum(planet_gms) == 0,
#             the planets are ignored. If sum(planet_gms) > 0 but G != P, there will be
#             problems. To ignore planets, set planet_gms to jnp.array([0.]).
#         asteroid_gms (jnp.ndarray(shape=(H,)), default=STANDARD_ASTEROID_GMS from jorbit.construct_perturbers):
#             Same as planet_gms but for the asteroids. If sum(asteroid_gms) != 0, then
#             H must equal W. To ignore asteroids, set asteroid_gms to jnp.array([0.]).
#         max_steps (jnp.ndarray(shape=(Z,)), default=jnp.arange(100)):
#             Any array of length Z, the maximum number of calls to single_step.
#         use_GR (bool, default=False):
#             Whether to use the PPN formalism to calculate the gravitational influence of
#             the planets. If False, the planets are treated as Newtonian point masses. The
#             asteroids are always treated as Newtonian point masses regardless of this flag.

#     Returns:
#         jnp.ndarray(shape=(1,)):
#             The log likelihood of the system given the data, assuming a Gaussian likelihood
#             with the given (possibly heteroscedastic) astrometric uncertainties.

#     Examples:

#     >>> from jorbit.engine import loglike_helper
#     >>> from jorbit import Observations, Particle
#     >>> from jorbit.construct_perturbers import (
#     ...      STANDARD_PLANET_PARAMS,
#     ...      STANDARD_ASTEROID_PARAMS,
#     ...      STANDARD_PLANET_GMS,
#     ...      STANDARD_ASTEROID_GMS,
#     ...  )
#     >>> import jax
#     >>> jax.config.update("jax_enable_x64", True)
#     >>> import jax.numpy as jnp
#     >>> from astropy.time import Time
#     >>> import astropy.units as u
#     >>> from astropy.coordinates import SkyCoord
#     >>> from astroquery.jplhorizons import Horizons
#     >>> target = 274301  # MBA (274301) Wikipedia
#     >>> time = Time("2023-01-01")
#     >>> times = Time(jnp.linspace(time.jd, time.jd + 40, 6), format="jd")
#     >>> horizons_query = Horizons(
#     ...     id=target,
#     ...     location="500@0",  # set the vector origin to the solar system barycenter
#     ...     epochs=[time.tdb.jd],
#     ... )
#     >>> horizons_vectors = horizons_query.vectors(
#     ...     refplane="earth"
#     ... )  # the refplane argument gives us ICRS-alinged vectors
#     >>> true_x0 = jnp.array(
#     ...     [horizons_vectors[0]["x"], horizons_vectors[0]["y"], horizons_vectors[0]["z"]]
#     ... )
#     >>> true_v0 = jnp.array(
#     ...     [horizons_vectors[0]["vx"], horizons_vectors[0]["vy"], horizons_vectors[0]["vz"]]
#     ... )
#     >>> horizons_query = Horizons(
#     ...     id=target,
#     ...     location="695@399",  # set the observer location to Kitt Peak
#     ...     epochs=[t.jd for t in times],
#     ... )
#     >>> horizons_astrometry = horizons_query.ephemerides(extra_precision=True)
#     >>> horizons_positions = jnp.column_stack(
#     ...     (horizons_astrometry["RA"].data.filled(), horizons_astrometry["DEC"].data.filled())
#     ... )
#     >>> obs = Observations(
#     ...     positions=SkyCoord(
#     ...         horizons_astrometry["RA"].data.filled(),
#     ...         horizons_astrometry["DEC"].data.filled(),
#     ...         unit=u.deg,
#     ...     ),
#     ...     times=times,
#     ...     observatory_locations="kitt peak",
#     ...     astrometric_uncertainties=100 * u.mas,
#     ... )
#     >>> asteroid = Particle(
#     ...     x=true_x0,
#     ...     v=true_v0,
#     ...     time=time,
#     ...     name="wiki",
#     ...     observations=obs,
#     ...     fit_state=True,
#     ...     fit_gm=False,
#     ... )
#     >>> loglike_helper(
#     ...     tracer_particle_xs=asteroid.x[None, :],
#     ...     tracer_particle_vs=asteroid.v[None, :],
#     ...     tracer_particle_times=obs.times[None, :],
#     ...     tracer_particle_ras=obs.ra[None, :],
#     ...     tracer_particle_decs=obs.dec[None, :],
#     ...     tracer_particle_observer_positions=obs.observer_positions[None, :],
#     ...     tracer_particle_astrometric_uncertainties=obs.astrometric_uncertainties[None, :],
#     ...     massive_particle_xs=jnp.empty((0, 3)),
#     ...     massive_particle_vs=jnp.empty((0, 3)),
#     ...     massive_particle_gms=jnp.empty((0)),
#     ...     massive_particle_times=jnp.empty((0, 1)),
#     ...     massive_particle_ras=jnp.empty((0, 1)),
#     ...     massive_particle_decs=jnp.empty((0, 1)),
#     ...     massive_particle_observer_positions=jnp.empty((0, 1, 3)),
#     ...     massive_particle_astrometric_uncertainties=jnp.empty((0, 1)),
#     ...     planet_params=STANDARD_PLANET_PARAMS,
#     ...     asteroid_params=STANDARD_ASTEROID_PARAMS,
#     ...     planet_gms=STANDARD_PLANET_GMS,
#     ...     asteroid_gms=STANDARD_ASTEROID_GMS,
#     ...     max_steps=jnp.arange(100),
#     ...     use_GR=True,
#     ... )
#     """
#     Q = 0.0

#     def false_func(C):
#         return 0.0

#     def tracer_true_func(C):
#         x_, v_, times, observer_pos, ra, dec, pos_uncertainty = C
#         x = jnp.concatenate((massive_particle_xs, x_[None, :]))
#         v = jnp.concatenate((massive_particle_vs, v_[None, :]))
#         xs, vs, final_times, success = integrate_multiple(
#             xs=x,
#             vs=v,
#             gms=jnp.concatenate((massive_particle_gms, jnp.array([0]))),
#             initial_time=times[0],
#             final_times=times[1:],
#             planet_params=planet_params,
#             asteroid_params=asteroid_params,
#             planet_gms=planet_gms,
#             asteroid_gms=asteroid_gms,
#             max_steps=max_steps,
#             use_GR=use_GR,
#         )

#         xs = jnp.concatenate((x_[None, None, :], xs[-1][None, :]), axis=1)
#         vs = jnp.concatenate((v_[None, None, :], vs[-1][None, :]), axis=1)

#         calc_RAs, calc_Decs = on_sky(
#             xs=xs[0],
#             vs=vs[0],
#             gms=jnp.zeros(len(xs[0]), dtype=jnp.float64),
#             times=times,
#             observer_positions=observer_pos,
#             planet_params=planet_params,
#             asteroid_params=asteroid_params,
#             planet_gms=planet_gms,
#             asteroid_gms=asteroid_gms,
#         )

#         err = sky_error(calc_ra=calc_RAs, calc_dec=calc_Decs, true_ra=ra, true_dec=dec)

#         sigma2 = pos_uncertainty**2
#         # p = jnp.log(2 * jnp.pi * sigma2)
#         q = -0.5 * jnp.sum(((err**2 / sigma2)))  # + p))
#         return q

#     def tracer_scan_func(carry, scan_over):
#         # x_, v_, times, observer_positions, ras, decs, astrometric_uncertainties = scan_over

#         # This is a little gross right now- if the very first entry in the list is inf,
#         # then there were no observations of this particle and it's not worth integrating/checking error.
#         # But, just because the first entry is not zero doesn't mean all of them are-
#         # to avoid a ragged array, these are all padded out to the length of whichever particle had the most
#         # observations. So, you'll be integrating out to a bunch of dummy times- you only get saved
#         # in the final loglike calc since 1/inf = 0, so the bogus times don't contribute to the total loglike.
#         # Possibly lots of wasted computation, but hopefully not terrible since a) all the bogus times are the same,
#         # so integrating should be instant and b) hopefully all particles have similar number of observations.
#         # Worst case scenario would be i.e. one particle has 1,000 observations, 50 particles have 1.
#         q = jax.lax.cond(
#             scan_over[-1][0] != jnp.inf, tracer_true_func, false_func, scan_over
#         )
#         return None, q

#     def massive_true_func(C):
#         ind = C
#         xs, vs, final_times, success = integrate_multiple(
#             xs=massive_particle_xs,
#             vs=massive_particle_vs,
#             gms=massive_particle_gms,
#             initial_time=massive_particle_times[ind][0],
#             final_times=massive_particle_times[ind][1:],
#             planet_params=planet_params,
#             asteroid_params=asteroid_params,
#             planet_gms=planet_gms,
#             asteroid_gms=asteroid_gms,
#             max_steps=max_steps,
#         )

#         xs = jnp.concatenate((massive_particle_xs[:, None, :], xs), axis=1)
#         vs = jnp.concatenate((massive_particle_vs[:, None, :], vs), axis=1)

#         # Note: here, when correcting for light travel time, we are setting the GMs
#         # of the massive particles to zero. Assuming that self-interaction is
#         # negligible on this short timescale
#         calc_RAs, calc_Decs = on_sky(
#             xs=xs[ind],
#             vs=vs[ind],
#             gms=jnp.zeros(len(xs[ind]), dtype=jnp.float64),
#             times=massive_particle_times[ind],
#             observer_positions=massive_particle_observer_positions[ind],
#             planet_params=planet_params,
#             asteroid_params=asteroid_params,
#             planet_gms=planet_gms,
#             asteroid_gms=asteroid_gms,
#         )

#         err = sky_error(
#             calc_ra=calc_RAs,
#             calc_dec=calc_Decs,
#             true_ra=massive_particle_ras[ind],
#             true_dec=massive_particle_decs[ind],
#         )

#         sigma2 = massive_particle_astrometric_uncertainties[ind] ** 2
#         # p = jnp.log(2 * jnp.pi * sigma2)
#         q = -0.5 * jnp.sum(((err**2 / sigma2)))  # + p))
#         return q

#     def massive_scan_func(carry, scan_over):
#         q = jax.lax.cond(
#             massive_particle_astrometric_uncertainties[scan_over][0] != 999.0,
#             massive_true_func,
#             false_func,
#             scan_over,
#         )
#         return None, q

#     if len(tracer_particle_astrometric_uncertainties) > 0:
#         tracer_contribution = jax.lax.scan(
#             tracer_scan_func,
#             None,
#             (
#                 tracer_particle_xs,
#                 tracer_particle_vs,
#                 tracer_particle_times,
#                 tracer_particle_observer_positions,
#                 tracer_particle_ras,
#                 tracer_particle_decs,
#                 tracer_particle_astrometric_uncertainties,
#             ),
#         )[1].sum()
#     else:
#         tracer_contribution = 0.0

#     if len(massive_particle_astrometric_uncertainties) > 0:
#         massive_contribution = jax.lax.scan(
#             massive_scan_func, None, jnp.arange(len(massive_particle_xs))
#         )[1].sum()
#     else:
#         massive_contribution = 0.0

#     Q += tracer_contribution
#     Q += massive_contribution

#     return Q

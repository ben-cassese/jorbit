import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.engine.yoshida_integrator import yoshida_integrate_multiple
from jorbit.engine.accelerations import acceleration

########################################################################################
# Portions of the integrator
########################################################################################


def _startup_scan_func(carry, scan_over, constants):
    """
    A portion of the Gauss-Jackson integrator. After some other integrator estimates
    the initial leading and trailing positions, this function refines those estimates
    """
    (inferred_as, little_s, big_S) = carry
    (
        MID_IND,
        dt,
        b_0k,
        a_0k,
        x0,
        v0,
        a0,
        b_front,
        a_front,
        init_planet_xs,
        init_planet_vs,
        init_planet_as,
        init_asteroid_xs,
        gms,
        planet_gms,
        asteroid_gms,
        use_GR,
    ) = constants

    # Calculate s0
    c1_prime = v0 / dt - (inferred_as * b_0k).sum(axis=1)
    c1 = c1_prime + a0 / 2
    little_s_mid = c1_prime

    # Calculate S0
    c2 = x0 / dt**2 - (inferred_as * a_0k).sum(axis=1) + c1
    big_S_mid = c2 - c1

    # Calculate sn
    pair_sums = (
        jnp.column_stack(
            (inferred_as[:, :-1, :][:, None], inferred_as[:, 1:, :][:, None])
        ).sum(axis=1)
        / 2
    )
    pair_sums = jnp.swapaxes(pair_sums, 0, 1)
    lower_s = jax.lax.scan(
        lambda carry, scan_over: (carry - scan_over, carry - scan_over),
        little_s_mid,
        pair_sums[:MID_IND, :, :][::-1],
    )[1][::-1]
    upper_s = jax.lax.scan(
        lambda carry, scan_over: (carry + scan_over, carry + scan_over),
        little_s_mid,
        pair_sums[MID_IND:, :, :],
    )[1]
    little_s = jnp.concatenate(
        (
            jnp.swapaxes(lower_s, 0, 1),
            little_s_mid[:, None, :],
            jnp.swapaxes(upper_s, 0, 1),
        ),
        axis=1,
    )

    # Calculate Sn
    swapped_little_s = jnp.swapaxes(little_s, 0, 1)
    swapped_inferred_as = jnp.swapaxes(inferred_as, 0, 1)
    lower_S = jax.lax.scan(
        lambda carry, scan_over: (
            carry - scan_over[0] + scan_over[1],
            carry - scan_over[0] + scan_over[1],
        ),
        big_S_mid,
        (
            swapped_little_s[1 : MID_IND + 1, :, :][::-1],
            swapped_inferred_as[1 : MID_IND + 1, :, :][::-1] / 2,
        ),
    )[1][::-1]
    upper_S = jax.lax.scan(
        lambda carry, scan_over: (
            carry + scan_over[0] + scan_over[1],
            carry + scan_over[0] + scan_over[1],
        ),
        big_S_mid,
        (
            swapped_little_s[MID_IND:-1, :, :],
            swapped_inferred_as[MID_IND:-1, :, :] / 2,
        ),
    )[1]
    big_S = jnp.concatenate(
        (
            jnp.swapaxes(lower_S, 0, 1),
            big_S_mid[:, None, :],
            jnp.swapaxes(upper_S, 0, 1),
        ),
        axis=1,
    )

    b_terms = (b_front * inferred_as[:, None, :, :]).sum(axis=2)
    a_terms = (a_front * inferred_as[:, None, :, :]).sum(axis=2)

    inferred_vs = dt * (little_s + b_terms)
    inferred_xs = dt**2 * (big_S + a_terms)

    new_acceleration = acceleration(
        xs=inferred_xs,
        vs=inferred_vs,
        gms=gms,
        planet_xs=init_planet_xs,
        planet_vs=init_planet_vs,
        planet_as=init_planet_as,
        asteroid_xs=init_asteroid_xs,
        planet_gms=planet_gms,
        asteroid_gms=asteroid_gms,
        use_GR=use_GR,
    )

    inferred_as = new_acceleration
    return (inferred_as, little_s, big_S), None


def _corrector_scan_func(carry, scan_over, constants):
    """
    Part of the Gauss-Jackson integrator. During each step forwards after startup, this
    is called repeatedly to refine the position estimate. Is called by
    _stepping_scan_func
    """
    # last 3 args are new_little_s, predicted_x, predicted_v
    inferred_as, _, _, _ = carry
    (
        dt,
        little_s,
        b_f,
        a_f,
        frozen_bs,
        frozen_as,
        big_S_last,
        planet_corrector_xs,
        planet_corrector_vs,
        planet_corrector_as,
        asteroid_corrector_xs,
        gms,
        planet_gms,
        asteroid_gms,
        use_GR,
    ) = constants

    new_little_s = little_s + (inferred_as[:, -1, :] + inferred_as[:, -2, :]) / 2
    b_coeffs = b_f * inferred_as[:, -1, :] + frozen_bs
    a_coeffs = a_f * inferred_as[:, -1, :] + frozen_as
    predicted_v = dt * (new_little_s + b_coeffs)
    predicted_x = dt**2 * (big_S_last + a_coeffs)

    refined_a = acceleration(
        xs=predicted_x[:, None, :],
        vs=predicted_v[:, None, :],
        gms=gms,
        planet_xs=planet_corrector_xs,
        planet_vs=planet_corrector_vs,
        planet_as=planet_corrector_as,
        asteroid_xs=asteroid_corrector_xs,
        planet_gms=planet_gms,
        asteroid_gms=asteroid_gms,
        use_GR=use_GR,
    )

    inferred_as = inferred_as.at[:, -1, :].set(refined_a[:, 0, :])

    return (inferred_as, new_little_s, predicted_x, predicted_v), None


def _stepping_scan_func(carry, scan_over, constants):
    """
    Part of the Gauss-Jackson integrator. This is the function that advances the system
    by one time step. It is called repeatedly after startup, and it itself calls
    _corrector_scan_func several times during each step
    """

    planet_xs_step, planet_vs_step, planet_as_step, asteroid_xs_step = scan_over
    (
        dt,
        b_f1,
        a_f1,
        b_f2,
        a_f2,
        frozen_b_jk,
        frozen_a_jk,
        gms,
        planet_gms,
        asteroid_gms,
        use_GR,
    ) = constants

    def false_func(carry):
        return carry, None

    def true_func(carry):
        xs, vs, inferred_as, little_s, big_S_last = carry
        big_S_last = big_S_last + little_s + (inferred_as[:, -1, :] / 2)

        b_terms = (b_f1 * inferred_as).sum(axis=1)
        a_terms = (a_f1 * inferred_as).sum(axis=1)

        predicted_v = dt * (little_s + b_terms + 0.5 * inferred_as[:, -1, :])
        predicted_x = dt**2 * (big_S_last + a_terms)

        predicted_next_a = acceleration(
            xs=predicted_x[:, None, :],
            vs=predicted_v[:, None, :],
            gms=gms,
            planet_xs=planet_xs_step[:, None, :],
            planet_vs=planet_vs_step[:, None, :],
            planet_as=planet_as_step[:, None, :],
            asteroid_xs=asteroid_xs_step[:, None, :],
            planet_gms=planet_gms,
            asteroid_gms=asteroid_gms,
            use_GR=use_GR,
        )

        inferred_as = inferred_as.at[:, :-1, :].set(inferred_as[:, 1:, :])
        inferred_as = inferred_as.at[:, -1, :].set(predicted_next_a[:, 0, :])

        frozen_bs = (frozen_b_jk * inferred_as[:, :-1, :]).sum(axis=1)
        frozen_as = (frozen_a_jk * inferred_as[:, :-1, :]).sum(axis=1)

        planet_corrector_xs = planet_xs_step[:, None, :]
        planet_corrector_vs = planet_vs_step[:, None, :]
        planet_corrector_as = planet_as_step[:, None, :]
        asteroid_corrector_xs = asteroid_xs_step[:, None, :]
        scan_func = jax.tree_util.Partial(
            _corrector_scan_func,
            constants=(
                dt,
                little_s,
                b_f2,
                a_f2,
                frozen_bs,
                frozen_as,
                big_S_last,
                planet_corrector_xs,
                planet_corrector_vs,
                planet_corrector_as,
                asteroid_corrector_xs,
                gms,
                planet_gms,
                asteroid_gms,
                use_GR,
            ),
        )

        inferred_as, new_little_s, predicted_x, predicted_v = jax.lax.scan(
            scan_func, (inferred_as, little_s, predicted_x, predicted_v), None, length=5
        )[0]

        return (predicted_x, predicted_v, inferred_as, new_little_s, big_S_last), None

    return jax.lax.cond(planet_xs_step[0, 0] != 999, true_func, false_func, carry)


########################################################################################
# Actual integrator
########################################################################################


def gj_integrate(
    x0,
    v0,
    gms,
    b_jk,
    a_jk,
    t0,
    tf,
    valid_steps,
    planet_xs,
    planet_vs,
    planet_as,
    asteroid_xs,
    planet_xs_warmup,
    asteroid_xs_warmup,
    dts_warmup,
    warmup_C,
    warmup_D,
    planet_gms,
    asteroid_gms,
    use_GR,
):
    """
    Advance a system of particles to a given epoch using the Gauss-Jackson integrator.

    Parameters:
        x0 (jnp.ndarray(shape=(N, 3))):
            Initial position of N particles in AU
        v0 (jnp.ndarray(shape=(N, 3))):
            Initial velocity of N particles in AU/day
        gms (jnp.ndarray(shape=(N,))):
            The GM values of N particles in AU^3/day^2
        b_jk (jnp.ndarray(shape=(K+2, K+1))):
            The "b_jk" coefficients for the Gauss-Jackson integrator, as defined in
            Berry and Healy 2004 [1]_. K is the order of the integrator. Values are
            precomputed/stored in jorbit.data.constants for orders 8, 10, 12, and 14.
        a_jk (jnp.ndarray(shape=(K+2, K+1))):
            The "a_jk" coefficients for the Gauss-Jackson integrator, as defined in
            Berry and Healy 2004 [1]_. Same as b_jk, these have been precomputed.
        t0 (float):
            The initial time in TDB JD
        tf (float):
            The final time in TDB JD
        valid_steps (int):
            The number of jumps to take between t0 and tf. Unless the array is padded,
            this should be S + K/2, the same as the shape of the second axis of
            planet_xs, planet_vs, etc. This is called out separately to enable padded
            planet positions.
        planet_xs (jnp.ndarray(shape=(M, S + K/2, 3))):
            The 3D positions of M planets. Each planet has S + K/2 positions, where S
            is the number of subpositions between t0 and tf. The first K/2 positions are
            *before* the initial time and are needed to warm up the integrator. Between
            t0 and tf, the integrator will take S-1 jumps: dt=(tf-t0)/(S - (K/2) - 1)
        planet_vs (jnp.ndarray(shape=(M, S + K/2, 3))):
            The 3D velocities of M planets. Same as planet_xs, but for velocities.
        planet_as (jnp.ndarray(shape=(M, S + K/2, 3))):
            The 3D accelerations of M planets. Same as planet_xs, but for accelerations.
        asteroid_xs (jnp.ndarray(shape=(P, S + K/2, 3))):
            The 3D positions of P asteroids. Same as planet_xs, but for asteroids.
        planet_xs_warmup (jnp.ndarray(shape=(2, K/2, M, Q, 3))):
            The 3D positions of M planets. Axis 0 is for the forward/backward warmup
            steps. Axis 1, K is the order of the GJ integrator, since that's how many
            steps you need to take forwards and backwards during warmup. Axis 2 is for
            the M planets. Axis 3 is for the Q substeps taken for each of the K
            integration steps. Axis 4 is for the {x,y,z} dimensions, in AU
        asteroid_xs_warmup (jnp.ndarray(shape=(2, K/2, P, Q, 3))):
            The 3D positions of P asteroids. Same as planet_xs_warmup, but for
            asteroids.
        dts_warmup (jnp.ndarray(shape=(2, K/2))):
            The time steps used during warmup. Axis 0 is for forward/backward warmup
            steps. Axis 1, K is the order of the GJ integrator, since that's how many
            steps you need to take forwards and backwards during warmup. See
            jorbit.engine.yoshida_integrator.yoshida_integrate_multiple for more
        warmup_C (jnp.ndarray):
            The C coefficients for the Yoshida leapfrog integrator used to warm up the
            larger Gauss-Jackson integor. See jorbit.engine.yoshida_integrator for more,
            these are precomputed for orders 4, 6, and 8 in joirbit.data.constants
        warmup_D (jnp.ndarray):
            The D coefficients for the Yoshida leapfrog integrator used to warm up the
            larger Gauss-Jackson integor. See jorbit.engine.yoshida_integrator for more,
            these are precomputed for orders 4, 6, and 8 in joirbit.data.constants
        planet_gms (jnp.ndarray(shape=(M,))):
            The GM values of M planets in AU^3/day^2
        asteroid_gms (jnp.ndarray(shape=(P,))):
            The GM values of P asteroids in AU^3/day^2
        use_GR (bool):
            Whether or not to use general relativity corrections

    Returns:
        Tuple[jnp.ndarray(shape=(N, 3)), jnp.ndarray(shape=(N, 3))]:
        xs (jnp.ndarray(shape=(N, 3))):
            The final positions of N particles in AU
        vs (jnp.ndarray(shape=(N, 3))):
            The final velocities of N particles in AU/day

    Examples:

        Integrate a main belt asteroid forwards for 2 months:

        >>> import jax.numpy as jnp
        >>> import astropy.units as u
        >>> from astropy.time import Time
        >>> from astroquery.jplhorizons import Horizons
        >>> from jorbit.data.constants import Y8_C, Y8_D, GJ14_A, GJ14_B
        >>> from jorbit.engine.gauss_jackson_integrator import prep_gj_integrator_single, gj_integrate
        >>> from jorbit.data import STANDARD_PLANET_PARAMS, STANDARD_ASTEROID_PARAMS, STANDARD_PLANET_GMS, STANDARD_ASTEROID_GMS
        >>> times = Time(['2023-04-08', '2023-06-08'])
        >>> target = 274301  # MBA (274301) Wikipedia
        >>> horizons_query = Horizons(
        ...     id=target,
        ...     location="500@0",
        ...     epochs=[t.tdb.jd for t in times],
        ... )
        >>> vectors = horizons_query.vectors(refplane="earth")
        >>> x0 = jnp.array([vectors[0]["x"], vectors[0]["y"], vectors[0]["z"]])
        >>> v0 = jnp.array([vectors[0]["vx"], vectors[0]["vy"], vectors[0]["vz"]])
        >>> xf = jnp.array([vectors[1]["x"], vectors[1]["y"], vectors[1]["z"]])
        >>> vf = jnp.array([vectors[1]["vx"], vectors[1]["vy"], vectors[1]["vz"]])
        >>> (
        ...     planet_xs,
        ...     planet_vs,
        ...     planet_as,
        ...     asteroid_xs,
        ...     planet_xs_warmup,
        ...     asteroid_xs_warmup,
        ...     dts_warmup,
        ... ) = prep_gj_integrator_single(
        ...     t0=times[0].tdb.jd,
        ...     tf=times[1].tdb.jd,
        ...     jumps=9,
        ...     a_jk=GJ14_A,
        ...     planet_params=STANDARD_PLANET_PARAMS,
        ...     asteroid_params=STANDARD_ASTEROID_PARAMS,
        ... )
        >>> calc_xf, calc_vf = gj_integrate(
        ...     x0=jnp.array([x0]),
        ...     v0=jnp.array([v0]),
        ...     gms=jnp.array([0.0]),
        ...     b_jk=GJ14_B,
        ...     a_jk=GJ14_A,
        ...     t0=times[0].tdb.jd,
        ...     tf=times[1].tdb.jd,
        ...     planet_xs=planet_xs,
        ...     planet_vs=planet_vs,
        ...     planet_as=planet_as,
        ...     asteroid_xs=asteroid_xs,
        ...     planet_xs_warmup=planet_xs_warmup,
        ...     asteroid_xs_warmup=asteroid_xs_warmup,
        ...     dts_warmup=dts_warmup,
        ...     warmup_C=Y8_C,
        ...     warmup_D=Y8_D,
        ...     planet_gms=STANDARD_PLANET_GMS,
        ...     asteroid_gms=STANDARD_ASTEROID_GMS,
        ...     use_GR=True,
        ... )
        >>> print(jnp.linalg.norm(calc_xf - xf) * u.au.to(u.m))
        >>> print(jnp.linalg.norm(calc_vf - vf) * (u.au/u.day).to(u.m/u.s))

    References:
        .. [1] "Implementation of Gauss-Jackson Integration for Orbit Propagation": https://doi.org/10.1007/BF03546367
        .. [2] "Construction of higher order symplectic integrators": https://doi.org/10.1016/0375-9601(90)90092-3

    """

    MID_IND = int((a_jk.shape[1] - 1) / 2)
    dt = (tf - t0) / (valid_steps - (MID_IND + 1))

    ####################################################################################
    # Initial integration to get leading/trailing points
    ####################################################################################

    backwards_x, backwards_v = yoshida_integrate_multiple(
        x0=x0,
        v0=v0,
        gms=gms,
        dts=dts_warmup[0],
        planet_xs=planet_xs_warmup[0],
        asteroid_xs=asteroid_xs_warmup[0],
        planet_gms=planet_gms,
        asteroid_gms=asteroid_gms,
        C=warmup_C,
        D=warmup_D,
    )

    forwards_x, forwards_v = yoshida_integrate_multiple(
        x0=x0,
        v0=v0,
        gms=gms,
        dts=dts_warmup[1],
        planet_xs=planet_xs_warmup[1],
        asteroid_xs=asteroid_xs_warmup[1],
        planet_gms=planet_gms,
        asteroid_gms=asteroid_gms,
        C=warmup_C,
        D=warmup_D,
    )

    inferred_xs = jnp.column_stack(
        [backwards_x[:, ::-1, :], x0[:, None, :], forwards_x]
    )
    inferred_vs = jnp.column_stack(
        [backwards_v[:, ::-1, :], v0[:, None, :], forwards_v]
    )

    inferred_as = acceleration(
        xs=inferred_xs,
        vs=inferred_vs,
        gms=gms,
        planet_xs=planet_xs[:, : 2 * MID_IND + 1, :],
        planet_vs=planet_vs[:, : 2 * MID_IND + 1, :],
        planet_as=planet_as[:, : 2 * MID_IND + 1, :],
        asteroid_xs=asteroid_xs[:, : 2 * MID_IND + 1, :],
        planet_gms=planet_gms,
        asteroid_gms=asteroid_gms,
        use_GR=use_GR,
    )

    a0 = inferred_as[:, MID_IND, :]

    ####################################################################################
    # Refine those guesses
    ####################################################################################

    b_0k = b_jk[MID_IND][None, :, None]
    a_0k = a_jk[MID_IND][None, :, None]
    b_front = b_jk[None, :-1, :, None]
    a_front = a_jk[None, :-1, :, None]
    init_planet_xs = planet_xs[:, : 2 * MID_IND + 1, :]
    init_planet_vs = planet_vs[:, : 2 * MID_IND + 1, :]
    init_planet_as = planet_as[:, : 2 * MID_IND + 1, :]
    init_asteroid_xs = asteroid_xs[:, : 2 * MID_IND + 1, :]

    scan_func = jax.tree_util.Partial(
        _startup_scan_func,
        constants=(
            MID_IND,
            dt,
            b_0k,
            a_0k,
            x0,
            v0,
            a0,
            b_front,
            a_front,
            init_planet_xs,
            init_planet_vs,
            init_planet_as,
            init_asteroid_xs,
            gms,
            planet_gms,
            asteroid_gms,
            use_GR,
        ),
    )
    inferred_as, little_s, big_S = jax.lax.scan(
        scan_func,
        (inferred_as, jnp.zeros_like(inferred_as), jnp.zeros_like(inferred_as)),
        None,
        length=5,
    )[0]
    little_s = little_s[:, -1, :]
    big_S_last = big_S[:, -1, :]

    ####################################################################################
    # Step forwards
    ####################################################################################
    b_f1 = b_jk[None, -1, :, None]
    a_f1 = a_jk[None, -1, :, None]
    b_f2 = b_jk[None, -2, -1, None]
    a_f2 = a_jk[None, -2, -1, None]
    frozen_b_jk = b_jk[None, -2, :-1, None]
    frozen_a_jk = a_jk[None, -2, :-1, None]

    scan_func = jax.tree_util.Partial(
        _stepping_scan_func,
        constants=(
            dt,
            b_f1,
            a_f1,
            b_f2,
            a_f2,
            frozen_b_jk,
            frozen_a_jk,
            gms,
            planet_gms,
            asteroid_gms,
            use_GR,
        ),
    )

    swapped_planet_xs = jnp.swapaxes(planet_xs[:, 2 * MID_IND + 1 :, :], 0, 1)
    swapped_planet_vs = jnp.swapaxes(planet_vs[:, 2 * MID_IND + 1 :, :], 0, 1)
    swapped_planet_as = jnp.swapaxes(planet_as[:, 2 * MID_IND + 1 :, :], 0, 1)
    swapped_asteroid_xs = jnp.swapaxes(asteroid_xs[:, 2 * MID_IND + 1 :, :], 0, 1)

    b_terms = (b_f1 * inferred_as).sum(axis=1)
    a_terms = (a_f1 * inferred_as).sum(axis=1)
    predicted_v = dt * (little_s + b_terms)
    predicted_x = dt**2 * (big_S_last + a_terms)

    # last 3 args are inferred_as, new_little_s, big_S_last
    predicted_x, predicted_v, _, _, _ = jax.lax.scan(
        scan_func,
        (predicted_x, predicted_v, inferred_as, little_s, big_S_last),
        (
            swapped_planet_xs,
            swapped_planet_vs,
            swapped_planet_as,
            swapped_asteroid_xs,
        ),
    )[0]

    return predicted_x, predicted_v


def gj_integrate_multiple(
    x0,
    v0,
    gms,
    b_jk,
    a_jk,
    t0,
    times,
    valid_steps,
    planet_xs,
    planet_vs,
    planet_as,
    asteroid_xs,
    planet_xs_warmup,
    asteroid_xs_warmup,
    dts_warmup,
    warmup_C,
    warmup_D,
    planet_gms,
    asteroid_gms,
    use_GR,
):
    """
    Integrate to multiple epochs using the Gauss-Jackson integrator

    Just a wrapper for gj_integrate that scans across multiple epochs, so long as number
    of jumps between all of them are the same.

    Parameters:
        x0 (jnp.ndarray(shape=(N, 3))):
            Initial position of N particles in AU
        v0 (jnp.ndarray(shape=(N, 3))):
            Initial velocity of N particles in AU/day
        gms (jnp.ndarray(shape=(N,))):
            The GM values of N particles in AU^3/day^2
        b_jk (jnp.ndarray(shape=(K+2, K+1))):
            The "b_jk" coefficients for the Gauss-Jackson integrator, as defined in
            Berry and Healy 2004 [1]_. K is the order of the integrator. Values are
            precomputed/stored in jorbit.data.constants for orders 8, 10, 12, 14,
            and 16.
        a_jk (jnp.ndarray(shape=(K+2, K+1))):
            The "a_jk" coefficients for the Gauss-Jackson integrator, as defined in
            Berry and Healy 2004 [1]_. Same as b_jk, these have been precomputed
        t0 (float):
            The initial time in TDB JD
        times (jnp.ndarray(shape=(S,))):
            The epochs to integrate to in TDB JD
        valid_steps (jnp.ndarray(shape=(R,))):
            The number planet_xs, planet_vs, etc to pay attention to between each epoch.
            Unless dealing with padded arrays, this should be J + K/2, the same as the
            third axis of planet_xs. If dealing with padded arrays however, setting this
            to something lower will effectively mask out the padded values.
        planet_xs (jnp.ndarray(shape=(S, M, J + K/2, 3))):
            The 3D positions of M planets. Each planet has J + K/2 positions, per
            integration epoch, and there are S epochs. The first K/2 positions of each
            epoch are *before* the epoch and are needed to warm up the integrator- see
            jorbit.engine.gauss_jackson_integrator.gj_integrate for more
        planet_vs (jnp.ndarray(shape=(S, M, J + K/2, 3))):
            The 3D velocities of M planets. Same as planet_xs, but for velocities. Units
            are AU/day
        planet_as (jnp.ndarray(shape=(S, M, J + K/2, 3))):
            The 3D accelerations of M planets. Same as planet_xs, but for accelerations.
            Units are AU/day^2
        asteroid_xs (jnp.ndarray(shape=(S, P, J + K/2, 3))):
            The 3D positions of P asteroids. Same as planet_xs, but for asteroids
        planet_xs_warmup (jnp.ndarray(shape=(S, 2, K/2, M, Q, 3))):
            The 3D positions of M planets during the integrator warmup time. Axis 0
            marks the epochs. Axis 1 is for the backwards and forwards positions. Axis
            2, K is the order of the GJ integrator, since that's how many warmup steps
            you need. Axis 3 is for the M planets. Axis 4 is for the Q substeps taken
            to reach each of the warmup positions. Axis 5 is for the {x,y,z} dimensions.
            Units are AU
        asteroid_xs_warmup (jnp.ndarray(shape=(S, 2, K/2, P, Q, 3))):
            The 3D positions of P asteroids during the integrator warmup time. Same as
            planet_xs_warmup, but for asteroids
        dts_warmup (jnp.ndarray(shape=(S, 2, K/2))):
            The length of the time steps used to reach each of the warmup positions.
            Axis 0 marks the epochs. Axis 1 is for the backwards and forwards positions.
            Axis 2, K is the order of the GJ integrator, since that's how many warmup
            steps are taken. Units are days
        warmup_C (jnp.ndarray):
            The C coefficients for the Yoshida leapfrog integrator used to warm up the
            larger Gauss-Jackson integor. See jorbit.engine.yoshida_integrator for more,
            these are precomputed for orders 4, 6, and 8 in joirbit.data.constants
        warmup_D (jnp.ndarray):
            The D coefficients for the Yoshida leapfrog integrator used to warm up the
            larger Gauss-Jackson integor. See jorbit.engine.yoshida_integrator for more,
            these are precomputed for orders 4, 6, and 8 in joirbit.data.constants
        planet_gms (jnp.ndarray(shape=(M,))):
            The GM values of the M planets in AU^3/day^2
        asteroid_gms (jnp.ndarray(shape=(P,))):
            The GM values of the P asteroids in AU^3/day^2
        use_GR (bool):
            Whether or not to use general relativity corrections

    Returns:
        Tuple[jnp.ndarray(shape=(N, S, 3)), jnp.ndarray(shape=(N, S, 3))]:
        xs (jnp.ndarray(shape=(N, S, 3))):
            The final positions of N particles at the S epochs in AU
        vs (jnp.ndarray(shape=(N, S, 3))):
            The final velocities of N particles at the S epochs in AU/day

    Examples:

        Integrate a main belt asteroid forwards by a few months:

        >>> import jax
        >>> jax.config.update("jax_enable_x64", True)
        >>> import jax.numpy as jnp
        >>> from astropy.time import Time
        >>> from astropy import units as u
        >>> from astroquery.jplhorizons import Horizons
        >>> from jorbit.data.constants import GJ14_A, GJ14_B, Y8_C, Y8_D
        >>> from jorbit.data import (
        ...     STANDARD_PLANET_PARAMS,
        ...     STANDARD_ASTEROID_PARAMS,
        ...     STANDARD_PLANET_GMS,
        ...     STANDARD_ASTEROID_GMS,
        ... )
        >>> from jorbit.engine.gauss_jackson_integrator import (
        ...     prep_gj_integrator_multiple,
        ...     gj_integrate_multiple,
        ... )
        >>> times = Time(["2023-04-08", "2023-06-08", "2023-10-01"])
        >>> target = 274301  # MBA (274301) Wikipedia
        >>> horizons_query = Horizons(
        ...     id=target,
        ...     location="500@0",  # set the vector origin to the solar system barycenter
        ...     epochs=[t.tdb.jd for t in times],
        ... )
        >>> vectors = horizons_query.vectors(refplane="earth")
        >>> x0 = jnp.array([vectors[0]["x"], vectors[0]["y"], vectors[0]["z"]])
        >>> v0 = jnp.array([vectors[0]["vx"], vectors[0]["vy"], vectors[0]["vz"]])
        >>> xi = jnp.array([vectors[1]["x"], vectors[1]["y"], vectors[1]["z"]])
        >>> vi = jnp.array([vectors[1]["vx"], vectors[1]["vy"], vectors[1]["vz"]])
        >>> xf = jnp.array([vectors[2]["x"], vectors[2]["y"], vectors[2]["z"]])
        >>> vf = jnp.array([vectors[2]["vx"], vectors[2]["vy"], vectors[2]["vz"]])
        >>> (
        ...     planet_xs,
        ...     planet_vs,
        ...     planet_as,
        ...     asteroid_xs,
        ...     planet_xs_warmup,
        ...     asteroid_xs_warmup,
        ...     dts_warmup,
        ... ) = prep_gj_integrator_multiple(
        ...     t0=times[0].tdb.jd,
        ...     times=times[1:].tdb.jd,
        ...     jumps=16,
        ...     a_jk=GJ14_A,
        ...     planet_params=STANDARD_PLANET_PARAMS,
        ...     asteroid_params=STANDARD_ASTEROID_PARAMS,
        ... )
        >>> x, v = gj_integrate_multiple(
        ...     x0=jnp.array([x0]),
        ...     v0=jnp.array([v0]),
        ...     gms=jnp.array([0.0]),
        ...     b_jk=GJ14_B,
        ...     a_jk=GJ14_A,
        ...     t0=times[0].tdb.jd,
        ...     times=times[1:].tdb.jd,
        ...     planet_xs=planet_xs,
        ...     planet_vs=planet_vs,
        ...     planet_as=planet_as,
        ...     asteroid_xs=asteroid_xs,
        ...     planet_xs_warmup=planet_xs_warmup,
        ...     asteroid_xs_warmup=asteroid_xs_warmup,
        ...     dts_warmup=dts_warmup,
        ...     warmup_C=Y8_C,
        ...     warmup_D=Y8_D,
        ...     planet_gms=STANDARD_PLANET_GMS,
        ...     asteroid_gms=STANDARD_ASTEROID_GMS,
        ...     use_GR=True,
        ... )
        >>> print(jnp.linalg.norm(x[0, 0] - xi) * u.au.to(u.m))
        >>> print(jnp.linalg.norm(x[0, 1] - xf) * u.au.to(u.m))

    """

    def scan_func(carry, scan_over):
        x, v = gj_integrate(
            x0=carry[0],
            v0=carry[1],
            gms=gms,
            b_jk=b_jk,
            a_jk=a_jk,
            t0=carry[2],
            tf=scan_over[0],
            valid_steps=scan_over[1],
            planet_xs=scan_over[2],
            planet_vs=scan_over[3],
            planet_as=scan_over[4],
            asteroid_xs=scan_over[5],
            planet_xs_warmup=scan_over[6],
            asteroid_xs_warmup=scan_over[7],
            dts_warmup=scan_over[8],
            warmup_C=warmup_C,
            warmup_D=warmup_D,
            planet_gms=planet_gms,
            asteroid_gms=asteroid_gms,
            use_GR=use_GR,
        )
        return (x, v, scan_over[0]), (x, v)

    x, v = jax.lax.scan(
        scan_func,
        (x0, v0, t0),
        (
            times,
            valid_steps,
            planet_xs,
            planet_vs,
            planet_as,
            asteroid_xs,
            planet_xs_warmup,
            asteroid_xs_warmup,
            dts_warmup,
        ),
    )[1]

    return jnp.swapaxes(x, 0, 1), jnp.swapaxes(v, 0, 1)


def gj_integrate_multiple_ragged(
    x0,
    v0,
    gms,
    t0,
    chunks,
    coeffs_dict,
    Planet_Xs,
    Planet_Vs,
    Planet_As,
    Asteroid_Xs,
    Planet_Xs_Warmup,
    Asteroid_Xs_Warmup,
    Dts_Warmup,
    warmup_C,
    warmup_D,
    planet_gms,
    asteroid_gms,
    use_GR,
):
    """
    An alternative to gj_integrate_multiple that works with ragged arrays.

    gj_integrate_multiple requires that every integration between epochs is done in the
    same number of steps, even if some are padded/dummy steps. This lets us use
    jax.lax.scan, which enables is much faster to jit compile. However, it requires a
    lot of memory to store those padded values, and is slower than it needs to be when
    scanning over those. In this alternative version, inputs are given as lists of
    varying length arrays. This takes *a long time to compile*, but ends up running ~2x
    faster than gj_integrate multiple. Leaving this in for now since it might be useful
    for intensive applications, but in most cases gj_integrate_multiple should be used.

    Parameters:
        x0 (jnp.ndarray(shape=(N, 3))):
            Initial position of N particles in AU
        v0 (jnp.ndarray(shape=(N, 3))):
            Initial velocity of N particles in AU/day
        gms (jnp.ndarray(shape=(N,))):
            The GM values of the N particles in AU^3/day^2
        t0 (float):
            The initial time in TDB JD
        chunks (List[Dict]):
            A list of dictionaries, each of which contains the following keys:
                "integrator order" (int):
                    The order of the Gauss-Jackson integrator to use for this chunk
                "times" (jnp.ndarray(shape=(S,))):
                    The epochs to integrate to in TDB JD
                "jumps per integration" (int):
                    The number planet_xs, planet_vs, etc to pay attention to between
                    each epoch. Unless dealing with padded arrays, this should be J +
                    K/2, the same as the third axis of planet_xs. If dealing with
                    padded arrays however, setting this to something lower will
                    effectively mask out the padded values. See
                    jorbit.engine.utils.prep_uneven_GJ_integrator for more
        coeffs_dict (Dict):
            A dictionary containing the "a_jk" and "b_jk" coefficients for the
            different orders of the Gauss-Jackson integrator. The structure is:
                {"a_jk" : {(str(order)): coeffs}, "b_jk" : {(str(order)): coeffs}}
        planet_xs: (List[jnp.ndarray()]):


    """
    Xs = jnp.zeros((x0.shape[0], len(chunks), 3))
    Vs = jnp.zeros((x0.shape[0], len(chunks), 3))

    x, v = x0[:, None, :], v0[:, None, :]
    t = t0

    # for i in range(len(chunks)):
    for i in range(len(chunks)):
        x, v = gj_integrate_multiple(
            x0=x[:, -1, :],
            v0=v[:, -1, :],
            gms=gms,
            a_jk=coeffs_dict["a_jk"][chunks[i]["integrator order"]],
            b_jk=coeffs_dict["b_jk"][chunks[i]["integrator order"]],
            t0=t,
            times=chunks[i]["times"],
            planet_xs=Planet_Xs[i],
            planet_vs=Planet_Vs[i],
            planet_as=Planet_As[i],
            asteroid_xs=Asteroid_Xs[i],
            planet_xs_warmup=Planet_Xs_Warmup[i],
            asteroid_xs_warmup=Asteroid_Xs_Warmup[i],
            dts_warmup=Dts_Warmup[i],
            warmup_C=warmup_C,
            warmup_D=warmup_D,
            planet_gms=planet_gms,
            asteroid_gms=asteroid_gms,
            use_GR=use_GR,
        )
        t = chunks[i]["times"][-1]
    return x

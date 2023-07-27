import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.experimental.ode import odeint

from jorbit.engine.ephemeris import planet_state
from jorbit.engine.accelerations import acceleration
from jorbit.data.constants import EPSILON

########################################################################################
# Helper functions
########################################################################################


def _ode_acceleration(state, dt, args):
    gms, planet_params, asteroid_params, planet_gms, asteroid_gms, use_GR, t0, flip = (
        args
    )
    t = jnp.where(flip, t0 - dt, t0 + dt)
    planet_xs, planet_vs, planet_as = planet_state(
        planet_params=planet_params,
        times=jnp.array([t]),
        velocity=True,
        acceleration=True,
    )

    asteroid_xs, _, _ = planet_state(
        planet_params=asteroid_params,
        times=jnp.array([t]),
        velocity=False,
        acceleration=False,
    )
    # print(state['x'].shape)
    # print(state['x'][:,None,:].shape)
    acc = acceleration(
        xs=state["x"][:, None, :],
        vs=state["v"][:, None, :],
        gms=gms,
        planet_xs=planet_xs,
        planet_vs=planet_vs,
        planet_as=planet_as,
        asteroid_xs=asteroid_xs,
        planet_gms=planet_gms,
        asteroid_gms=asteroid_gms,
        use_GR=use_GR,
    )

    return {"x": state["v"], "v": acc}


def _startup_scan_func(carry, scan_over, constants):
    # variables, constants = carry
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

    # #jax.debug.print(f"b_0k: {b_0k.shape}")
    # #jax.debug.print(f"inferred_as: {inferred_as.shape}")
    # #jax.debug.print(f"v0: {v0.shape}")
    # ##jax.debug.print(f"dt: {dt.shape}")
    c1_prime = v0 / dt - (inferred_as * b_0k).sum(axis=1)
    # #jax.debug.print(f"c1_prime: {c1_prime.shape}")
    # #jax.debug.print(f"a0: {a0.shape}")
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
    # #jax.debug.print(f"pair_sums: {pair_sums.shape}")
    # #jax.debug.print(f"little_s_mid: {little_s_mid.shape}")
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
    # #jax.debug.print(f"lower_s: {lower_s.shape}")
    # #jax.debug.print(f"upper_s: {upper_s.shape}")
    little_s = jnp.concatenate(
        (
            jnp.swapaxes(lower_s, 0, 1),
            little_s_mid[:, None, :],
            jnp.swapaxes(upper_s, 0, 1),
        ),
        axis=1,
    )
    # #jax.debug.print(f"little_s: {little_s.shape}")

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
    # #jax.debug.print(f"lower_S: {lower_S.shape}")

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
    # #jax.debug.print(f"upper_S: {upper_S.shape}")
    big_S = jnp.concatenate(
        (
            jnp.swapaxes(lower_S, 0, 1),
            big_S_mid[:, None, :],
            jnp.swapaxes(upper_S, 0, 1),
        ),
        axis=1,
    )
    # #jax.debug.print(f"big_S: {big_S.shape}")

    b_terms = (b_front * inferred_as[:, None, :, :]).sum(axis=1)
    a_terms = (a_front * inferred_as[:, None, :, :]).sum(axis=1)

    inferred_vs = dt * (little_s + b_terms)
    inferred_xs = dt**2 * (big_S + a_terms)

    # #jax.debug.print(f"inferred_vs: {inferred_vs.shape}")
    # #jax.debug.print(f"inferred_xs: {inferred_xs.shape}")

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

    # #jax.debug.print("max diff: {m}", m=jnp.max(jnp.abs(new_acceleration - inferred_as)))

    inferred_as = new_acceleration
    return (inferred_as, little_s, big_S), None


def _corrector_scan_func(carry, scan_over, constants):
    inferred_as, _, _ = carry
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
    #         # print(inferred_as[-1,0])
    new_little_s = little_s + (inferred_as[:, -1, :] + inferred_as[:, -2, :]) / 2
    b_coeffs = b_f * inferred_as[:, -1, :] + frozen_bs
    a_coeffs = a_f * inferred_as[:, -1, :] + frozen_as
    predicted_v = dt * (little_s + b_coeffs)
    predicted_x = dt**2 * (big_S_last + a_coeffs)

    # jax.debug.print("predicted_v: {v}", v=predicted_v.shape)
    # jax.debug.print("predicted_x: {x}", x=predicted_x.shape)

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
    # jax.debug.print("refined_a: {a}", a=refined_a.shape)
    inferred_as = inferred_as.at[:, -1, :].set(refined_a[:, 0, :])

    return (inferred_as, new_little_s, predicted_x), None


def _stepping_scan_func(carry, scan_over, constants):
    _, inferred_as, little_s, big_S_last = carry
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

    #     # print(f"iteration {i}")
    #     # predict
    big_S_last = big_S_last + little_s + (inferred_as[:, -1, :] / 2)
    #     # print(big_S_last)

    b_terms = (b_f1 * inferred_as).sum(axis=1)
    a_terms = (a_f1 * inferred_as).sum(axis=1)
    # jax.debug.print("b_terms: {b}", b=b_terms.shape)
    # jax.debug.print("a_terms: {a}", a=a_terms.shape)
    # jax.debug.print("little_s: {l}", l=little_s.shape)
    # jax.debug.print("big_S_last: {b}", b=big_S_last.shape)

    predicted_v = dt * (little_s + b_terms)
    predicted_x = dt**2 * (big_S_last + a_terms)

    # jax.debug.print("predicted_v: {v}", v=predicted_v.shape)
    # jax.debug.print("predicted_x: {x}", x=predicted_x.shape)

    # jax.debug.print("planet_xs_step: {x}", x=planet_xs_step.shape)
    # jax.debug.print("reshaped: {x}", x=planet_xs_step[:, None, :].shape)

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
    # jax.debug.print("predicted_next_a: {a}", a=predicted_next_a.shape)

    #     inferred_as = jnp.concatenate((inferred_as[1:], predicted_next_a[None, :]))
    inferred_as = inferred_as.at[:, :-1, :].set(inferred_as[:, 1:, :])
    inferred_as = inferred_as.at[:, -1, :].set(predicted_next_a[:, 0, :])

    # jax.debug.print("inferred_as: {a}", a=inferred_as.shape)

    frozen_bs = (frozen_b_jk * inferred_as[:, :-1, :]).sum(axis=1)
    frozen_as = (frozen_a_jk * inferred_as[:, :-1, :]).sum(axis=1)
    # jax.debug.print("frozen_as: {a}", a=frozen_as.shape)

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

    inferred_as, new_little_s, predicted_x = jax.lax.scan(
        scan_func, (inferred_as, little_s, predicted_x), None, length=5
    )[0]

    return (predicted_x, inferred_as, new_little_s, big_S_last), None


########################################################################################
# Actual integrator
########################################################################################


def integrate(
    x0,
    v0,
    gms,
    t0,
    b_jk,
    a_jk,
    dt,
    planet_xs,
    planet_vs,
    planet_as,
    asteroid_xs,
    planet_params,
    asteroid_params,
    planet_gms,
    asteroid_gms,
    use_GR,
):
    MID_IND = int((a_jk.shape[1] - 1) / 2)

    ####################################################################################
    # Initial integration to get leading/trailing points
    ####################################################################################
    state = {"x": x0, "v": v0}
    forwards = odeint(
        _ode_acceleration,
        state,
        jnp.arange(MID_IND + 1) * dt,
        (
            gms,
            planet_params,
            asteroid_params,
            planet_gms,
            asteroid_gms,
            use_GR,
            t0,
            False,
        ),
        rtol=EPSILON,
        atol=EPSILON,
        mxstep=jnp.inf,
        hmax=jnp.inf,
    )

    state["v"] *= -1
    backwards = odeint(
        _ode_acceleration,
        state,
        jnp.arange(MID_IND + 1) * dt,
        (
            gms,
            planet_params,
            asteroid_params,
            planet_gms,
            asteroid_gms,
            use_GR,
            t0,
            True,
        ),
        rtol=EPSILON,
        atol=EPSILON,
        mxstep=jnp.inf,
        hmax=jnp.inf,
    )
    # jax.debug.print("{x}", x=backwards)

    inferred_xs = jnp.concatenate([backwards["x"][::-1], forwards["x"][1:]])
    inferred_vs = jnp.concatenate([backwards["v"][::-1] * -1, forwards["v"][1:]])
    inferred_xs = jnp.swapaxes(inferred_xs, 0, 1)
    inferred_vs = jnp.swapaxes(inferred_vs, 0, 1)

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
    # jax.debug.print("{x}", x=inferred_xs)

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
        length=4,
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

    swapped_planet_xs = jnp.swapaxes(planet_xs, 0, 1)
    swapped_planet_vs = jnp.swapaxes(planet_vs, 0, 1)
    swapped_planet_as = jnp.swapaxes(planet_as, 0, 1)
    swapped_asteroid_xs = jnp.swapaxes(asteroid_xs, 0, 1)

    predicted_x, _, _, _ = jax.lax.scan(
        scan_func,
        (x0, inferred_as, little_s, big_S_last),
        (
            swapped_planet_xs,
            swapped_planet_vs,
            swapped_planet_as,
            swapped_asteroid_xs,
        ),
    )[0]

    return predicted_x

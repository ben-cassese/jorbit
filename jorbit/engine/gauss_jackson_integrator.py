import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.experimental.ode import odeint

from jorbit.engine.ephemeris import planet_state
from jorbit.engine.accelerations import acceleration
from jorbit.data.constants import EPSILON


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


def integrate(
    x0,
    v0,
    gms,
    t0,
    b_jk,
    a_jk,
    steps,
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

    # Initial guesses for trailing/leading points
    # a0 = acceleration(x0)
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
    # return inferred_xs

    # Refine the trailing/leading points
    def startup_scan_func(carry, scan_over):
        (inferred_as, little_s, big_S) = carry

        # Calculate s0
        b_0k = b_jk[MID_IND][:, None]
        c1_prime = v0 / dt - (inferred_as * b_0k).sum(axis=0)
        c1 = c1_prime + a0 / 2
        little_s_mid = c1_prime

        # Calculate S0
        a_0k = a_jk[MID_IND][:, None]
        c2 = x0 / dt**2 - (inferred_as * a_0k).sum(axis=0) + c1
        big_SMID_IND = c2 - c1

        # Calculate sn
        pair_sums = (
            jnp.column_stack((inferred_as[:-1][:, None], inferred_as[1:][:, None])).sum(
                axis=1
            )
            / 2
        )
        lower_s = jax.lax.scan(
            lambda carry, scan_over: (carry - scan_over, carry - scan_over),
            little_s_mid,
            pair_sums[:MID_IND][::-1],
        )[1][::-1]
        upper_s = jax.lax.scan(
            lambda carry, scan_over: (carry + scan_over, carry + scan_over),
            little_s_mid,
            pair_sums[MID_IND:],
        )[1]
        little_s = jnp.concatenate((lower_s, jnp.array([little_s_mid]), upper_s))

        # Calculate Sn
        lower_S = jax.lax.scan(
            lambda carry, scan_over: (
                carry - scan_over[0] + scan_over[1],
                carry - scan_over[0] + scan_over[1],
            ),
            big_SMID_IND,
            (little_s[1 : MID_IND + 1][::-1], inferred_as[1 : MID_IND + 1][::-1] / 2),
        )[1][::-1]

        upper_S = jax.lax.scan(
            lambda carry, scan_over: (
                carry + scan_over[0] + scan_over[1],
                carry + scan_over[0] + scan_over[1],
            ),
            big_SMID_IND,
            (little_s[MID_IND:-1], inferred_as[MID_IND:-1] / 2),
        )[1]
        big_S = jnp.concatenate((lower_S, jnp.array([big_SMID_IND]), upper_S))

        b_terms = (b_jk[:-1, :, None] * inferred_as[None, :]).sum(axis=1)
        a_terms = (a_jk[:-1, :, None] * inferred_as[None, :]).sum(axis=1)

        inferred_vs = dt * (little_s + b_terms)
        inferred_xs = dt**2 * (big_S + a_terms)

        new_acceleration = acceleration(inferred_xs)
        # print(jnp.max(jnp.abs(new_acceleration - inferred_as)))
        inferred_as = new_acceleration
        return (inferred_as, little_s, big_S), None

    inferred_as, little_s, big_S = jax.lax.scan(
        startup_scan_func,
        (inferred_as, jnp.zeros_like(inferred_as), jnp.zeros_like(inferred_as)),
        None,
        length=3,
    )[0]
    little_s = little_s[-1]
    big_S_last = big_S[-1]

    return little_s, big_S_last

    # def stepping_scan_func(carry, scan_over):
    #     _, inferred_as, little_s, big_S_last = carry
    #     # print(f"iteration {i}")
    #     # predict
    #     big_S_last = big_S_last + little_s + (inferred_as[-1] / 2)
    #     # print(big_S_last)

    #     b_terms = (b_jk[-1, :, None] * inferred_as).sum(axis=0)
    #     a_terms = (a_jk[-1, :, None] * inferred_as).sum(axis=0)

    #     predicted_x = dt**2 * (big_S_last + a_terms)

    #     predicted_next_a = acceleration(predicted_x)

    #     inferred_as = jnp.concatenate((inferred_as[1:], predicted_next_a[None, :]))

    #     frozen_as = (a_jk[-2, :-1, None] * inferred_as[:-1]).sum(axis=0)

    #     def corrector_scan_func(carry, scan_over):
    #         inferred_as, _, _ = carry
    #         # print(inferred_as[-1,0])
    #         new_little_s = little_s + (inferred_as[-1] + inferred_as[-2]) / 2
    #         a_coeffs = a_jk[-2, -1, None] * inferred_as[-1] + frozen_as
    #         predicted_x = dt**2 * (big_S_last + a_coeffs)
    #         inferred_as = inferred_as.at[-1].set(acceleration(predicted_x))
    #         return (inferred_as, new_little_s, predicted_x), None

    #     inferred_as, new_little_s, predicted_x = jax.lax.scan(
    #         corrector_scan_func, (inferred_as, little_s, predicted_x), None, length=5
    #     )[0]

    #     return (predicted_x, inferred_as, new_little_s, big_S_last), None

    # norbits = 100
    # slices = 50
    # Q = norbits * slices + slices - MID_IND
    # predicted_x, _, _, _ = jax.lax.scan(
    #     stepping_scan_func, (x0, inferred_as, little_s, big_S_last), None, length=Q
    # )[0]

    # return predicted_x

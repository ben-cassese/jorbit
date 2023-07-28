import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.engine.ephemeris import planet_state
from jorbit.engine.accelerations import acceleration
from jorbit.data.constants import EPSILON

from jorbit.data import (
    STANDARD_PLANET_PARAMS,
    STANDARD_ASTEROID_PARAMS,
    STANDARD_PLANET_GMS,
    STANDARD_ASTEROID_GMS,
)


def _create_yoshida_coeffs(Ws):
    """Ws from tables 1 and 2"""
    w0 = 1 - 2 * (jnp.sum(Ws))
    w = jnp.concatenate((jnp.array([w0]), Ws))

    Ds = jnp.zeros(2 * len(Ws) + 1)
    Ds = Ds.at[: len(Ws)].set(Ws[::-1])
    Ds = Ds.at[len(Ws)].set(w0)
    Ds = Ds.at[len(Ws) + 1 :].set(Ws)

    Cs = jnp.zeros(2 * len(Ws) + 2)
    for i in range(len(w) - 1):
        Cs = Cs.at[i + 1].set(0.5 * (w[len(w) - 1 - i] + w[len(w) - 2 - i]))

    Cs = Cs.at[int(len(Cs) / 2) :].set(Cs[: int(len(Cs) / 2)][::-1])
    Cs = Cs.at[0].set(0.5 * w[-1])
    Cs = Cs.at[-1].set(0.5 * w[-1])

    # to do it at extended precision, use Decimal:
    # tmp = 0
    # for i in Ws:
    #     tmp += i
    # w0 = 1 - 2 * tmp
    # w = [w0] + Ws

    # Ds = [0]*(2 * len(Ws) + 1)
    # Ds[:len(Ws)] = Ws[::-1]
    # Ds[len(Ws)] = w0
    # Ds[len(Ws) + 1:] = Ws

    # Cs = [0]*(2 * len(Ws) + 2)
    # for i in range(len(w) - 1):
    #     Cs[i + 1] = Decimal(0.5) * (w[len(w) - 1 - i] + w[len(w) - 2 - i])
    # Cs[int(len(Cs) / 2):] = Cs[: int(len(Cs) / 2)][::-1]
    # Cs[0] = Decimal(0.5) * w[-1]
    # Cs[-1] = Decimal(0.5) * w[-1]

    return jnp.array(Cs), jnp.array(Ds)


def _leapfrog_acceleration(x, perturber_pos, perturber_GMs):
    r = x - perturber_pos
    return jnp.sum(
        -perturber_GMs[:, None] / ((jnp.linalg.norm(r, axis=1)) ** 3)[:, None] * r,
        axis=0,
    )


def _single_step(x0, v0, perturber_pos, perturber_GMs, dt, C, D):
    def leapfrog_scan(X, mid_step_coeffs):
        x, v = X
        c, d = mid_step_coeffs
        x = x + c * v * dt
        v = (
            v
            + d
            * _leapfrog_acceleration(
                x=x, perturber_pos=perturber_pos, perturber_GMs=perturber_GMs
            )
            * dt
        )
        return (x, v), None

    q = jax.lax.scan(leapfrog_scan, (x0, v0), (C[:-1], D))[0]
    x, v = q
    x = x + C[-1] * v * dt
    return x, v


def integrate(
    x0,
    v0,
    t0,
    tf,
    steps,
    C,
    D,
    planet_params=STANDARD_PLANET_PARAMS,
    asteroid_params=STANDARD_ASTEROID_PARAMS,
    planet_GMs=STANDARD_PLANET_GMS,
    asteroid_GMs=STANDARD_ASTEROID_GMS,
):
    times = jnp.linspace(t0, tf, steps)
    dt = jnp.diff(times)[0]
    perturber_gms = jnp.concatenate((planet_GMs, asteroid_GMs))

    planet_xs, _, _ = planet_state(
        planet_params=planet_params, times=times, velocity=False, acceleration=False
    )
    asteroid_xs, _, _ = planet_state(
        planet_params=asteroid_params, times=times, velocity=False, acceleration=False
    )

    return planet_xs, asteroid_xs

    def scan_func(carry, scan_over):
        return (
            _single_step(
                x0=carry[0],
                v0=carry[1],
                perturber_pos=scan_over,
                perturber_GMs=perturber_gms,
                dt=dt,
                C=C,
                D=D,
            ),
            None,
        )

    return jax.lax.scan(scan_func, (x0, v0), jnp.row_stack((planet_xs, asteroid_xs)))[0]

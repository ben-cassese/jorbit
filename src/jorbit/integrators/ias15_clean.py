"""General refactor of ias15.py to avoid dataclasses make it more JAX friendly."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.data.constants import IAS15_BV_DENOMS, IAS15_BX_DENOMS


def _estimate_x_v_from_b(
    a0: jnp.ndarray,
    v0: jnp.ndarray,
    x0: jnp.ndarray,
    h: jnp.ndarray,
    dt: jnp.ndarray,
    bp: jnp.ndarray,  # remember to flip it!
) -> tuple[jnp.ndarray, jnp.ndarray]:
    xcoeffs = bp * dt * dt / IAS15_BX_DENOMS
    x, _ = jax.lax.scan(lambda y, _p: (y * h + _p, None), jnp.zeros_like(x0), xcoeffs)
    x *= h * h * h
    x += (v0 * dt) * h + (a0 * dt * dt / 2.0) * h * h + x0

    vcoeffs = bp * dt / IAS15_BV_DENOMS
    v, _ = jax.lax.scan(lambda y, _p: (y * h + _p, None), jnp.zeros_like(x0), vcoeffs)
    v *= h * h
    v += v0 + (a0 * dt) * h
    return x, v


def _refine_sub_g(
    at: jnp.ndarray, a0: jnp.ndarray, previous_gs: jnp.ndarray, r: jnp.ndarray
) -> jnp.ndarray:

    def scan_body(carry: tuple, scan_over: tuple) -> tuple:
        result = carry
        g, r_sub = scan_over
        result = (result - g) * r_sub
        return result, None

    initial_result = (at - a0) * r[0]
    new_g, _ = jax.lax.scan(scan_body, initial_result, (previous_gs, r[1:]))
    return new_g

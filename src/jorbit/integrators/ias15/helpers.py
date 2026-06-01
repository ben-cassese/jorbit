"""Low-level IAS15 primitives shared across the step and interpolation modules."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jorbit.data.constants import (
    IAS15_BV_DENOMS,
    IAS15_BX_DENOMS,
)
from jorbit.utils.states import IAS15IntegratorState


def initialize_ias15_integrator_state(a0: jnp.ndarray) -> IAS15IntegratorState:
    """Initializes the IAS15IntegratorState dataclass with zeros.

    Args:
        a0 (jnp.ndarray):
            The initial acceleration.

    Returns:
        IAS15IntegratorState:
            An instance of the IAS15IntegratorState dataclass with zeros.
    """
    n_particles = a0.shape[0]
    zeros_b = jnp.zeros((7, n_particles, 3), dtype=jnp.float64)
    return IAS15IntegratorState(
        g=zeros_b,
        b=zeros_b,
        e=zeros_b,
        csx=jnp.zeros((n_particles, 3), dtype=jnp.float64),
        csv=jnp.zeros((n_particles, 3), dtype=jnp.float64),
        a0=a0,
        dt=0.001,
        dt_last_done=0.0,
    )


@jax.jit
def add_cs(p: jnp.ndarray, csp: jnp.ndarray, inp: jnp.ndarray) -> tuple:
    """Compensated summation.

    Args:
        p (jnp.ndarray):
            The current sum.
        csp (jnp.ndarray):
            The current compensation.
        inp (jnp.ndarray):
            The input to add.

    Returns:
        tuple:
            The new sum and compensation.
    """
    y = inp - csp
    t = p + y
    csp = (t - p) - y
    p = t
    return p, csp


@jax.jit
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

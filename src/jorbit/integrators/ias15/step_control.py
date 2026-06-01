"""Adaptive step-size controllers for the IAS15 integrator.

These are passed to :func:`ias15_step` as the ``step_scheduler`` callable; they are not
called internally by the step machinery.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jorbit.data.constants import (
    IAS15_EPSILON,
    IAS15_MIN_DT,
    IAS15_SAFETY_FACTOR,
    IAS15_EPS_Modified,
)


@jax.jit
def next_proposed_dt_PRS23(
    a0: jnp.ndarray,
    at_fresh: jnp.ndarray,
    b: jnp.ndarray,
    dt_done: float,
    x_end: jnp.ndarray,
    v_end: jnp.ndarray,
) -> jnp.ndarray:
    """The PRS23 step controller."""
    tmp = a0 + jnp.sum(b, axis=0)
    y2 = jnp.sum(tmp * tmp, axis=1)

    coeffs_1 = jnp.arange(1, 8)
    tmp = jnp.sum(coeffs_1[:, None, None] * b, axis=0)
    y3 = jnp.sum(tmp * tmp, axis=1)

    coeffs_2 = jnp.arange(2, 8) * jnp.arange(1, 7)
    tmp = jnp.sum(coeffs_2[:, None, None] * b[1:], axis=0)
    y4 = jnp.sum(tmp * tmp, axis=1)

    timescale2 = 2.0 * y2 / (y3 + jnp.sqrt(y4 * y2))  # PRS23
    min_timescale2 = jnp.nanmin(timescale2)
    dt_new = jnp.sqrt(min_timescale2) * dt_done * IAS15_EPS_Modified
    return dt_new


@jax.jit
def next_proposed_dt_global(
    a0: jnp.ndarray,
    at_fresh: jnp.ndarray,
    b: jnp.ndarray,
    dt_done: float,
    x_end: jnp.ndarray,
    v_end: jnp.ndarray,
) -> jnp.ndarray:
    """REBOUND's GLOBAL step controller (legacy, used by ASSIST).

    Compares the magnitude of the highest-order polynomial coefficient (`b[6]`)
    to the freshly-evaluated end-of-step acceleration (`at_fresh`, taken from
    the last predictor-corrector sub-step at h = IAS15_H[7] = 0.977). Includes
    REBOUND's "slow-acceleration" filter that skips particles with
    `v²·dt²/x² < 1e-16`, evaluated on the END-of-step predictor state
    (`x_end, v_end`) to match REBOUND's `particles[mi]` semantics
    (`integrator_ias15.c:543-558`). Falls back to `dt/safety_factor` growth
    when no particle contributes. Finally clamps the proposed step to
    `IAS15_MIN_DT`. See REBOUND `integrator_ias15.c:534-619`. ASSIST forces
    this mode at `assist.c:446`.
    """
    del a0
    v2 = jnp.sum(v_end * v_end, axis=1)
    x2 = jnp.sum(x_end * x_end, axis=1)
    keep = (v2 * dt_done * dt_done / x2) >= 1e-16
    at_masked = jnp.where(keep[:, None], at_fresh, 0.0)
    b6_masked = jnp.where(keep[:, None], b[6], 0.0)
    maxa = jnp.max(jnp.abs(at_masked))
    maxj = jnp.max(jnp.abs(b6_masked))
    integrator_error = maxj / maxa
    dt_new = jnp.where(
        jnp.isfinite(integrator_error) & (integrator_error > 0),
        dt_done * jnp.power(IAS15_EPSILON / integrator_error, 1.0 / 7.0),
        dt_done / IAS15_SAFETY_FACTOR,
    )
    dt_new = jnp.where(
        jnp.abs(dt_new) < IAS15_MIN_DT,
        jnp.sign(dt_new) * IAS15_MIN_DT,
        dt_new,
    )
    return dt_new

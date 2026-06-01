"""Dynamic-integrator residuals/log-likelihood helpers used for orbit fitting."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from jorbit.astrometry.sky_projection import tangent_plane_projection
from jorbit.particle.ephem import _ephem
from jorbit.utils.states import (
    CartesianState,
    IAS15IntegratorState,
    KeplerianState,
    LeapfrogIntegratorState,
)


@jax.jit
def _residuals(
    times: jnp.ndarray,
    gravity: Callable,
    integrator: Callable,
    integrator_state: IAS15IntegratorState | LeapfrogIntegratorState,
    observer_positions: jnp.ndarray,
    ra: jnp.ndarray,
    dec: jnp.ndarray,
    relevant_inds: jnp.ndarray,
    particle_state: CartesianState | KeplerianState,
    step_scheduler: Callable,
) -> jnp.ndarray:
    ras, decs = _ephem(
        times,
        particle_state,
        gravity,
        integrator,
        integrator_state,
        observer_positions,
        relevant_inds,
        step_scheduler,
    )

    xis_etas = jax.vmap(tangent_plane_projection)(ra, dec, ras, decs)

    return xis_etas


# note: this external jitted function does not have fwd mode autodiff enforced, will
# break on reverse mode when using ias15
@jax.jit
def _loglike(
    times: jnp.ndarray,
    gravity: Callable,
    integrator: Callable,
    integrator_state: IAS15IntegratorState | LeapfrogIntegratorState,
    observer_positions: jnp.ndarray,
    ra: jnp.ndarray,
    dec: jnp.ndarray,
    inv_cov_matrices: jnp.ndarray,
    cov_log_dets: jnp.ndarray,
    relevant_inds: jnp.ndarray,
    particle_state: CartesianState | KeplerianState,
    step_scheduler: Callable,
) -> float:
    xis_etas = _residuals(
        times,
        gravity,
        integrator,
        integrator_state,
        observer_positions,
        ra,
        dec,
        relevant_inds,
        particle_state,
        step_scheduler,
    )

    quad = jnp.einsum("bi,bij,bj->b", xis_etas, inv_cov_matrices, xis_etas)

    ll = jnp.sum(-0.5 * (2 * jnp.log(2 * jnp.pi) + cov_log_dets + quad))

    return ll

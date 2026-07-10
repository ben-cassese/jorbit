"""Compile-once batched forward model for a ``System`` scored against shared observations.

The multi-particle analog of :mod:`jorbit.particle.likelihood`. Where ``Particle`` pairs a
single state with an ``Observations`` object and exposes ``loglike``/``residuals`` over that
one state, this module batches over the ``System``'s particle axis: every particle is scored
against the *same* shared ``Observations``. The callables use the bounded-arc IAS15 dense
path (:func:`jorbit.system.ias15_dense._ephem_ias15_bounded`) — a single dense buffer per
direction, no host stitching — so they are jit-able and fast enough for MCMC inner loops.

All observation-dependent (state-independent) work is done once in
:func:`precompute_system_forward_model_data`; :func:`create_system_forward_model` closes over
it and returns jitted callables over a ``(P, 6)`` batch of candidate states.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from astropy.time import Time

from jorbit.astrometry.sky_projection import tangent_plane_projection
from jorbit.system.ias15_dense import _ephem_ias15_bounded


def precompute_system_forward_model_data(
    system: System,  # noqa: F821
    observations: Observations,  # noqa: F821
    step_scheduler: Callable,
) -> tuple:
    """Bind the observation-dependent data for a fast, reusable ``System`` forward model.

    Args:
        system (System):
            The System providing the acceleration function (``system.gravity``) and
            reference epoch (``system._t_ref_jd``).
        observations (Observations):
            The shared astrometric observations every particle is scored against.
        step_scheduler (Callable):
            The adaptive step-size controller used by the dense integrator.

    Returns:
        tuple:
            The frozen inputs consumed by :func:`create_system_forward_model`.
    """
    t_ref_jd = system._t_ref_jd
    acc_func = system.gravity

    # Absolute-JD astropy times preserve sub-ns precision through the offset subtraction.
    obs_times_astropy = observations.times_astropy
    if obs_times_astropy is None:
        obs_times_astropy = Time(observations.times, format="jd", scale="tdb")
    times_off = system._times_to_offsets(obs_times_astropy)

    # The obs times cross the reference epoch; one dense-output run integrates toward the
    # single farthest time, so split into a forward pass (offsets >= 0) and a backward pass
    # (offsets < 0), clamping the other pass's times to 0. These masks are frozen constants.
    fwd_mask = times_off >= 0.0
    times_fwd = jnp.where(fwd_mask, times_off, 0.0)
    times_bwd = jnp.where(fwd_mask, 0.0, times_off)

    return (
        acc_func,
        t_ref_jd,
        times_off,
        fwd_mask,
        times_fwd,
        times_bwd,
        observations.observer_positions,
        observations.ra,
        observations.dec,
        observations.inv_cov_matrices,
        observations.cov_log_dets,
        step_scheduler,
    )


def create_system_forward_model(inputs: tuple) -> dict:
    """Build the jitted, reusable forward-model callables over a ``(P, 6)`` state batch.

    Each callable takes a ``(P, 6)`` array of barycentric equatorial Cartesian states
    ``[x, y, z, vx, vy, vz]`` (AU, AU/day) at the reference epoch and returns per-particle
    outputs. Truncation (the shared-schedule arc exceeding one dense buffer) is handled on
    device: unreachable observations are poisoned to ``NaN`` in ``model_radec``/``residuals``
    (the diagnostic), and ``loglike``/``chi2`` collapse to ``-inf``/``+inf`` so a sampler
    rejects the step rather than crashing or accepting a finite-but-wrong value. Because the
    step schedule is shared across the batch, truncation is batch-wide.

    Args:
        inputs (tuple):
            The output of :func:`precompute_system_forward_model_data`.

    Returns:
        dict:
            ``model_radec``/``residuals``/``chi2``/``loglike`` jitted callables plus ``n_obs``.
    """
    (
        acc_func,
        t_ref_jd,
        times_off,
        fwd_mask,
        times_fwd,
        times_bwd,
        observer_positions,
        obs_ra,
        obs_dec,
        inv_cov_matrices,
        cov_log_dets,
        step_scheduler,
    ) = inputs

    n_obs = int(times_off.shape[0])
    log_2pi_2 = 2.0 * jnp.log(2.0 * jnp.pi)

    def _model(states: jnp.ndarray) -> tuple:
        # (ras, decs) each (P, n_obs); reached (n_obs,) — shared across the batch.
        return _ephem_ias15_bounded(
            states,
            times_off,
            fwd_mask,
            times_fwd,
            times_bwd,
            observer_positions,
            t_ref_jd,
            acc_func,
            step_scheduler,
        )

    def _raw_residuals(ras: jnp.ndarray, decs: jnp.ndarray) -> jnp.ndarray:
        # tangent_plane_projection(obs_ra, obs_dec, model_ra, model_dec) -> (model - obs)
        # offset in arcsec, matching particle/likelihood.py:_residuals.
        def per_particle(ra_row: jnp.ndarray, dec_row: jnp.ndarray) -> jnp.ndarray:
            return jax.vmap(tangent_plane_projection)(obs_ra, obs_dec, ra_row, dec_row)

        return jax.vmap(per_particle)(ras, decs)  # (P, n_obs, 2)

    @jax.jit
    def model_radec(states: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        ras, decs, reached = _model(states)
        mask = reached[None, :]
        return jnp.where(mask, ras, jnp.nan), jnp.where(mask, decs, jnp.nan)

    @jax.jit
    def residuals(states: jnp.ndarray) -> jnp.ndarray:
        ras, decs, reached = _model(states)
        r = _raw_residuals(ras, decs)
        return jnp.where(reached[None, :, None], r, jnp.nan)  # (P, n_obs, 2)

    @jax.jit
    def chi2(states: jnp.ndarray) -> jnp.ndarray:
        ras, decs, reached = _model(states)
        r = _raw_residuals(ras, decs)
        quad = jnp.einsum("pbi,bij,pbj->p", r, inv_cov_matrices, r)  # (P,)
        return jnp.where(jnp.all(reached), quad, jnp.inf)

    @jax.jit
    def loglike(states: jnp.ndarray) -> jnp.ndarray:
        ras, decs, reached = _model(states)
        r = _raw_residuals(ras, decs)
        quad = jnp.einsum("pbi,bij,pbj->pb", r, inv_cov_matrices, r)  # (P, n_obs)
        ll = jnp.sum(-0.5 * (log_2pi_2 + cov_log_dets[None, :] + quad), axis=1)  # (P,)
        return jnp.where(jnp.all(reached), ll, -jnp.inf)

    return {
        "model_radec": model_radec,
        "residuals": residuals,
        "chi2": chi2,
        "loglike": loglike,
        "n_obs": n_obs,
    }

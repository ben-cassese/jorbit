"""Compile-once batched forward model for a ``System`` scored against shared observations.

The multi-particle analog of :mod:`jorbit.particle.likelihood`. Where ``Particle`` pairs a
single state with an ``Observations`` object and exposes ``loglike``/``residuals`` over that
one state, this module batches over the ``System``'s particle axis: every particle is scored
against the *same* shared ``Observations``. The callables use the bounded-arc IAS15 dense
path (:func:`jorbit.system.ias15_dense._ephem_ias15_bounded`) — a single dense buffer per
direction, no host stitching — so they are jit-able and fast enough for MCMC inner loops.

All observation-dependent (state-independent) work is done once in
:func:`precompute_system_forward_model_data`; :func:`create_system_forward_model` binds it
as a Partial argument to shared jitted callables over a ``(P, 6)`` batch of candidate states.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from astropy.time import Time

from jorbit.astrometry.sky_projection import tangent_plane_projection
from jorbit.data.constants import SPEED_OF_LIGHT
from jorbit.integrators import create_leapfrog_times, leapfrog_evolve
from jorbit.system.ephem import _ephem
from jorbit.system.ias15_dense import _ephem_ias15_bounded
from jorbit.utils.states import SystemState


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


def precompute_system_leapfrog_forward_model_data(
    system: System,  # noqa: F821
    observations: Observations,  # noqa: F821
) -> tuple:
    """Bind the observation-dependent data for a fixed-step leapfrog ``System`` model.

    The leapfrog counterpart of :func:`precompute_system_forward_model_data`. Leapfrog
    lands on every requested time by construction, so there is no forward/backward dense
    buffer to split: instead the observation times are expanded once, here, into a step
    schedule no coarser than the System's ``max_step_size``
    (:func:`jorbit.integrators.create_leapfrog_times`, a host-side loop that cannot run
    under ``jax.jit``).

    The returned tuple keeps the *observation* slots (indices 6-10) in the same positions
    as the IAS15 tuple, since the shared ``_residuals``/``_chi2``/``_loglike`` wrappers
    index them positionally.

    Args:
        system (System):
            The System providing the acceleration function (``system.gravity``), the
            reference epoch (``system._t_ref_jd``), and the leapfrog coefficients and
            step size (``system._integrator_state``).
        observations (Observations):
            The shared astrometric observations every particle is scored against.

    Returns:
        tuple:
            The frozen inputs consumed by :func:`create_system_forward_model` when it is
            given ``model_fn=_model_leapfrog``.
    """
    obs_times_astropy = observations.times_astropy
    if obs_times_astropy is None:
        obs_times_astropy = Time(observations.times, format="jd", scale="tdb")
    times_off = system._times_to_offsets(obs_times_astropy)

    expanded_times, inds = create_leapfrog_times(
        t0=system._state.relative_time,
        times=times_off,
        biggest_allowed_dt=system._integrator_state.dt,
    )

    return (
        system.gravity,
        system._t_ref_jd,
        times_off,
        expanded_times,
        inds,
        system._integrator_state,
        observations.observer_positions,
        observations.ra,
        observations.dec,
        observations.inv_cov_matrices,
        observations.cov_log_dets,
        jax.tree_util.Partial(leapfrog_evolve),
    )


_LOG_2PI_2 = 2.0 * jnp.log(2.0 * jnp.pi)


def _model(inputs: tuple, states: jnp.ndarray, max_steps: int | None = None) -> tuple:
    # (ras, decs) each (P, n_obs); reached (n_obs,) — shared across the batch.
    (
        acc_func,
        t_ref_jd,
        times_off,
        fwd_mask,
        times_fwd,
        times_bwd,
        observer_positions,
        _obs_ra,
        _obs_dec,
        _inv_cov_matrices,
        _cov_log_dets,
        step_scheduler,
    ) = inputs
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
        max_steps,
    )


def _model_leapfrog(
    inputs: tuple, states: jnp.ndarray, max_steps: int | None = None
) -> tuple:
    # (ras, decs) each (P, n_obs); reached (n_obs,) — always True, since a fixed-step
    # schedule lands on every requested time and so cannot truncate. max_steps is
    # accepted and ignored to keep the model-function signature uniform.
    del max_steps
    (
        acc_func,
        t_ref_jd,
        times_off,
        expanded_times,
        inds,
        integrator_state,
        observer_positions,
        _obs_ra,
        _obs_dec,
        _inv_cov_matrices,
        _cov_log_dets,
        integrator,
    ) = inputs
    empty3 = jnp.empty((0, 3))
    state = SystemState(
        tracer_positions=states[:, :3],
        tracer_velocities=states[:, 3:],
        massive_positions=empty3,
        massive_velocities=empty3,
        log_gms=jnp.empty((0,)),
        time_reference=jnp.asarray(t_ref_jd),
        relative_time=jnp.asarray(0.0),
        fixed_perturber_positions=empty3,
        fixed_perturber_velocities=empty3,
        fixed_perturber_log_gms=jnp.empty((0,)),
        acceleration_func_kwargs={"c2": SPEED_OF_LIGHT**2},
    )
    ras, decs = _ephem(
        expanded_times,
        state,
        acc_func,
        integrator,
        integrator_state,
        observer_positions,
        inds,
        None,  # leapfrog ignores the step scheduler
    )
    return ras, decs, jnp.ones(times_off.shape[0], dtype=bool)


def _raw_residuals(
    obs_ra: jnp.ndarray, obs_dec: jnp.ndarray, ras: jnp.ndarray, decs: jnp.ndarray
) -> jnp.ndarray:
    # tangent_plane_projection(obs_ra, obs_dec, model_ra, model_dec) -> (model - obs)
    # offset in arcsec, matching particle/likelihood.py:_residuals.
    def per_particle(ra_row: jnp.ndarray, dec_row: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(tangent_plane_projection)(obs_ra, obs_dec, ra_row, dec_row)

    return jax.vmap(per_particle)(ras, decs)  # (P, n_obs, 2)


@partial(jax.jit, static_argnames=["max_steps"])
def _model_radec(
    model_fn: Callable,
    inputs: tuple,
    states: jnp.ndarray,
    max_steps: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    ras, decs, reached = model_fn(inputs, states, max_steps)
    mask = reached[None, :]
    return jnp.where(mask, ras, jnp.nan), jnp.where(mask, decs, jnp.nan)


@partial(jax.jit, static_argnames=["max_steps"])
def _model_radec_status(
    model_fn: Callable,
    inputs: tuple,
    states: jnp.ndarray,
    max_steps: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Diagnostic twin of _model_radec: the same NaN-poisoned (ras, decs), plus the
    # reach mask _model_radec discards. A separate jitted function rather than a flag
    # on _model_radec, so callers that never ask for the status never compile for it.
    ras, decs, reached = model_fn(inputs, states, max_steps)
    mask = reached[None, :]
    return jnp.where(mask, ras, jnp.nan), jnp.where(mask, decs, jnp.nan), reached


@partial(jax.jit, static_argnames=["max_steps"])
def _residuals(
    model_fn: Callable,
    inputs: tuple,
    states: jnp.ndarray,
    max_steps: int | None = None,
) -> jnp.ndarray:
    obs_ra, obs_dec = inputs[7], inputs[8]
    ras, decs, reached = model_fn(inputs, states, max_steps)
    r = _raw_residuals(obs_ra, obs_dec, ras, decs)
    return jnp.where(reached[None, :, None], r, jnp.nan)  # (P, n_obs, 2)


@partial(jax.jit, static_argnames=["max_steps"])
def _chi2(
    model_fn: Callable,
    inputs: tuple,
    states: jnp.ndarray,
    max_steps: int | None = None,
) -> jnp.ndarray:
    obs_ra, obs_dec, inv_cov_matrices = inputs[7], inputs[8], inputs[9]
    ras, decs, reached = model_fn(inputs, states, max_steps)
    r = _raw_residuals(obs_ra, obs_dec, ras, decs)
    quad = jnp.einsum("pbi,bij,pbj->p", r, inv_cov_matrices, r)  # (P,)
    return jnp.where(jnp.all(reached), quad, jnp.inf)


@partial(jax.jit, static_argnames=["max_steps"])
def _loglike(
    model_fn: Callable,
    inputs: tuple,
    states: jnp.ndarray,
    max_steps: int | None = None,
) -> jnp.ndarray:
    obs_ra, obs_dec, inv_cov_matrices, cov_log_dets = (
        inputs[7],
        inputs[8],
        inputs[9],
        inputs[10],
    )
    ras, decs, reached = model_fn(inputs, states, max_steps)
    r = _raw_residuals(obs_ra, obs_dec, ras, decs)
    quad = jnp.einsum("pbi,bij,pbj->pb", r, inv_cov_matrices, r)  # (P, n_obs)
    ll = jnp.sum(-0.5 * (_LOG_2PI_2 + cov_log_dets[None, :] + quad), axis=1)  # (P,)
    return jnp.where(jnp.all(reached), ll, -jnp.inf)


def create_system_forward_model(
    inputs: tuple, max_steps: int | None = None, model_fn: Callable = _model
) -> dict:
    """Build the jitted, reusable forward-model callables over a ``(P, 6)`` state batch.

    Each callable takes a ``(P, 6)`` array of barycentric equatorial Cartesian states
    ``[x, y, z, vx, vy, vz]`` (AU, AU/day) at the reference epoch and returns per-particle
    outputs. Truncation (the shared-schedule arc exceeding one dense buffer) is handled on
    device: unreachable observations are poisoned to ``NaN`` in ``model_radec``/``residuals``
    (the diagnostic), and ``loglike``/``chi2`` collapse to ``-inf``/``+inf`` so a sampler
    rejects the step rather than crashing or accepting a finite-but-wrong value. Because the
    step schedule is shared across the batch, truncation is batch-wide: one particle whose
    orbit needs more steps than the buffer holds truncates every particle in the batch, so
    there is no single ``max_steps`` that suits a dynamically heterogeneous batch — group
    (or sort) batches by dynamical timescale before sizing the buffer down.

    The callables are ``jax.tree_util.Partial`` bindings of shared module-level jitted
    functions, with ``inputs`` passed through as a pytree argument. Binding (rather than
    closure-capturing) keeps the ephemeris/observation arrays out of the compiled
    executables — closure constants are re-embedded per compilation, ~150 MB each for
    the default ephemeris — and lets Systems with matching shapes share compilations.

    Args:
        inputs (tuple):
            The output of :func:`precompute_system_forward_model_data`.
        model_fn (Callable):
            The ``(inputs, states, max_steps) -> (ras, decs, reached)`` step. Defaults
            to the bounded-arc IAS15 dense model; pass :func:`_model_leapfrog` (with
            leapfrog ``inputs``) for a fixed-step ``System``, which never truncates and
            so ignores ``max_steps``.
        max_steps (int | None):
            Dense-output buffer depth (max accepted IAS15 steps per directional pass)
            used by every callable. None (default) keeps the backend's
            ``IAS15_MAX_DYNAMIC_STEPS`` (15000). Short arcs need only a handful of
            steps, and the full-size buffers are pure allocation overhead (gigabytes
            at large ``P``), so sizing this to the arc can speed the forward model up
            by 1-2 orders of magnitude; an undersized buffer fails loudly through the
            truncation contract above rather than returning finite-but-wrong values.

    Returns:
        dict:
            ``model_radec``/``residuals``/``chi2``/``loglike`` jitted callables plus
            ``model_radec_with_status`` and ``n_obs``. ``model_radec_with_status``
            returns the same ``(ras, decs)`` as ``model_radec`` plus the ``(n_obs,)``
            boolean reach mask behind the ``NaN`` poisoning, so a caller can tell
            buffer truncation (``reached`` False) from a genuine dynamical failure
            such as a ``NaN`` acceleration (``reached`` True but the values ``NaN``).
    """
    n_obs = int(inputs[2].shape[0])

    model_fn = jax.tree_util.Partial(model_fn)

    def bind(func: Callable) -> jax.tree_util.Partial:
        if max_steps is None:
            return jax.tree_util.Partial(func, model_fn, inputs)
        # functools.partial (not tree_util.Partial) keeps max_steps a plain Python
        # int in the callable's aux data, so it stays hashable/static even when a
        # user wraps the returned callable in their own jax.jit.
        return jax.tree_util.Partial(
            partial(func, max_steps=max_steps), model_fn, inputs
        )

    return {
        "model_radec": bind(_model_radec),
        "model_radec_with_status": bind(_model_radec_status),
        "residuals": bind(_residuals),
        "chi2": bind(_chi2),
        "loglike": bind(_loglike),
        "n_obs": n_obs,
    }

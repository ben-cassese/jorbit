"""Tests for the truncation-proof orchestration in jorbit.integrators.budgeted.

These exercise the host-side machinery that sits on top of the (hard-capped) IAS15
backends so that the public Particle/System methods never silently truncate:

- stitching successive interpolation chunks (15k dense-output buffer), and
- inserting dummy landing times for forced landing (10k per-interval cap).

References are kept internal (stitched vs. forced-landing vs. direct backend) so the
suite runs offline; observers are passed as explicit position arrays for the same
reason. Horizons-level accuracy is covered by the existing test_particle.py suite.
"""

import itertools
from collections.abc import Callable

import jax

jax.config.update("jax_enable_x64", True)

import astropy.units as u
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.time import Time

import jorbit.integrators.ias15 as ias15_module
from jorbit import Particle, System
from jorbit.accelerations import newtonian_gravity
from jorbit.data.constants import SPEED_OF_LIGHT
from jorbit.integrators import (
    budgeted_forced_landing,
    ias15_evolve,
    ias15_evolve_forced_landing,
    ias15_evolve_with_dense_output,
    stitched_interpolate,
)
from jorbit.integrators.budgeted import (
    _loaded_ephemeris_bounds_jd,
    _no_progress_error,
    discover_natural_step_times,
    insert_budget_dummy_times,
)
from jorbit.utils.states import CartesianState

X0 = jnp.array([-2.003779703686627, 1.780533558134481, 0.5203350526739642])
V0 = jnp.array([-0.006668390915419885, -0.006621147093559814, -0.002036640485149475])
T0 = Time("2025-01-01")


@pytest.fixture(scope="module")
def particle() -> Particle:
    """A single perturbed-gravity (newtonian planets) particle reused across tests."""
    return Particle(x=X0, v=V0, time=T0, gravity="newtonian planets")


@pytest.fixture(scope="module")
def two_body_system() -> System:
    """A two-tracer System reused across tests."""
    p1 = Particle(x=X0, v=V0, time=T0, gravity="newtonian planets")
    p2 = Particle(x=X0 + 1e-3, v=V0 - 1e-5, time=T0, gravity="newtonian planets")
    return System(particles=[p1, p2], gravity="newtonian planets")


def _shrink_cap(value: int) -> Callable[[], None]:
    """Save/patch/restore the IAS15 dense-output buffer cap (and clear jit caches).

    Returns a (restore) callable. Patching the module constant before clearing the jit
    caches forces the backend to re-trace with the smaller buffer, so a normal-length
    span is driven to truncate without needing a genuinely enormous integration.

    The constant must be patched on the ``evolve`` submodule, since that is the
    namespace the dense-output loops read it from; patching the re-exported binding on
    the ``ias15`` package would not change what those functions see.
    """
    original = ias15_module.evolve.IAS15_MAX_DYNAMIC_STEPS
    ias15_module.evolve.IAS15_MAX_DYNAMIC_STEPS = value
    jax.clear_caches()

    def restore() -> None:
        ias15_module.evolve.IAS15_MAX_DYNAMIC_STEPS = original
        jax.clear_caches()

    return restore


# ---------------------------------------------------------------------------
# Pure-function: dummy-time insertion
# ---------------------------------------------------------------------------


def test_insert_budget_dummy_times() -> None:
    """Dummies subdivide an over-budget interval; requested times are recoverable."""
    # 99 natural step boundaries strictly inside (0, 10].
    nst = jnp.cumsum(jnp.full(100, 0.1))  # 0.1 .. 10.0
    aug, inds = insert_budget_dummy_times(nst, jnp.array([10.0]), t0=0.0, budget=30)

    # Requested time is recoverable via relevant_inds.
    assert float(aug[inds][0]) == pytest.approx(10.0)
    # The requested time is the last entry.
    assert int(inds[0]) == aug.shape[0] - 1

    # No (sub-)interval between consecutive augmented landings exceeds the budget.
    bounds = np.concatenate([[0.0], np.asarray(aug)])
    nst_np = np.asarray(nst)
    for a, b in itertools.pairwise(bounds):
        n = int(np.sum((nst_np > a) & (nst_np < b)))
        assert n <= 30

    # A within-budget interval needs no dummies.
    aug2, inds2 = insert_budget_dummy_times(nst, jnp.array([2.0]), t0=0.0, budget=30)
    assert aug2.shape[0] == 1
    assert int(inds2[0]) == 0


# ---------------------------------------------------------------------------
# Common case: stitching/budgeting is a no-op (bit-identical to one backend call)
# ---------------------------------------------------------------------------


def test_stitched_interpolate_matches_single_chunk(particle: Particle) -> None:
    """A short span fits in one chunk; stitched output is bit-identical to the backend."""
    times = T0 + np.linspace(0, 90, 10) * u.day
    toff = particle._times_to_offsets(times)
    state = particle._cartesian_state.to_system()

    xs, vs, steps = stitched_interpolate(
        state,
        particle.gravity,
        toff,
        particle._integrator_state,
        particle._step_scheduler,
    )
    pos_d, vel_d, _, _, it = ias15_evolve(
        state,
        particle.gravity,
        toff,
        particle._integrator_state,
        particle._step_scheduler,
    )
    assert float(jnp.max(jnp.abs(xs - pos_d))) == 0.0
    assert float(jnp.max(jnp.abs(vs - vel_d))) == 0.0
    assert steps == int(it)


def test_integrate_methods_agree(particle: Particle) -> None:
    """Integrate (forced landing) and integrate_or_interpolate agree on a clean span."""
    times = T0 + np.linspace(0, 120, 8) * u.day
    xi, _ = particle.integrate_or_interpolate(times)
    xf, _ = particle.integrate(times)
    # Different numerical paths (interpolation vs. forced landing) but same orbit.
    assert float(jnp.max(jnp.abs(xi - xf))) < 1e-9  # AU


def test_system_integrate_matches_single_chunk(two_body_system: System) -> None:
    """System.integrate is bit-identical to a direct single-chunk backend call."""
    sys = two_body_system
    times = T0 + np.linspace(0, 90, 7) * u.day
    toff = sys._times_to_offsets(times)
    # the System's own default scheduler, so the direct backend call below matches
    # whatever sys.integrate() uses
    scheduler = sys._step_scheduler

    pos, _, steps = sys.integrate(times, return_steps=True)
    pos_d, _, _, _, it = ias15_evolve(
        sys._state, sys.gravity, toff, sys._integrator_state, scheduler
    )
    assert float(jnp.max(jnp.abs(pos - pos_d))) == 0.0
    assert steps == int(it)


# ---------------------------------------------------------------------------
# Natural-step discovery and forced-landing dummy round-trip
# ---------------------------------------------------------------------------


def test_discover_natural_step_times(particle: Particle) -> None:
    """Cumulative natural step times are monotonic and span past the target."""
    times = T0 + np.linspace(0, 120, 6) * u.day
    toff = particle._times_to_offsets(times)
    state = particle._cartesian_state.to_system()

    nst = discover_natural_step_times(
        state,
        particle.gravity,
        toff,
        particle._integrator_state,
        particle._step_scheduler,
    )
    assert jnp.all(jnp.diff(nst) > 0)
    assert float(nst[0]) > float(state.relative_time)
    assert float(nst[-1]) >= float(jnp.max(toff))


def test_forced_landing_dummy_roundtrip(particle: Particle) -> None:
    """Inserting (and dropping) dummy landings reproduces the forced-landing answer.

    Uses a tiny budget so dummies are inserted even on a clean span; the positions at
    the originally requested times should be unchanged but for the negligible
    perturbation of splitting steps at the dummy landings.
    """
    times = T0 + np.linspace(0, 90, 5) * u.day
    toff = particle._times_to_offsets(times)
    state = particle._cartesian_state.to_system()

    pos_ref, _, _, _, _ = ias15_evolve_forced_landing(
        state,
        particle.gravity,
        toff,
        particle._integrator_state,
        particle._step_scheduler,
    )
    nst = discover_natural_step_times(
        state,
        particle.gravity,
        toff,
        particle._integrator_state,
        particle._step_scheduler,
    )
    aug, inds = insert_budget_dummy_times(
        nst, toff, float(state.relative_time), budget=2
    )
    assert aug.shape[0] > toff.shape[0]  # dummies were actually inserted
    pos_aug, _, _, _, _ = ias15_evolve_forced_landing(
        state,
        particle.gravity,
        aug,
        particle._integrator_state,
        particle._step_scheduler,
    )
    assert float(jnp.max(jnp.abs(pos_aug[inds] - pos_ref))) < 1e-7  # AU (~15 m)


# ---------------------------------------------------------------------------
# Pathological span: stitching recovers what a single buffer would truncate
# ---------------------------------------------------------------------------


def test_stitching_recovers_truncated_span(particle: Particle) -> None:
    """With a shrunk buffer the single-chunk backend truncates; stitching recovers."""
    restore = _shrink_cap(6)
    try:
        times = T0 + np.linspace(0, 365, 9) * u.day
        toff = particle._times_to_offsets(times)
        state = particle._cartesian_state.to_system()

        # A single dense-output buffer no longer reaches the end.
        _, _, fss, _, _ = ias15_evolve(
            state,
            particle.gravity,
            toff,
            particle._integrator_state,
            particle._step_scheduler,
        )
        assert float(fss.relative_time) < float(jnp.max(toff)) - 1e-6

        # Stitched interpolation reaches the end across multiple chunks, and matches the
        # forced-landing path (whose 10k per-interval cap is unaffected by the shrink).
        xs, _, steps = stitched_interpolate(
            state,
            particle.gravity,
            toff,
            particle._integrator_state,
            particle._step_scheduler,
        )
        xf, _, _ = budgeted_forced_landing(
            state,
            particle.gravity,
            toff,
            particle._integrator_state,
            particle._step_scheduler,
        )
        assert steps > 6  # more than one chunk's worth
        assert float(jnp.max(jnp.abs(xs - xf))) < 1e-8  # AU
    finally:
        restore()


def test_particle_integrate_truncation_proof(particle: Particle) -> None:
    """Particle.integrate(_or_interpolate) reach the end even when one buffer cannot."""
    restore = _shrink_cap(6)
    try:
        times = T0 + np.linspace(0, 365, 9) * u.day
        xi, _ = particle.integrate_or_interpolate(times)
        xf, _ = particle.integrate(times)
        # The two truncation-proof paths agree on the final position.
        assert float(jnp.max(jnp.abs(xi[-1] - xf[-1]))) < 1e-8  # AU
    finally:
        restore()


# ---------------------------------------------------------------------------
# Step-count exposure (the user-facing transparency feature)
# ---------------------------------------------------------------------------


def test_ephemeris_return_steps(particle: Particle) -> None:
    """Ephemeris exposes a step count for both interpolation modes."""
    times = T0 + np.linspace(0, 60, 5) * u.day
    observer = jnp.zeros((5, 3))  # barycentric observer keeps the test offline

    coords_i, steps_i = particle.ephemeris(
        times, observer=observer, interpolate=True, return_steps=True
    )
    coords_f, steps_f = particle.ephemeris(
        times, observer=observer, interpolate=False, return_steps=True
    )
    assert int(steps_i) > 0
    assert int(steps_f) > 0
    # Dense-LTT interpolation and Taylor-LTT forced landing agree to ~uas.
    sep = coords_i.separation(coords_f).to(u.arcsec).value
    assert np.all(sep < 1e-3)


def test_system_return_steps(two_body_system: System) -> None:
    """System.integrate/ephemeris expose step counts."""
    sys = two_body_system
    times = T0 + np.linspace(0, 60, 5) * u.day
    observer = jnp.zeros((5, 3))

    _, _, steps = sys.integrate(times, return_steps=True)
    assert int(steps) > 0
    _, esteps = sys.ephemeris(times, observer=observer, return_steps=True)
    assert int(esteps) > 0


# ---------------------------------------------------------------------------
# max_steps knob: arc-sized dense-output buffers
# ---------------------------------------------------------------------------


def test_max_steps_kernel_buffers(particle: Particle) -> None:
    """The kernel's dense buffers honor max_steps and the outputs are bit-identical."""
    times = T0 + np.linspace(0, 90, 10) * u.day
    toff = particle._times_to_offsets(times)
    state = particle._cartesian_state.to_system()

    out_small = ias15_evolve_with_dense_output(
        state,
        particle.gravity,
        toff,
        particle._integrator_state,
        particle._step_scheduler,
        64,
    )
    out_default = ias15_evolve_with_dense_output(
        state,
        particle.gravity,
        toff,
        particle._integrator_state,
        particle._step_scheduler,
    )
    assert out_small[5].shape[0] == 64  # b_buf
    assert out_default[5].shape[0] == ias15_module.evolve.IAS15_MAX_DYNAMIC_STEPS
    assert float(jnp.max(jnp.abs(out_small[0] - out_default[0]))) == 0.0
    assert float(jnp.max(jnp.abs(out_small[1] - out_default[1]))) == 0.0


def test_max_steps_stitching_equivalence(particle: Particle) -> None:
    """A tiny per-chunk buffer stitches to the same answer as the default buffer."""
    times = T0 + np.linspace(0, 365, 9) * u.day
    toff = particle._times_to_offsets(times)
    state = particle._cartesian_state.to_system()

    xs_ref, vs_ref, _ = stitched_interpolate(
        state,
        particle.gravity,
        toff,
        particle._integrator_state,
        particle._step_scheduler,
    )
    xs_small, vs_small, steps_small = stitched_interpolate(
        state,
        particle.gravity,
        toff,
        particle._integrator_state,
        particle._step_scheduler,
        6,
    )
    assert steps_small > 6  # multiple chunks were actually stitched
    assert float(jnp.max(jnp.abs(xs_small - xs_ref))) < 1e-12  # AU
    assert float(jnp.max(jnp.abs(vs_small - vs_ref))) < 1e-14  # AU/day


def test_particle_ias15_max_steps_knob(particle: Particle) -> None:
    """The Particle-level knob reproduces the default paths on integrate/ephemeris."""
    p_small = Particle(
        x=X0, v=V0, time=T0, gravity="newtonian planets", ias15_max_steps=8
    )
    times = T0 + np.linspace(0, 365, 9) * u.day

    xi_ref, _ = particle.integrate_or_interpolate(times)
    xi_small, _ = p_small.integrate_or_interpolate(times)
    assert float(jnp.max(jnp.abs(xi_small - xi_ref))) < 1e-12  # AU

    observer = jnp.zeros((9, 3))
    c_ref = particle.ephemeris(times, observer=observer)
    c_small = p_small.ephemeris(times, observer=observer)
    assert np.all(c_ref.separation(c_small).to(u.mas).value < 1e-6)


# ---------------------------------------------------------------------------
# Ephemeris-window detection in the no-progress error path
# ---------------------------------------------------------------------------


def test_loaded_ephemeris_bounds(particle: Particle) -> None:
    """The bounds introspected from an acceleration function match the loaded window."""
    bounds = _loaded_ephemeris_bounds_jd(particle.gravity)
    assert bounds is not None
    start_jd, end_jd = bounds

    # The particle's window is the default 1980-01-01/2050-01-01, padded by 100 days
    # and snapped inward to Chebyshev interval boundaries (<= 32 days each).
    early_padded = Time("1980-01-01", scale="tdb").jd - 100.0
    late_padded = Time("2050-01-01", scale="tdb").jd + 100.0
    assert early_padded - 33.0 < start_jd <= early_padded + 1.0
    assert late_padded - 33.0 < end_jd <= late_padded + 1.0

    # Ephemeris-free gravity has no window to report.
    assert _loaded_ephemeris_bounds_jd(jax.tree_util.Partial(newtonian_gravity)) is None


def test_out_of_window_times_raise(particle: Particle, two_body_system: System) -> None:
    """Public methods refuse times outside the loaded ephemeris window.

    Out-of-window queries wrap the Chebyshev interval lookup to wrong-era
    coefficients and previously returned smooth garbage (they only sometimes
    stalled into the no-progress error). The window check in _times_to_offsets
    now fails loudly up front on every public path.
    """
    with pytest.raises(ValueError, match="ephemeris window") as excinfo:
        particle.integrate_or_interpolate(Time("1975-01-01"))
    assert "`earliest_time`/`latest_time`" in str(excinfo.value)
    with pytest.raises(ValueError, match="ephemeris window"):
        particle.integrate(Time("2055-01-01"))
    with pytest.raises(ValueError, match="ephemeris window"):
        particle.ephemeris(Time("2055-01-01"), observer=jnp.zeros((1, 3)))
    with pytest.raises(ValueError, match="ephemeris window"):
        two_body_system.integrate(Time("1975-01-01"))
    with pytest.raises(ValueError, match="ephemeris window"):
        two_body_system.ephemeris(Time("1975-01-01"), observer=jnp.zeros((1, 3)))

    # Keplerian particles have no ephemeris, hence no window to enforce.
    p_kep = Particle(x=X0, v=V0, time=T0, gravity="keplerian")
    x, _ = p_kep.integrate(Time("1975-01-01"))
    assert np.all(np.isfinite(np.asarray(x)))


def test_no_progress_error_window_detection(particle: Particle) -> None:
    """Stalls at/beyond the loaded ephemeris window name the bound; others stay generic."""
    start_jd, end_jd = _loaded_ephemeris_bounds_jd(particle.gravity)
    t_ref = float(Time("2025-01-01", scale="tdb").jd)

    # Backward integration off the early edge (the bug-report scenario).
    stall = (start_jd - 5.0) - t_ref
    err = _no_progress_error(stall, stall - 1500.0, -1.0, t_ref, particle.gravity)
    msg = str(err)
    assert "earliest" in msg
    assert "`earliest_time`" in msg
    assert f"JD {start_jd:.2f}" in msg
    assert "ephemeris window" in msg

    # Forward integration off the late edge.
    stall = (end_jd + 5.0) - t_ref
    err = _no_progress_error(stall, stall + 1500.0, 1.0, t_ref, particle.gravity)
    assert "`latest_time`" in str(err)

    # A stall well inside the window keeps the generic dynamics message.
    err = _no_progress_error(100.0, 200.0, 1.0, t_ref, particle.gravity)
    assert "NaN acceleration" in str(err)
    assert "ephemeris window" not in str(err)

    # No ephemeris data reachable -> generic message even off-window.
    err = _no_progress_error(
        (start_jd - 5.0) - t_ref,
        (start_jd - 1500.0) - t_ref,
        -1.0,
        t_ref,
        jax.tree_util.Partial(newtonian_gravity),
    )
    assert "NaN acceleration" in str(err)


# ---------------------------------------------------------------------------
# Detect-and-raise on the autodiff covariance path
# ---------------------------------------------------------------------------


def test_cov_ephemeris_raises_on_truncation() -> None:
    """ephemeris(uncertainty=True) raises (not truncates) on an over-long span."""
    c = CartesianState(
        x=jnp.array([X0]),
        v=jnp.array([V0]),
        time_reference=T0.tdb.jd,
        acceleration_func_kwargs={"c2": SPEED_OF_LIGHT**2},
        cov=jnp.eye(6) * 1e-12,
    )
    p = Particle(state=c, gravity="newtonian planets")
    restore = _shrink_cap(6)
    try:
        times = T0 + np.linspace(0, 365, 9) * u.day
        observer = jnp.zeros((9, 3))
        with pytest.raises(RuntimeError, match="truncate"):
            p.ephemeris(times, observer=observer, uncertainty=True)
    finally:
        restore()


# ---------------------------------------------------------------------------
# Non-IAS15 paths are untouched
# ---------------------------------------------------------------------------


def test_keplerian_return_steps_is_none() -> None:
    """The analytic Keplerian path reports None steps (it cannot truncate)."""
    p = Particle(x=X0, v=V0, time=T0, gravity="keplerian")
    times = T0 + np.linspace(0, 30, 4) * u.day
    observer = jnp.zeros((4, 3))

    _, _, steps = p.integrate(times, return_steps=True)
    assert steps is None
    _, esteps = p.ephemeris(times, observer=observer, return_steps=True)
    assert esteps is None

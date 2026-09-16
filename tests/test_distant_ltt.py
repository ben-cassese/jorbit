"""Regression tests for the dense-output light-travel-time (LTT) correction.

The dense-LTT ``on_sky`` path used to evaluate the polynomial of the step containing the
*observation* time at the *retarded* time ``t_obs - LTT``. Whenever the light travel time
exceeded that step's length, the 7th-order polynomial was used as an extrapolant and the
sky position degraded badly: ~3 arcsec at 127 AU on the short arc below, where every
dense step is arc-limited, and up to tens of arcsec near perihelion, where the adaptive
controller shortens steps to ~1e-3 days.

It now looks the retarded time up in the dense buffer and interpolates the step that
actually contains it. Two things have to hold for that to be well-defined, and both are
tested here: the propagator must select by time rather than stay pinned to one step, and
the buffer must reach far enough back to contain the retarded time of even the earliest
observation (which precedes the integration epoch).
"""

import jax

jax.config.update("jax_enable_x64", True)

import astropy.units as u
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time

import jorbit.particle.ias15_dense as particle_dense
import jorbit.system.ias15_dense as system_dense
from jorbit import Observations, Particle, System
from jorbit.data.constants import SPEED_OF_LIGHT
from jorbit.integrators import (
    DenseOutput,
    apply_ltt_seed_floor,
    assert_ltt_span_covered,
    budgeted,
    ias15_evolve_with_dense_output,
    initialize_ias15_integrator_state,
    interpolate_from_dense_output,
    ltt_seed_floor,
    ltt_span_shortfall,
    make_ltt_propagator,
    precompute_interpolation_indices,
    stitched_dense_buffers,
)
from jorbit.integrators.ias15 import interpolation
from jorbit.integrators.ias15.interpolation import _covered_end
from jorbit.utils.states import KeplerianState

EPOCH = Time("2025-01-01", scale="tdb")
GM_SUN = 2.9591220828559115e-04  # AU^3/day^2, only used to build plausible orbits

# A single 3-observation tracklet: first obs at the particle epoch itself, which is
# the worst case (the emission time predates the whole integration span).
SHORT_ARC = EPOCH + jnp.array([0.0, 0.0055, 0.011]) * u.day


def _circular_particle(r: float) -> Particle:
    """A near-circular test particle at barycentric distance ``r`` AU."""
    x = jnp.array([r / np.sqrt(2), r / np.sqrt(2), 0.0])
    vcirc = np.sqrt(GM_SUN / r)
    v = jnp.array([-vcirc / np.sqrt(2), vcirc / np.sqrt(2), 0.0])
    return Particle(x=x, v=v, time=EPOCH, gravity="default solar system")


def test_distant_short_arc_dense_ltt_accuracy() -> None:
    """Dense-output ephemeris matches the forced-landing (Taylor-LTT) reference.

    The forced-landing path lands exactly on each observation time and applies the
    constant-acceleration Taylor LTT correction, which is essentially exact for a
    distant near-linear orbit; it is unaffected by the extrapolation bug. Before the
    seed floor this comparison failed at ~3e3 mas on the epoch observation.
    """
    p = _circular_particle(127.0)
    eph_forced = p.ephemeris(SHORT_ARC, "kitt peak", interpolate=False)
    eph_dense = p.ephemeris(SHORT_ARC, "kitt peak", interpolate=True)
    seps = eph_dense.separation(eph_forced).to(u.mas)
    assert np.all(seps < 0.1 * u.mas)


def test_distant_short_arc_system_forward_model() -> None:
    """The batched System forward model (_ephem_ias15_bounded) gets the same fix."""
    p = _circular_particle(127.0)
    eph_forced = p.ephemeris(SHORT_ARC, "kitt peak", interpolate=False)
    obs = Observations(
        observed_coordinates=eph_forced,
        times=SHORT_ARC,
        observatories="kitt peak",
        astrometric_uncertainties=0.1 * u.arcsec,
    )
    system = System(particles=[p], observations=obs, gravity="default solar system")

    truth = jnp.concatenate([jnp.asarray(p._x), jnp.asarray(p._v)])[None, :]
    ras, decs = system.model_radec(truth)
    model_sc = SkyCoord(
        ra=np.asarray(ras[0]), dec=np.asarray(decs[0]), unit=u.rad, frame="icrs"
    )
    seps = model_sc.separation(eph_forced).to(u.mas)
    assert np.all(seps < 0.1 * u.mas)
    # Before the seed floor this was ~1e3 (a jagged, biased likelihood surface).
    assert float(system.chi2(truth)[0]) < 1e-6


def test_near_object_short_arc_unchanged() -> None:
    """The changed step seeding does not degrade a nearby object on the same arc."""
    p = _circular_particle(2.5)
    eph_forced = p.ephemeris(SHORT_ARC, "kitt peak", interpolate=False)
    eph_dense = p.ephemeris(SHORT_ARC, "kitt peak", interpolate=True)
    seps = eph_dense.separation(eph_forced).to(u.mas)
    assert np.all(seps < 0.2 * u.mas)


def test_dense_ltt_propagator_interpolates_at_every_excursion() -> None:
    """The propagator interpolates the containing step, however far back the emission is.

    The unit-level form of the bug: on an eccentric inner-solar-system orbit the
    adaptive controller takes steps from 1e-3 to ~3.6 days, so a query near perihelion
    sits in a step far shorter than any plausible light travel time. Ask the propagator
    for positions 1 to 20 step lengths before the observation and compare against
    interpolating the step that actually contains each of those times. Extrapolating a
    single step instead reaches ~1e-3 AU (hundreds of arcsec at 1 AU) by 20 step lengths.

    The excursion is chosen rather than waited for on purpose: whether the adaptive
    controller realises a given ``LTT/dt_step`` ratio is not reproducible across
    machines, so an end-to-end test that waits for one would be flaky.
    """
    t_ref = 2458849.5
    particle = Particle(
        state=KeplerianState(
            semi=jnp.asarray([0.55]),
            ecc=jnp.asarray([0.80]),
            inc=jnp.asarray([7.0]),
            Omega=jnp.asarray([80.0]),
            omega=jnp.asarray([40.0]),
            nu=jnp.asarray([170.0]),
            acceleration_func_kwargs={"c2": SPEED_OF_LIGHT**2},
            time_reference=t_ref,
        ),
        gravity="default solar system",
        step_scheduler="global",
    )
    state = particle._cartesian_state.to_system()
    integrator_state = initialize_ias15_integrator_state(particle.gravity(state))
    times = jnp.linspace(state.relative_time + 0.5, state.relative_time + 120.0, 200)
    out = ias15_evolve_with_dense_output(
        state, particle.gravity, times, integrator_state, particle._step_scheduler, 4096
    )
    b, a0, x0, v0, dts, t_step_starts = out[5:11]
    step_indices = out[11]
    dense = DenseOutput(*out[5:11])

    # The query whose containing step is shortest: the perihelion regime that produces
    # light-travel-time-to-step-length ratios well above one.
    q = int(np.argmin(np.abs(np.asarray(dts))[np.asarray(step_indices)]))
    dt_step = float(dts[int(step_indices[q])])
    t_obs = float(times[q])
    propagator = make_ltt_propagator(
        dense, None, state.relative_time, jnp.asarray(t_obs)
    )

    for excursion in (1.0, 2.0, 5.0, 10.0, 20.0):
        delta = excursion * dt_step
        idx, h = precompute_interpolation_indices(
            t_step_starts, dts, jnp.asarray([t_obs - delta])
        )
        reference, _ = interpolate_from_dense_output(b, a0, x0, v0, dts, idx, h)
        err = float(jnp.linalg.norm(propagator(jnp.asarray(-delta)) - reference[0, 0]))
        assert err < 1e-14, f"{excursion}x step lengths back: {err:.3e} AU"


def test_distant_short_arc_accurate_without_seed_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The backward span, not the step seeding, is what protects the distant short arc.

    Disabling the seed floor recreates the old arc-limited step schedule (every dense
    step ~1e-3 days against a 0.73 day light travel time). That used to produce ~3
    arcsec errors; it is now harmless, because the backward pass is extended past the
    epoch far enough to contain the retarded times.
    """
    # The seed floor is applied inside stitched_dense_buffers now, so this has to patch
    # jorbit.integrators.budgeted -- patching particle_dense would silently do nothing and
    # leave the test passing while testing the opposite of what it claims.
    monkeypatch.setattr(budgeted, "apply_ltt_seed_floor", lambda state, *args: state)
    p = _circular_particle(127.0)
    eph_forced = p.ephemeris(SHORT_ARC, "kitt peak", interpolate=False)
    eph_dense = p.ephemeris(SHORT_ARC, "kitt peak", interpolate=True)
    seps = eph_dense.separation(eph_forced).to(u.mas)
    assert np.all(seps < 0.1 * u.mas)


def test_uncovered_retarded_time_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    """A retarded time outside the dense buffer is covered by extending, not refused.

    Zeroing the backward padding leaves the 127 AU tracklet's emission times 0.73 days
    before the start of the integration, where there is no step to interpolate. That used
    to raise; ``stitched_dense_buffers`` now measures the shortfall and extends the backward
    pass until it covers, so the answer comes back and is correct.
    """
    monkeypatch.setattr(budgeted, "ltt_seed_floor", lambda *args: 0.0)
    p = _circular_particle(127.0)
    eph_forced = p.ephemeris(SHORT_ARC, "kitt peak", interpolate=False)
    eph_dense = p.ephemeris(SHORT_ARC, "kitt peak", interpolate=True)
    seps = eph_dense.separation(eph_forced).to(u.mas)
    assert np.all(seps < 0.1 * u.mas)


def _distant_observer_setup(
    obs_distance: float = 500.0,
) -> tuple[Particle, jnp.ndarray, jnp.ndarray, float]:
    """A nearby particle watched from far away, so the light travel time dwarfs the arc.

    Decouples the required backward reach (set by the observer distance, ~2.9 days at
    500 AU) from the particle's own dynamics, which is what lets the tests below dial the
    overshoot regime in directly.
    """
    p = _circular_particle(2.5)
    state = p._cartesian_state.to_system()
    t0 = float(state.relative_time)
    times = jnp.asarray([t0, t0 + 1.0])
    observer = jnp.tile(jnp.asarray([obs_distance, 0.0, 0.0]), (len(times), 1))
    required = (
        float(jnp.linalg.norm(state.tracer_positions[0] - observer[0])) / SPEED_OF_LIGHT
    )
    return p, times, observer, required


def test_assert_ltt_span_covered_still_raises_on_an_uncovered_buffer() -> None:
    """The bare check and its non-raising twin agree, on the downstream-pinned fixture.

    This is the reproduction a downstream user pins their own workaround against, so the
    function name, the message and the measured shortfall all have to keep working when
    called directly. ``stitched_dense_buffers`` only runs the check when it is given
    ``observer_positions``, so building without them leaves the buffers uncovered on
    purpose.
    """
    p, times, observer, required = _distant_observer_setup()
    state = p._cartesian_state.to_system()
    t0 = float(state.relative_time)
    fwd, bwd, _steps = stitched_dense_buffers(
        state,
        p.gravity,
        times,
        initialize_ias15_integrator_state(p.gravity(state)),
        p._step_scheduler,
        64,
        backward_pad=0.0,
    )

    # One observation mid-arc, so its retarded time is interior for a nearby observer.
    obs_times = jnp.asarray([t0 + 0.5])

    # A nearby observer is covered by the forward buffer alone.
    near = jnp.asarray([[1.0, 0.0, 0.0]])
    assert_ltt_span_covered(fwd, bwd, t0, obs_times, near)
    _retarded, shortfall = ltt_span_shortfall(fwd, bwd, t0, obs_times, near)
    assert shortfall == 0.0

    # A 500 AU observer is not: its light travel time reaches back past the epoch, and
    # with no backward pad there is no step there to interpolate.
    far = observer[:1]
    _retarded, shortfall = ltt_span_shortfall(fwd, bwd, t0, obs_times, far)
    assert shortfall > 0.0
    assert shortfall == pytest.approx(required - 0.5, abs=0.05)
    with pytest.raises(RuntimeError, match="outside the integrated span"):
        assert_ltt_span_covered(fwd, bwd, t0, obs_times, far)


def test_backward_pass_extends_past_a_large_overshoot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The extension targets the measured retarded time, not a multiple of the pad.

    A backward pass stops at the end of the first natural step *past* its target, so what
    it actually covers is the pad plus that overshoot. Sizing an extension from the pad
    alone (``pad + 2 * shortfall``, the obvious formula) lands *inside* the existing buffer
    whenever the overshoot is at least twice the shortfall -- and because the adaptive steps
    are natural, the rebuilt pass then stops at exactly the same step and changes nothing,
    so the check raises again. Roughly a third of shortfalls fall in that window.

    This test pins the regime rather than hoping to land in it: it asserts
    ``overshoot >= 2 * shortfall`` on the first pass, so if a future change makes the
    geometry degenerate the test fails instead of quietly stopping to exercise the bug.
    """
    p, times, observer, required = _distant_observer_setup()
    state = p._cartesian_state.to_system()
    t0 = float(state.relative_time)
    pad = 0.01
    # Seed the first backward step at 0.7x the required reach. The controller accepts it
    # as-is, so the first pass covers ~0.7 * required: short of what is needed, but
    # overshooting its own target by far more than the remaining shortfall.
    seeded = initialize_ias15_integrator_state(p.gravity(state)).replace(
        dt=jnp.asarray(0.7 * required)
    )

    # First pass only: without observer_positions there is no seed floor, no extension and
    # no check, so the seeded dt above is what the pass actually starts from.
    fwd, bwd, _steps = stitched_dense_buffers(
        state,
        p.gravity,
        times,
        seeded,
        p._step_scheduler,
        None,
        backward_pad=pad,
    )
    _retarded, shortfall = ltt_span_shortfall(fwd, bwd, t0, times, observer)
    overshoot = (float(jnp.min(times)) - pad) - _covered_end(bwd)
    assert shortfall > 0.0, "geometry no longer produces a shortfall"
    assert overshoot >= 2 * shortfall, (
        f"geometry no longer exercises the no-op window: overshoot {overshoot:.4f} < "
        f"2 * shortfall {2 * shortfall:.4f}"
    )

    # Now with the coverage contract on. Patch the seed floor to reproduce the same first
    # backward step, and the pad to the same small value, so the only difference from the
    # run above is that the extension is allowed to happen.
    monkeypatch.setattr(
        interpolation, "ltt_seed_floor", lambda *a: jnp.asarray(0.7 * required)
    )
    monkeypatch.setattr(budgeted, "ltt_seed_floor", lambda *a: jnp.asarray(pad))
    fwd2, bwd2, _steps2 = stitched_dense_buffers(
        state,
        p.gravity,
        times,
        initialize_ias15_integrator_state(p.gravity(state)),
        p._step_scheduler,
        None,
        obs_times=times,
        observer_positions=observer,
    )
    retarded_min, shortfall2 = ltt_span_shortfall(fwd2, bwd2, t0, times, observer)
    assert shortfall2 == 0.0
    assert _covered_end(bwd2) <= retarded_min


def test_extension_is_bitwise_non_perturbing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Extending the backward pass cannot move an already-covered observation.

    The adaptive steps are natural -- the target only ends the integration loop, it never
    clamps a step -- so a deeper backward target produces a prefix-identical superset of the
    shorter pass. Anything the short buffer already covered therefore interpolates the very
    same polynomial. Equality here is exact, not a tolerance.
    """
    p, times, _observer, required = _distant_observer_setup()
    state = p._cartesian_state.to_system()

    def buffers(pad: float) -> DenseOutput:
        return stitched_dense_buffers(
            state,
            p.gravity,
            times,
            initialize_ias15_integrator_state(p.gravity(state)),
            p._step_scheduler,
            None,
            backward_pad=pad,
        )[1]

    short, long = buffers(0.5 * required), buffers(4.0 * required)
    n_short = int(jnp.sum(jnp.abs(short.dts) < 1e29))
    assert (
        int(jnp.sum(jnp.abs(long.dts) < 1e29)) > n_short
    ), "not actually a longer pass"
    for name in ("b", "a0", "x0", "v0", "dts", "t_step_starts"):
        a = np.asarray(getattr(short, name))[:n_short]
        b = np.asarray(getattr(long, name))[:n_short]
        assert np.array_equal(a, b), f"{name} differs in the shared prefix"


def test_bounded_path_reports_and_poisons_an_ltt_shortfall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The jitted System path cannot extend or raise, so it reports and poisons instead.

    The traced counterpart of the host-side check: same predicate, same tolerance, but the
    answer comes back as a per-observation mask. Values that fail it are poisoned exactly
    the way a buffer truncation is, since an extrapolated polynomial has no error bound.
    """
    p, times, observer, _required = _distant_observer_setup()
    state = p._cartesian_state.to_system()
    t0 = float(state.relative_time)
    times_off = times - t0
    fwd_mask = times_off >= 0.0
    states = jnp.concatenate((state.tracer_positions, state.tracer_velocities), axis=1)

    def run() -> tuple:
        # _ephem_ias15_bounded is jitted, and it resolves ltt_seed_floor from its module
        # globals at trace time, so a cached executable would silently ignore the patches
        # below and leave this test asserting nothing.
        system_dense._ephem_ias15_bounded.clear_cache()
        return system_dense._ephem_ias15_bounded(
            states,
            times_off,
            fwd_mask,
            jnp.where(fwd_mask, times_off, 0.0),
            jnp.where(fwd_mask, 0.0, times_off),
            observer,
            jnp.asarray(float(p._t_ref_jd)),
            p.gravity,
            p._step_scheduler,
        )

    # The real seed floor covers this geometry comfortably.
    _ras, _decs, reached, ltt_covered = run()
    assert bool(jnp.all(reached))
    assert bool(jnp.all(ltt_covered))

    # Starve the pad and the mask drops, on the observation whose retarded time is worst.
    monkeypatch.setattr(system_dense, "ltt_seed_floor", lambda *a: jnp.asarray(0.0))
    monkeypatch.setattr(interpolation, "ltt_seed_floor", lambda *a: jnp.asarray(1e-6))
    _ras, _decs, reached, ltt_covered = run()
    assert bool(jnp.all(reached)), "truncation would confound this test"
    assert not bool(jnp.all(ltt_covered))


def test_system_callables_poison_an_ltt_shortfall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An uncovered retarded time is poisoned exactly the way a truncation is.

    The user-facing half of the contract. ``model_radec``/``residuals`` go ``NaN`` per
    observation and ``loglike``/``chi2`` collapse batch-wide, so a sampler rejects a step
    it cannot evaluate rather than accepting a polynomial extrapolated outside its own
    interval. ``model_radec_with_status`` still separates the two causes.
    """
    p = _circular_particle(127.0)
    eph_forced = p.ephemeris(SHORT_ARC, "kitt peak", interpolate=False)
    obs = Observations(
        observed_coordinates=eph_forced,
        times=SHORT_ARC,
        observatories="kitt peak",
        astrometric_uncertainties=0.1 * u.arcsec,
    )
    system = System(particles=[p], observations=obs, gravity="default solar system")
    truth = jnp.concatenate([jnp.asarray(p._x), jnp.asarray(p._v)])[None, :]

    # Starve the backward pad so the 0.73 day light travel time reaches off the buffer.
    monkeypatch.setattr(system_dense, "ltt_seed_floor", lambda *a: jnp.asarray(0.0))
    monkeypatch.setattr(interpolation, "ltt_seed_floor", lambda *a: jnp.asarray(1e-6))
    system_dense._ephem_ias15_bounded.clear_cache()

    ras, decs, reached, ltt_covered = system.model_radec_with_status(truth)
    assert bool(jnp.all(reached)), "truncation would confound this test"
    assert not bool(jnp.all(ltt_covered))

    bad = ~np.asarray(ltt_covered)
    assert np.array_equal(np.isnan(np.asarray(ras))[0], bad)
    assert np.array_equal(np.isnan(np.asarray(decs))[0], bad)
    assert np.all(np.isnan(np.asarray(system.residuals(truth))[0][bad]))
    assert float(system.loglike(truth)[0]) == -np.inf
    assert float(system.chi2(truth)[0]) == np.inf


def test_host_and_traced_coverage_checks_agree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two guards are wrappers over one predicate, so they must flag the same rows."""
    p, times, observer, _required = _distant_observer_setup()
    state = p._cartesian_state.to_system()
    t0 = float(state.relative_time)
    times_off = times - t0
    fwd_mask = times_off >= 0.0

    monkeypatch.setattr(system_dense, "ltt_seed_floor", lambda *a: jnp.asarray(0.0))
    monkeypatch.setattr(interpolation, "ltt_seed_floor", lambda *a: jnp.asarray(1e-6))
    system_dense._ephem_ias15_bounded.clear_cache()
    _ras, _decs, reached, ltt_covered = system_dense._ephem_ias15_bounded(
        jnp.concatenate((state.tracer_positions, state.tracer_velocities), axis=1),
        times_off,
        fwd_mask,
        jnp.where(fwd_mask, times_off, 0.0),
        jnp.where(fwd_mask, 0.0, times_off),
        observer,
        jnp.asarray(float(p._t_ref_jd)),
        p.gravity,
        p._step_scheduler,
    )
    assert bool(jnp.all(reached))

    # Same geometry, host side: build the same starved buffers and measure per-observation.
    fwd, bwd, _steps = stitched_dense_buffers(
        state,
        p.gravity,
        times,
        initialize_ias15_integrator_state(p.gravity(state)),
        p._step_scheduler,
        None,
        backward_pad=1e-6,
    )
    excursion, _ltts, _retarded, _lo, _hi = interpolation._ltt_span_check(
        fwd, bwd, t0, times, observer
    )
    host_failed = np.asarray(excursion) > 1e-9
    assert np.array_equal(host_failed, ~np.asarray(ltt_covered))


def test_covariance_path_raises_on_uncovered_ltt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ephemeris(uncertainty=True) refuses rather than returning an extrapolation.

    Its buffers are built inside ``jax.jacfwd``, so unlike the nominal path it cannot
    extend them. It gets the same mask the batched System path does and raises on it.
    """
    p = _circular_particle(127.0)
    monkeypatch.setattr(particle_dense, "ltt_seed_floor", lambda *args: 0.0)
    # Same jit-cache hazard as the bounded path: _ephem_ias15 bakes in whatever
    # ltt_seed_floor resolved to when it was first traced.
    particle_dense._ephem_ias15.clear_cache()
    particle_dense._ephem_ias15_with_cov.clear_cache()
    state = p.cartesian_state.replace(cov=jnp.eye(6) * 1e-16)
    with pytest.raises(RuntimeError, match="outside the integrated span"):
        p.ephemeris(SHORT_ARC, "kitt peak", state=state, uncertainty=True)


# ---------------------------------------------------------------------------
# Where the backward pass is anchored (1.6.5)
#
# 1.6.3 extended the backward pass past the earliest requested time so that the
# retarded time of even the first observation lands inside a real step. All three
# implementations of that actually anchored the extension at the *epoch*
# (``t0 - pad``), which only reaches past the earliest requested time when that time
# happens to fall inside ``[t0 - pad, t0]``. Two consequences: a state epoch sitting
# one to two light times after the first observation put the retarded time off the
# back of the buffer (a raise on the checked paths, a silent extrapolation on the two
# jitted ones), and an ordinary long pre-epoch arc got no extension at all, leaving
# coverage to however far the last natural step happened to overshoot.
# ---------------------------------------------------------------------------


def _observer_positions(times_jd: np.ndarray) -> jnp.ndarray:
    """A synthetic 1 AU observer track, so these tests need no Horizons query.

    ``on_sky`` is purely geometric -- three fixed-point iterations of the light travel
    time, then a unit vector -- so any smooth observer track exercises the correction
    the same way a real observatory does.
    """
    theta = 2 * np.pi * (np.asarray(times_jd, dtype=float) - 2451545.0) / 365.25
    return jnp.asarray(
        np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=-1)
    )


def _eccentric_particle() -> Particle:
    """An eccentric sub-AU orbit sitting near perihelion at its epoch.

    The regime where the adaptive controller takes ~1e-3 day steps, so the overshoot
    past the earliest requested time is far shorter than the light travel time.
    """
    return Particle(
        state=KeplerianState(
            semi=jnp.asarray([0.55]),
            ecc=jnp.asarray([0.80]),
            inc=jnp.asarray([7.0]),
            Omega=jnp.asarray([80.0]),
            omega=jnp.asarray([40.0]),
            nu=jnp.asarray([5.0]),
            acceleration_func_kwargs={"c2": SPEED_OF_LIGHT**2},
            time_reference=EPOCH.tdb.jd,
        ),
        gravity="default solar system",
    )


def _pad_days(p: Particle) -> float:
    """The backward pad the dense paths will use for this particle, in days."""
    state = p._cartesian_state.to_system()
    obs_pos = _observer_positions(np.array([float(p._t_ref_jd)]))
    return float(ltt_seed_floor(state.tracer_positions, obs_pos))


def _shifted_arc(p: Particle, pre_pads: float) -> tuple[np.ndarray, float]:
    """A 5-point arc whose first point precedes the epoch by ``pre_pads`` pad lengths."""
    pad = _pad_days(p)
    t0 = float(p._t_ref_jd)
    arc_len = max(4.0 * pad, 0.05)
    return t0 - pre_pads * pad + np.linspace(0.0, arc_len, 5), pad


@pytest.mark.parametrize("pre_pads", [-3.0, 0.0, 0.25, 0.9, 2.0, 20.0])
@pytest.mark.parametrize(
    "make_particle",
    [lambda: _circular_particle(127.0), _eccentric_particle],
    ids=["distant", "perihelion"],
)
def test_backward_span_covers_retarded_times(
    make_particle: object, pre_pads: float
) -> None:
    """The backward buffer reaches a full pad past the earliest requested time.

    This is the invariant the dense-LTT correction needs, stated directly on the
    buffer rather than inferred from a sky position: a light travel time is at most
    half a pad, so a buffer that reaches ``min(times) - pad`` contains the retarded
    time of every observation whatever the epoch geometry or the step sizes. Anchoring
    the pad at the epoch satisfied it only for ``0 <= pre_pads <= 1``, and even there
    only by accident.
    """
    p = make_particle()
    times_jd, pad = _shifted_arc(p, pre_pads)
    obs_pos = _observer_positions(times_jd)
    times_off = p._times_to_offsets(Time(times_jd, format="jd", scale="tdb"))

    state = p._cartesian_state.to_system()
    integrator_state = initialize_ias15_integrator_state(p.gravity(state))
    integrator_state = apply_ltt_seed_floor(
        integrator_state, state.tracer_positions, obs_pos
    )
    _fwd, bwd, _steps = stitched_dense_buffers(
        state,
        p.gravity,
        times_off,
        integrator_state,
        p._step_scheduler,
        None,
        backward_pad=pad,
    )

    t_min = float(jnp.min(times_off))
    shortfall = _covered_end(bwd) - (t_min - pad)
    assert shortfall <= 1e-9, (
        f"backward buffer ends {shortfall:.6f} days short of a full pad "
        f"({pad:.6f} d) before the earliest requested time"
    )


@pytest.mark.parametrize("pre_pads", [0.25, 0.9, 2.0])
@pytest.mark.parametrize("r_au", [127.0, 2.5])
def test_epoch_after_first_observation_builds(r_au: float, pre_pads: float) -> None:
    """An arc that starts a light time or two before the epoch still builds, correctly.

    ``pre_pads=0.9`` is the geometry a downstream user hit on real data: the epoch sits
    inside the pad but after the first observation, so the epoch-anchored clamp stopped
    the backward pass exactly at that observation and ``assert_ltt_span_covered``
    raised. The other two offsets are controls that always worked.
    """
    p = _circular_particle(r_au)
    times_jd, _pad = _shifted_arc(p, pre_pads)
    times = Time(times_jd, format="jd", scale="tdb")
    obs_pos = _observer_positions(times_jd)

    eph_forced = p.ephemeris(times, obs_pos, interpolate=False)
    eph_dense = p.ephemeris(times, obs_pos, interpolate=True)
    seps = eph_dense.separation(eph_forced).to(u.mas)
    assert np.all(seps < 0.2 * u.mas)


def test_all_dense_paths_agree_on_shifted_epoch() -> None:
    """The three dense backends agree on the geometry that used to break one of them.

    The epoch-anchored clamp was written out separately in all three, and only
    ``_ephem_ias15_stitched`` runs a host-side coverage check -- ``_ephem_ias15`` (the
    ``uncertainty=True`` Jacobian path) and ``_ephem_ias15_bounded`` (the batched System
    forward model) would have extrapolated in silence. In practice they did not: their
    answers here are bit-identical before and after the fix, because a 7th-order
    polynomial carried a fraction of a step past its own interval still resolves these
    geometries to the last double. The check that matters is therefore this one --
    three copies of one formula staying in step with each other -- not a tolerance.
    They are called directly because that is where the three copies live.
    """
    p = _circular_particle(127.0)
    times_jd, _pad = _shifted_arc(p, 0.9)
    obs_pos = _observer_positions(times_jd)
    times_off = p._times_to_offsets(Time(times_jd, format="jd", scale="tdb"))
    relevant_inds = jnp.arange(times_off.shape[0])

    state = p._cartesian_state
    sys_state = state.to_system()
    integrator_state = initialize_ias15_integrator_state(p.gravity(sys_state))

    ra_stitched, dec_stitched, _steps = particle_dense._ephem_ias15_stitched(
        times_off,
        state,
        p.gravity,
        integrator_state,
        obs_pos,
        relevant_inds,
        p._step_scheduler,
    )
    ra_jit, dec_jit, ltt_jit = particle_dense._ephem_ias15(
        times_off,
        state,
        p.gravity,
        integrator_state,
        obs_pos,
        relevant_inds,
        p._step_scheduler,
    )
    assert bool(jnp.all(ltt_jit))
    fwd_mask = times_off >= 0.0
    ra_bounded, dec_bounded, reached, ltt_bounded = system_dense._ephem_ias15_bounded(
        jnp.concatenate(
            (sys_state.tracer_positions, sys_state.tracer_velocities), axis=1
        ),
        times_off,
        fwd_mask,
        jnp.where(fwd_mask, times_off, 0.0),
        jnp.where(fwd_mask, 0.0, times_off),
        obs_pos,
        jnp.asarray(float(p._t_ref_jd)),
        p.gravity,
        p._step_scheduler,
    )
    assert bool(jnp.all(reached))
    assert bool(jnp.all(ltt_bounded))

    def _sc(ras: jnp.ndarray, decs: jnp.ndarray) -> SkyCoord:
        return SkyCoord(
            ra=np.asarray(ras), dec=np.asarray(decs), unit=u.rad, frame="icrs"
        )

    reference = _sc(ra_stitched, dec_stitched)
    for name, ras, decs in [
        ("jacfwd", ra_jit, dec_jit),
        ("bounded", ra_bounded[0], dec_bounded[0]),
    ]:
        seps = _sc(ras, decs).separation(reference).to(u.mas)
        assert np.all(seps < 0.1 * u.mas), f"{name}: {np.max(seps)}"


def test_dense_ltt_matches_independent_retarded_integration() -> None:
    """The dense-LTT sky position matches one built with no polynomial at all.

    Every other test here compares the dense path against the forced-landing path,
    which shares ``on_sky``'s constant-acceleration Taylor expansion. This one replaces
    the propagator outright: it runs the same three-iteration fixed point that
    ``on_sky`` runs, but evaluates the position at each trial retarded time with a
    forced-landing IAS15 integration that arrives exactly there. Nothing is
    interpolated and nothing is Taylor-expanded, so a sign error, a frame error, or a
    mis-selected step anywhere in the dense-LTT chain shows up as a sky offset.
    """
    p = _circular_particle(2.5)
    times = EPOCH + np.linspace(-15.0, 15.0, 5) * u.day
    times_jd = np.asarray(times.tdb.jd)
    obs_pos = np.asarray(_observer_positions(times_jd))

    eph_dense = p.ephemeris(times, jnp.asarray(obs_pos), interpolate=True)

    x_ret = np.asarray(p.integrate(times)[0])
    for _ in range(3):
        ltts = np.linalg.norm(x_ret - obs_pos, axis=1) / SPEED_OF_LIGHT
        retarded = Time(times_jd - ltts, format="jd", scale="tdb")
        x_ret = np.asarray(p.integrate(retarded)[0])

    sightlines = x_ret - obs_pos
    ra_ref = np.mod(np.arctan2(sightlines[:, 1], sightlines[:, 0]), 2 * np.pi)
    dec_ref = np.pi / 2 - np.arccos(
        sightlines[:, 2] / np.linalg.norm(sightlines, axis=1)
    )
    reference = SkyCoord(ra=ra_ref, dec=dec_ref, unit=u.rad, frame="icrs")

    seps = eph_dense.separation(reference).to(u.mas)
    assert np.all(seps < 0.1 * u.mas), f"worst {np.max(seps)}"

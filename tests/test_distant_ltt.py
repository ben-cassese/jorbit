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
    ias15_evolve_with_dense_output,
    initialize_ias15_integrator_state,
    interpolate_from_dense_output,
    ltt_seed_floor,
    make_ltt_propagator,
    precompute_interpolation_indices,
    stitched_dense_buffers,
)
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
    monkeypatch.setattr(
        particle_dense, "apply_ltt_seed_floor", lambda state, *args: state
    )
    p = _circular_particle(127.0)
    eph_forced = p.ephemeris(SHORT_ARC, "kitt peak", interpolate=False)
    eph_dense = p.ephemeris(SHORT_ARC, "kitt peak", interpolate=True)
    seps = eph_dense.separation(eph_forced).to(u.mas)
    assert np.all(seps < 0.1 * u.mas)


def test_uncovered_retarded_time_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A retarded time outside the dense buffer is reported, not silently clamped.

    Zeroing the backward padding leaves the 127 AU tracklet's emission times 0.73 days
    before the start of the integration, where there is no step to interpolate. The old
    code extrapolated there without complaint; that silence is what made this expensive
    to find in the first place.
    """
    monkeypatch.setattr(particle_dense, "ltt_seed_floor", lambda *args: 0.0)
    p = _circular_particle(127.0)
    with pytest.raises(RuntimeError, match="outside the integrated span"):
        p.ephemeris(SHORT_ARC, "kitt peak", interpolate=True)


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
    ra_jit, dec_jit = particle_dense._ephem_ias15(
        times_off,
        state,
        p.gravity,
        integrator_state,
        obs_pos,
        relevant_inds,
        p._step_scheduler,
    )
    fwd_mask = times_off >= 0.0
    ra_bounded, dec_bounded, reached = system_dense._ephem_ias15_bounded(
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

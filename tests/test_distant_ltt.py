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
from jorbit import Observations, Particle, System
from jorbit.data.constants import SPEED_OF_LIGHT
from jorbit.integrators import (
    DenseOutput,
    ias15_evolve_with_dense_output,
    initialize_ias15_integrator_state,
    interpolate_from_dense_output,
    make_ltt_propagator,
    precompute_interpolation_indices,
)
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

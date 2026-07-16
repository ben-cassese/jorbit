"""Regression tests for dense-output LTT extrapolation on distant, short-arc objects.

Before the LTT seed floor (see :func:`jorbit.integrators.ias15.interpolation.ltt_seed_floor`),
a short observation arc left every IAS15 dense step far shorter than the light travel
time of a distant object, so the dense-LTT ``on_sky`` path extrapolated the step
polynomial by hundreds of step lengths: ~arcsec-level, state-discontinuous RA/Dec
errors (e.g. 3 arcsec at 127 AU for the scenario below).
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


def test_extrapolation_warning_fires_without_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The host-side catch warns when steps stay shorter than the light travel time."""
    # Disable the seed floor to recreate the old arc-limited step schedule.
    monkeypatch.setattr(
        particle_dense, "apply_ltt_seed_floor", lambda state, *args: state
    )
    p = _circular_particle(127.0)
    with pytest.warns(UserWarning, match="light travel time"):
        p.ephemeris(SHORT_ARC, "kitt peak", interpolate=True)

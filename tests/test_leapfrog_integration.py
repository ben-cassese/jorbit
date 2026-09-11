"""Regression tests for the Yoshida leapfrog integrators as reached through Particle/System.

Covers three bugs reported from downstream use (2026-09-10):

1. ``leapfrog_evolve`` discarded the sign of ``dt``, so every step was taken forward
   regardless of the direction of travel. Any integration to a time before the state's
   epoch was garbage, and because the state then marched away from the requested times,
   the error *grew* as the step size shrank. The two step sizes in
   :func:`test_leapfrog_matches_ias15` are therefore load-bearing: a single step size
   cannot distinguish this from ordinary truncation error.
2. ``jorbit.system.ephem._ephem`` scanned ``on_sky`` over the pre-expanded time array
   while the positions had already been gathered down to the requested times, so
   ``System.ephemeris`` raised for any ``max_step_size`` small enough to need
   intermediate steps.
3. ``System`` had no observation-scoring callables on the leapfrog path.

All of it uses synthetic states and explicit observer positions, so nothing here queries
Horizons.
"""

import jax

jax.config.update("jax_enable_x64", True)

import astropy.units as u
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time

from jorbit import Observations, Particle, System

_X0 = jnp.array([-2.003779703686627, 1.780533558134481, 0.5203350526739642])
_V0 = jnp.array([-0.006668390915419885, -0.006621147093559814, -0.002036640485149475])
_T0 = Time("2025-01-01")

# A fixed, Earth-like barycentric observer. Only has to be the same when the synthetic
# astrometry is generated and when it is scored.
_OBSERVER = jnp.array([-0.17, 0.89, 0.39])

_KW = {
    "gravity": "newtonian solar system",
    "earliest_time": Time("2024-06-01"),
    "latest_time": Time("2025-06-01"),
}

# Building a Particle/System loads an ephemeris, so reuse them across the parametrized
# cases rather than rebuilding per case.
_CACHE: dict = {}


def _objects(integrator: str, max_step_size: float | None) -> tuple[Particle, System]:
    """A (Particle, System) pair on the same state, memoized per integrator setting."""
    key = (integrator, max_step_size)
    if key not in _CACHE:
        step = None if max_step_size is None else max_step_size * u.day
        particle = Particle(
            x=_X0, v=_V0, time=_T0, integrator=integrator, max_step_size=step, **_KW
        )
        system = System(
            particles=[particle], integrator=integrator, max_step_size=step, **_KW
        )
        _CACHE[key] = (particle, system)
    return _CACHE[key]


_ARCS = {
    "forward": jnp.array([0.5, 2.0, 5.0, 8.0]),
    "backward": jnp.array([-0.5, -1.0, -1.5, -1.9]),
    # The epoch sits inside the arc: one backward leg, then one forward leg. This is the
    # shape every downstream caller hits when the epoch is derived from the obs times.
    "straddling": jnp.array([-1.975, -1.0, 1.0, 4.0, 8.0]),
}


@pytest.mark.parametrize("arc", list(_ARCS))
@pytest.mark.parametrize("max_step_size", [1.0, 0.1])
def test_leapfrog_matches_ias15(arc: str, max_step_size: float) -> None:
    """Y8 agrees with IAS15 on forward, backward, and epoch-straddling arcs."""
    offsets = _ARCS[arc]
    times = _T0 + np.asarray(offsets) * u.day

    ref_particle, ref_system = _objects("ias15", None)
    lf_particle, lf_system = _objects("Y8", max_step_size)

    p_ref, v_ref = ref_particle.integrate(times)
    p_lf, v_lf = lf_particle.integrate(times)
    assert jnp.max(jnp.abs(p_lf - p_ref)) < 1e-10
    assert jnp.max(jnp.abs(v_lf - v_ref)) < 1e-10

    p_ref, v_ref = ref_system.integrate(times)
    p_lf, v_lf = lf_system.integrate(times)
    assert jnp.max(jnp.abs(p_lf - p_ref)) < 1e-10
    assert jnp.max(jnp.abs(v_lf - v_ref)) < 1e-10


def test_leapfrog_round_trip_is_reversible() -> None:
    """Integrating out to +10 d and back to the epoch returns the initial state.

    The Yoshida coefficient vectors are palindromic, so the negative step is the exact
    inverse of the positive one; this holds to machine precision, not just to the
    integrator's truncation error.
    """
    particle, _ = _objects("Y8", 0.5)
    times = _T0 + np.array([10.0, 0.0]) * u.day
    x, v = particle.integrate(times)

    assert jnp.max(jnp.abs(x[1] - _X0)) < 1e-12
    assert jnp.max(jnp.abs(v[1] - _V0)) < 1e-12


def test_system_ephemeris_leapfrog_with_expanded_steps() -> None:
    """System.ephemeris works when max_step_size forces intermediate steps.

    Previously raised ``ValueError: scan got values with different leading axis sizes``
    for any step size below the observation spacing.
    """
    times = _T0 + np.array([-5.0, -1.0, 1.0, 5.0, 15.0]) * u.day
    observer = jnp.tile(_OBSERVER, (len(times), 1))

    _, ref_system = _objects("ias15", None)
    _, lf_system = _objects("Y8", 0.5)

    coords_ref = ref_system.ephemeris(times, observer=observer)
    coords_lf = lf_system.ephemeris(times, observer=observer)

    seps = coords_lf.separation(coords_ref).to(u.mas)
    assert np.all(np.isfinite(np.asarray(seps)))
    assert np.max(np.asarray(seps)) < 1.0


def _observations() -> tuple[Observations, SkyCoord, Time]:
    """Synthetic astrometry straddling the epoch, from the trusted IAS15 path."""
    obs_times = _T0 + np.array([-8.0, -3.0, -1.0, 2.0, 6.0, 11.0]) * u.day
    observer = jnp.tile(_OBSERVER, (len(obs_times), 1))
    ref_particle, _ = _objects("ias15", None)
    truth = ref_particle.ephemeris(obs_times, observer=observer)
    obs = Observations(
        observed_coordinates=truth,
        times=obs_times,
        observatories=observer,
        astrometric_uncertainties=0.1 * u.arcsec,
    )
    return obs, truth, obs_times


def test_system_leapfrog_forward_model() -> None:
    """A leapfrog System exposes loglike/residuals/chi2/model_radec and they agree."""
    obs, truth, obs_times = _observations()
    step = 0.5 * u.day
    particle = Particle(
        x=_X0,
        v=_V0,
        time=_T0,
        integrator="Y8",
        max_step_size=step,
        observations=obs,
        **_KW,
    )
    system = System(
        particles=[particle],
        integrator="Y8",
        max_step_size=step,
        observations=obs,
        **_KW,
    )

    assert system.loglike is not None
    assert system.residuals is not None
    assert system.chi2 is not None
    assert system.model_radec is not None

    states = jnp.concatenate([_X0, _V0])[None, :]

    # 1. The modelled sky positions reproduce the (IAS15-generated) truth.
    ras, decs = system.model_radec(states)
    model_sc = SkyCoord(
        ra=np.asarray(ras[0]), dec=np.asarray(decs[0]), unit=u.rad, frame="icrs"
    )
    assert np.max(np.asarray(model_sc.separation(truth).to(u.mas))) < 1.0

    # 2. Residuals are sub-mas and correctly shaped.
    resid = system.residuals(states)
    assert resid.shape == (1, len(obs_times), 2)
    assert np.max(np.abs(np.asarray(resid))) < 1e-3  # arcsec

    # 3. loglike/chi2 match the independently-implemented single-particle path.
    assert jnp.allclose(
        system.loglike(states)[0], particle.loglike(particle.cartesian_state)
    )
    chi2_direct = jnp.einsum("bi,bij,bj->", resid[0], obs.inv_cov_matrices, resid[0])
    assert jnp.allclose(system.chi2(states)[0], chi2_direct)

    # 4. The callables batch over the particle axis (the point of the forward model):
    # the same schedule is shared, but each row gets its own trajectory and LTT.
    offset = jnp.array([1e-4, 0.0, 0.0, 0.0, 0.0, 0.0])
    batch = jnp.stack([states[0], states[0] + offset])
    ll = system.loglike(batch)
    assert ll.shape == (2,)
    assert jnp.allclose(ll[0], system.loglike(states)[0])
    assert ll[1] < ll[0]
    assert system.residuals(batch).shape == (2, len(obs_times), 2)


def test_leapfrog_supports_reverse_mode_ad() -> None:
    """jax.grad works natively on the leapfrog paths, unlike the IAS15 while_loop.

    Checked away from the likelihood maximum: at the optimum the gradient is ~0, so a
    relative comparison there measures cancellation rather than agreement.
    """
    obs, _truth, _obs_times = _observations()
    step = 0.5 * u.day
    particle = Particle(
        x=_X0,
        v=_V0,
        time=_T0,
        integrator="Y8",
        max_step_size=step,
        observations=obs,
        **_KW,
    )
    system = System(
        particles=[particle],
        integrator="Y8",
        max_step_size=step,
        observations=obs,
        **_KW,
    )

    # Off the optimum, so the gradient is comfortably nonzero.
    vec = jnp.concatenate([_X0, _V0]) + jnp.array([1e-6, 0.0, 0.0, 0.0, 0.0, 0.0])

    def f(v: jnp.ndarray) -> jnp.ndarray:
        return system.loglike(v[None, :])[0]

    g_rev = jax.grad(f)(vec)
    g_fwd = jax.jacfwd(f)(vec)
    assert jnp.max(jnp.abs(g_rev - g_fwd)) / jnp.max(jnp.abs(g_fwd)) < 1e-8

    # Particle.loglike uses the plain reverse-mode path for leapfrog rather than the
    # jacfwd-backed custom VJP that the IAS15 while_loop requires. Checking the type
    # pins the optimization: a custom-VJP-wrapped loglike would still return the right
    # gradient, just at ~6x the cost.
    assert isinstance(particle.loglike, jax.tree_util.Partial)
    state = particle.cartesian_state.replace(
        x=vec[None, :3].copy(), v=vec[None, 3:].copy()
    )
    g_particle = jax.grad(particle.loglike)(state)
    g_particle_vec = jnp.concatenate([g_particle.x.flatten(), g_particle.v.flatten()])
    assert jnp.max(jnp.abs(g_particle_vec - g_fwd)) / jnp.max(jnp.abs(g_fwd)) < 1e-8


def test_leapfrog_rejects_nonpositive_step_size() -> None:
    """max_step_size <= 0 fails at construction, not with an int overflow later."""
    for bad in (0 * u.day, -1 * u.day):
        with pytest.raises(AssertionError, match="must be positive"):
            Particle(x=_X0, v=_V0, time=_T0, integrator="Y8", max_step_size=bad, **_KW)
        with pytest.raises(AssertionError, match="must be positive"):
            System(
                particles=[_objects("ias15", None)[0]],
                integrator="Y8",
                max_step_size=bad,
                **_KW,
            )


def test_leapfrog_return_steps_is_consistent() -> None:
    """Particle and System report the same leapfrog step count (not None)."""
    times = _T0 + np.array([-2.0, 3.0, 7.0]) * u.day
    particle, system = _objects("Y8", 0.5)

    _x, _v, p_steps = particle.integrate(times, return_steps=True)
    _x, _v, s_steps = system.integrate(times, return_steps=True)
    assert p_steps is not None and s_steps is not None
    assert int(p_steps) == int(s_steps)

    observer = jnp.tile(_OBSERVER, (len(times), 1))
    _coords, e_steps = system.ephemeris(times, observer=observer, return_steps=True)
    assert int(e_steps) == int(s_steps)


def test_leapfrog_ephemeris_ignores_interpolate() -> None:
    """Interpolate is a documented no-op on the leapfrog path (no dense output)."""
    times = _T0 + np.array([-2.0, 3.0, 7.0]) * u.day
    observer = jnp.tile(_OBSERVER, (len(times), 1))
    particle, _ = _objects("Y8", 0.5)

    a = particle.ephemeris(times, observer=observer, interpolate=True)
    b = particle.ephemeris(times, observer=observer, interpolate=False)
    assert np.array_equal(np.asarray(a.ra.deg), np.asarray(b.ra.deg))
    assert np.array_equal(np.asarray(a.dec.deg), np.asarray(b.dec.deg))

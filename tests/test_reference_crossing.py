"""Regression tests for the reference-crossing time-array bug (JORBIT_BUG.md).

When a time array passed to integrate/ephemeris contains times both *before*
and *after* the reference epoch, the backward entries must be just as accurate
as the forward ones.  The original bug silently returned wrong positions (off
by ~0.1-0.25 AU) or NaN for the backward entries because the integrator
accumulated state across the direction change instead of treating each
direction independently.

Strategy
--------
For each method under test we compare an **array call** (all times in one
invocation) against **independent scalar calls** (one fresh object per time).
If the array call produces results that differ from the scalar calls by more
than a tight tolerance, the reference-crossing logic is broken.  We also
verify that no NaN values appear.

All tests use a synthetic state vector and fake observer positions so they run
fully offline (no Horizons queries).
"""

import jax

jax.config.update("jax_enable_x64", True)

import astropy.units as u
import jax.numpy as jnp
import numpy as np
from astropy.time import Time

from jorbit import Particle
from jorbit.data.constants import SPEED_OF_LIGHT
from jorbit.system import System
from jorbit.utils.states import SystemState

# ---------------------------------------------------------------------------
# Shared test data — a simple bound heliocentric orbit at ~2 AU
# ---------------------------------------------------------------------------

_X0 = jnp.array([-2.003779703686627, 1.780533558134481, 0.5203350526739642])
_V0 = jnp.array([-0.006668390915419885, -0.006621147093559814, -0.002036640485149475])
_T0 = Time("2025-01-01")
# Offsets that cross the reference epoch: two before, two after.
_DTS_DAYS = np.array([-20.0, -10.0, 10.0, 20.0])
# Position tolerance in AU — the scalar-vs-array difference should be zero
# for a correct implementation; 1e-10 AU ≈ 15 mm leaves generous headroom.
_POS_ATOL_AU = 1e-10
# On-sky tolerance for ephemeris comparisons.
_SKY_ATOL_MAS = 0.1  # 0.1 milliarcsecond


def _make_particle(**kwargs) -> Particle:  # noqa: ANN003
    defaults = {"x": _X0, "v": _V0, "time": _T0, "gravity": "newtonian planets"}
    defaults.update(kwargs)
    return Particle(**defaults)


def _make_system(**kwargs) -> System:  # noqa: ANN003
    """Build a System with one tracer and no massive bodies."""
    empty3 = jnp.empty((0, 3))
    t_ref = _T0.tdb.jd
    st = SystemState(
        tracer_positions=jnp.asarray(_X0).reshape(1, 3),
        tracer_velocities=jnp.asarray(_V0).reshape(1, 3),
        massive_positions=empty3,
        massive_velocities=empty3,
        log_gms=jnp.empty((0,)),
        time_reference=jnp.asarray(float(t_ref)),
        relative_time=jnp.asarray(0.0),
        fixed_perturber_positions=empty3,
        fixed_perturber_velocities=empty3,
        fixed_perturber_log_gms=jnp.empty((0,)),
        acceleration_func_kwargs={"c2": SPEED_OF_LIGHT**2},
    )
    defaults = {"state": st, "gravity": "newtonian planets"}
    defaults.update(kwargs)
    return System(**defaults)


# ===================================================================
# Particle.integrate — forced-landing path
# ===================================================================


class TestParticleIntegrateCrossing:
    """Particle.integrate with times crossing the reference epoch."""

    def test_array_matches_scalar(self) -> None:
        """Array call must match independent scalar calls for crossing times."""
        times_arr = _T0 + _DTS_DAYS * u.day

        p = _make_particle()
        pos_arr, _vel_arr = p.integrate(times_arr)
        pos_arr = np.asarray(pos_arr)

        pos_scalar = np.empty((len(_DTS_DAYS), 3))
        for i, dt in enumerate(_DTS_DAYS):
            p_s = _make_particle()
            t = _T0 + dt * u.day
            pos, _ = p_s.integrate(t)
            pos_scalar[i] = np.asarray(pos).reshape(3)

        diffs = np.linalg.norm(pos_arr - pos_scalar, axis=1)
        assert np.all(
            np.isfinite(pos_arr)
        ), f"Particle.integrate array call produced NaN at dt={_DTS_DAYS[np.isnan(pos_arr).any(axis=1)]}"
        assert np.all(
            diffs < _POS_ATOL_AU
        ), f"Particle.integrate array-vs-scalar mismatch: {diffs} AU"

    def test_backward_only(self) -> None:
        """Pure backward array must also work (no forward times at all)."""
        dts = np.array([-30.0, -15.0, -5.0])
        times_arr = _T0 + dts * u.day

        p = _make_particle()
        pos_arr, _ = p.integrate(times_arr)
        pos_arr = np.asarray(pos_arr)

        for i, dt in enumerate(dts):
            p_s = _make_particle()
            pos_s, _ = p_s.integrate(_T0 + dt * u.day)
            diff = np.linalg.norm(pos_arr[i] - np.asarray(pos_s).reshape(3))
            assert diff < _POS_ATOL_AU

    def test_unsorted_crossing(self) -> None:
        """Unsorted time array crossing the reference must still be correct."""
        dts = np.array([10.0, -20.0, 20.0, -10.0])
        times_arr = _T0 + dts * u.day

        p = _make_particle()
        pos_arr, _ = p.integrate(times_arr)
        pos_arr = np.asarray(pos_arr)

        for i, dt in enumerate(dts):
            p_s = _make_particle()
            pos_s, _ = p_s.integrate(_T0 + dt * u.day)
            diff = np.linalg.norm(pos_arr[i] - np.asarray(pos_s).reshape(3))
            assert diff < _POS_ATOL_AU


# ===================================================================
# Particle.integrate_or_interpolate — dense-output (interpolation) path
# ===================================================================


class TestParticleInterpolateCrossing:
    """Particle.integrate_or_interpolate with reference-crossing times."""

    def test_array_matches_scalar(self) -> None:
        """Array call must match independent scalar calls for crossing times."""
        times_arr = _T0 + _DTS_DAYS * u.day

        p = _make_particle()
        pos_arr, _ = p.integrate_or_interpolate(times_arr)
        pos_arr = np.asarray(pos_arr)

        pos_scalar = np.empty((len(_DTS_DAYS), 3))
        for i, dt in enumerate(_DTS_DAYS):
            p_s = _make_particle()
            pos, _ = p_s.integrate_or_interpolate(_T0 + dt * u.day)
            pos_scalar[i] = np.asarray(pos).reshape(3)

        diffs = np.linalg.norm(pos_arr - pos_scalar, axis=1)
        assert np.all(
            np.isfinite(pos_arr)
        ), "Particle.integrate_or_interpolate array call produced NaN"
        assert np.all(
            diffs < _POS_ATOL_AU
        ), f"Particle.integrate_or_interpolate array-vs-scalar mismatch: {diffs} AU"

    def test_backward_only(self) -> None:
        """Pure backward array via interpolation."""
        dts = np.array([-30.0, -15.0, -5.0])
        times_arr = _T0 + dts * u.day

        p = _make_particle()
        pos_arr, _ = p.integrate_or_interpolate(times_arr)
        pos_arr = np.asarray(pos_arr)

        for i, dt in enumerate(dts):
            p_s = _make_particle()
            pos_s, _ = p_s.integrate_or_interpolate(_T0 + dt * u.day)
            diff = np.linalg.norm(pos_arr[i] - np.asarray(pos_s).reshape(3))
            assert diff < _POS_ATOL_AU


# ===================================================================
# Particle.ephemeris
# ===================================================================


class TestParticleEphemerisCrossing:
    """Particle.ephemeris with reference-crossing times."""

    def test_array_matches_scalar(self) -> None:
        """Ephemeris array call must match independent scalar calls."""
        times_arr = _T0 + _DTS_DAYS * u.day
        n = len(_DTS_DAYS)
        # Use barycentric-origin observer (zeros) to stay offline.
        obs_arr = jnp.zeros((n, 3))

        p = _make_particle()
        eph_arr = p.ephemeris(times_arr, observer=obs_arr)

        for i, dt in enumerate(_DTS_DAYS):
            p_s = _make_particle()
            t = _T0 + dt * u.day
            obs_s = jnp.zeros((1, 3))
            eph_s = p_s.ephemeris(t, observer=obs_s)
            sep_mas = float(eph_arr[i].separation(eph_s[0]).to(u.mas).value)
            assert (
                sep_mas < _SKY_ATOL_MAS
            ), f"Particle.ephemeris array-vs-scalar at dt={dt}d: {sep_mas:.4f} mas"

    def test_no_nan_in_backward_entries(self) -> None:
        """Backward ephemeris entries must be finite (no NaN RA/Dec)."""
        dts = np.array([-20.0, -10.0, 10.0, 20.0])
        times_arr = _T0 + dts * u.day
        obs = jnp.zeros((len(dts), 3))

        p = _make_particle()
        eph = p.ephemeris(times_arr, observer=obs)
        assert np.all(np.isfinite(eph.ra.deg)), "NaN in RA"
        assert np.all(np.isfinite(eph.dec.deg)), "NaN in Dec"

    def test_ephemeris_forced_landing_mode(self) -> None:
        """Ephemeris with interpolate=False (forced landing) also handles crossing."""
        times_arr = _T0 + _DTS_DAYS * u.day
        n = len(_DTS_DAYS)
        obs_arr = jnp.zeros((n, 3))

        p = _make_particle()
        eph_arr = p.ephemeris(times_arr, observer=obs_arr, interpolate=False)

        for i, dt in enumerate(_DTS_DAYS):
            p_s = _make_particle()
            t = _T0 + dt * u.day
            obs_s = jnp.zeros((1, 3))
            eph_s = p_s.ephemeris(t, observer=obs_s, interpolate=False)
            sep_mas = float(eph_arr[i].separation(eph_s[0]).to(u.mas).value)
            assert (
                sep_mas < _SKY_ATOL_MAS
            ), f"Particle.ephemeris(interpolate=False) at dt={dt}d: {sep_mas:.4f} mas"


# ===================================================================
# System.integrate
# ===================================================================


class TestSystemIntegrateCrossing:
    """System.integrate with reference-crossing times."""

    def test_array_matches_scalar(self) -> None:
        """System.integrate array call must match independent scalar calls."""
        times_arr = _T0 + _DTS_DAYS * u.day

        sys = _make_system()
        pos_arr, _ = sys.integrate(times_arr)
        pos_arr = np.asarray(pos_arr).reshape(len(_DTS_DAYS), 3)

        for i, dt in enumerate(_DTS_DAYS):
            sys_s = _make_system()
            t = _T0 + dt * u.day
            pos_s, _ = sys_s.integrate(t)
            pos_s = np.asarray(pos_s).reshape(3)
            diff = np.linalg.norm(pos_arr[i] - pos_s)
            assert np.isfinite(diff), f"System.integrate NaN at dt={dt}"
            assert (
                diff < _POS_ATOL_AU
            ), f"System.integrate array-vs-scalar at dt={dt}d: {diff} AU"

    def test_backward_only(self) -> None:
        """Pure backward System.integrate."""
        dts = np.array([-30.0, -15.0, -5.0])
        times_arr = _T0 + dts * u.day

        sys = _make_system()
        pos_arr, _ = sys.integrate(times_arr)
        pos_arr = np.asarray(pos_arr).reshape(len(dts), 3)

        for i, dt in enumerate(dts):
            sys_s = _make_system()
            pos_s, _ = sys_s.integrate(_T0 + dt * u.day)
            pos_s = np.asarray(pos_s).reshape(3)
            diff = np.linalg.norm(pos_arr[i] - pos_s)
            assert diff < _POS_ATOL_AU

    def test_unsorted_crossing(self) -> None:
        """Unsorted crossing times through System.integrate."""
        dts = np.array([10.0, -20.0, 20.0, -10.0])
        times_arr = _T0 + dts * u.day

        sys = _make_system()
        pos_arr, _ = sys.integrate(times_arr)
        pos_arr = np.asarray(pos_arr).reshape(len(dts), 3)

        for i, dt in enumerate(dts):
            sys_s = _make_system()
            pos_s, _ = sys_s.integrate(_T0 + dt * u.day)
            pos_s = np.asarray(pos_s).reshape(3)
            diff = np.linalg.norm(pos_arr[i] - pos_s)
            assert diff < _POS_ATOL_AU


# ===================================================================
# System.ephemeris
# ===================================================================


class TestSystemEphemerisCrossing:
    """System.ephemeris with reference-crossing times."""

    def test_array_matches_scalar(self) -> None:
        """System.ephemeris array call must match independent scalar calls."""
        times_arr = _T0 + _DTS_DAYS * u.day
        n = len(_DTS_DAYS)
        obs_arr = jnp.zeros((n, 3))

        sys = _make_system()
        eph_arr = sys.ephemeris(times_arr, observer=obs_arr)

        for i, dt in enumerate(_DTS_DAYS):
            sys_s = _make_system()
            t = _T0 + dt * u.day
            obs_s = jnp.zeros((1, 3))
            eph_s = sys_s.ephemeris(t, observer=obs_s)
            # System ephemeris shape is (n_particles, n_times); we have 1 particle.
            sep_mas = float(eph_arr[0, i].separation(eph_s[0, 0]).to(u.mas).value)
            assert (
                sep_mas < _SKY_ATOL_MAS
            ), f"System.ephemeris array-vs-scalar at dt={dt}d: {sep_mas:.4f} mas"

    def test_no_nan_in_backward_entries(self) -> None:
        """Backward System.ephemeris entries must be finite."""
        dts = np.array([-20.0, -10.0, 10.0, 20.0])
        times_arr = _T0 + dts * u.day
        obs = jnp.zeros((len(dts), 3))

        sys = _make_system()
        eph = sys.ephemeris(times_arr, observer=obs)
        assert np.all(np.isfinite(eph.ra.deg)), "NaN in System ephemeris RA"
        assert np.all(np.isfinite(eph.dec.deg)), "NaN in System ephemeris Dec"


# ===================================================================
# Cross-method consistency with crossing times
# ===================================================================


class TestCrossMethodConsistency:
    """Different integration paths must agree when times cross the reference."""

    def test_particle_integrate_vs_interpolate(self) -> None:
        """Particle.integrate and integrate_or_interpolate must agree on crossing times."""
        times_arr = _T0 + _DTS_DAYS * u.day

        p = _make_particle()
        pos_int, _ = p.integrate(times_arr)
        pos_interp, _ = p.integrate_or_interpolate(times_arr)

        diff = float(jnp.max(jnp.abs(pos_int - pos_interp)))
        assert (
            diff < 1e-9
        ), f"Particle integrate vs interpolate disagree by {diff} AU on crossing times"

    def test_particle_vs_system(self) -> None:
        """Particle and System with the same state must agree on crossing times."""
        times_arr = _T0 + _DTS_DAYS * u.day

        p = _make_particle()
        pos_p, _ = p.integrate(times_arr)

        sys = _make_system()
        pos_s, _ = sys.integrate(times_arr)
        pos_s = np.asarray(pos_s).reshape(len(_DTS_DAYS), 3)

        diff = float(np.max(np.abs(np.asarray(pos_p) - pos_s)))
        assert (
            diff < 1e-6
        ), f"Particle vs System disagree by {diff} AU on crossing times"

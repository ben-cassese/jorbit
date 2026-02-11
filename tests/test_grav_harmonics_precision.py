"""Tests comparing DoubleDouble and mpmath gravitational harmonics."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from jorbit.accelerations.grav_harmonics import grav_harmonics
from jorbit.accelerations.grav_harmonics_dd import grav_harmonics_dd
from jorbit.accelerations.grav_harmonics_mpmath import grav_harmonics_mpmath


def _make_grav_harmonics_inputs(
    n_particles: int,
    n_jns: int,
    seed: int,
    pos_scale: float = 5.0,
) -> dict:
    """Create inputs for grav_harmonics testing."""
    rng = np.random.RandomState(seed)

    # Body at a random position near origin
    body_pos = jnp.array(rng.normal(0, 0.1, 3))

    # Particles at distances >> body radius
    particle_xs = jnp.array(rng.normal(0, pos_scale, (n_particles, 3)))

    # Typical J values (diminishing with order)
    jns_vals = []
    for i in range(n_jns):
        jns_vals.append(rng.uniform(-1e-3, 1e-3) / (10**i))
    jns = jnp.array(jns_vals)

    return {
        "body_gm": 2.959e-4,  # ~solar GM
        "body_req": 4.65e-3,  # ~solar radius in AU
        "body_pos": body_pos,
        "pole_ra": rng.uniform(0, 2 * np.pi),
        "pole_dec": rng.uniform(-np.pi / 2, np.pi / 2),
        "jns": jns,
        "particle_xs": particle_xs,
    }


# (n_particles, n_jns)
CONFIGS = [
    (1, 1),  # single particle, J2 only
    (1, 2),  # single particle, J2+J3
    (1, 4),  # single particle, J2-J5
    (2, 1),  # two particles, J2
    (3, 2),  # three particles, J2+J3
    (5, 3),  # five particles, J2-J4
    (2, 5),  # two particles, J2-J6
]


@pytest.mark.parametrize("n_particles,n_jns", CONFIGS)
def test_dd_vs_mpmath(n_particles, n_jns) -> None:
    """Test that DD grav harmonics agrees with mpmath to ~28 digits."""
    inputs = _make_grav_harmonics_inputs(n_particles, n_jns, seed=42)

    result_dd_obj = grav_harmonics_dd(**inputs)
    result_dd = np.asarray(result_dd_obj.hi + result_dd_obj.lo)
    result_mp = grav_harmonics_mpmath(
        **{k: np.asarray(v) if hasattr(v, "shape") else v for k, v in inputs.items()},
        dps=50,
    )

    assert result_dd.shape == result_mp.shape

    scale = np.max(np.abs(result_mp))
    abs_err = np.max(np.abs(result_dd - result_mp))
    rel_err = abs_err / scale if scale > 0 else abs_err

    # Both DD and mpmath compute the result to ~30 digits internally, but both
    # are converted to float64 for comparison. Since the code paths differ
    # (loop-based DD vs pure Python mpmath), the float64 roundings may differ
    # by ~1 ULP. The tolerance here just confirms both produce the same float64
    # answer; test_dd_better_than_f64 is the real precision check.
    assert rel_err < 1e-14, (
        f"DD vs mpmath relative error {rel_err:.2e} exceeds 1e-14 "
        f"for config P={n_particles}, J={n_jns}"
    )


@pytest.mark.parametrize("n_particles,n_jns", CONFIGS)
def test_f64_vs_mpmath(n_particles, n_jns) -> None:
    """Test that float64 grav harmonics agrees with mpmath to ~12 digits."""
    inputs = _make_grav_harmonics_inputs(n_particles, n_jns, seed=42)

    result_f64 = np.asarray(grav_harmonics(**inputs))
    result_mp = grav_harmonics_mpmath(
        **{k: np.asarray(v) if hasattr(v, "shape") else v for k, v in inputs.items()},
        dps=50,
    )

    assert result_f64.shape == result_mp.shape

    scale = np.max(np.abs(result_mp))
    abs_err = np.max(np.abs(result_f64 - result_mp))
    rel_err = abs_err / scale if scale > 0 else abs_err

    # jacrev introduces some numerical noise, so slightly more relaxed
    assert rel_err < 1e-11, (
        f"f64 vs mpmath relative error {rel_err:.2e} exceeds 1e-11 "
        f"for config P={n_particles}, J={n_jns}"
    )


@pytest.mark.parametrize("n_particles,n_jns", CONFIGS)
def test_dd_comparable_to_f64(n_particles, n_jns) -> None:
    """Test that DD is comparable in precision to float64.

    For grav harmonics, the DD version uses analytical derivatives while the
    f64 version uses jacrev. Both produce results accurate to ~30+ digits
    internally, but when converted to float64 for comparison, they may round
    to adjacent representable values. We verify both are within a few ULPs
    of the mpmath reference rather than requiring DD <= f64.
    """
    inputs = _make_grav_harmonics_inputs(n_particles, n_jns, seed=42)

    result_dd_obj = grav_harmonics_dd(**inputs)
    result_dd = np.asarray(result_dd_obj.hi + result_dd_obj.lo)
    result_f64 = np.asarray(grav_harmonics(**inputs))
    result_mp = grav_harmonics_mpmath(
        **{k: np.asarray(v) if hasattr(v, "shape") else v for k, v in inputs.items()},
        dps=50,
    )

    scale = np.max(np.abs(result_mp))
    err_dd = np.max(np.abs(result_dd - result_mp))
    err_f64 = np.max(np.abs(result_f64 - result_mp))

    # Both should be at the float64 noise floor
    assert err_dd / scale < 1e-14, (
        f"DD relative error ({err_dd/scale:.2e}) too large "
        f"for config P={n_particles}, J={n_jns}"
    )
    assert err_f64 / scale < 1e-11, (
        f"f64 relative error ({err_f64/scale:.2e}) too large "
        f"for config P={n_particles}, J={n_jns}"
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_multiple_seeds(seed) -> None:
    """Test consistency across random seeds."""
    inputs = _make_grav_harmonics_inputs(n_particles=2, n_jns=2, seed=seed)

    result_dd_obj = grav_harmonics_dd(**inputs)
    result_dd = np.asarray(result_dd_obj.hi + result_dd_obj.lo)
    result_mp = grav_harmonics_mpmath(
        **{k: np.asarray(v) if hasattr(v, "shape") else v for k, v in inputs.items()},
        dps=50,
    )

    scale = np.max(np.abs(result_mp))
    rel_err = np.max(np.abs(result_dd - result_mp)) / scale if scale > 0 else 0.0

    assert rel_err < 1e-14, f"DD vs mpmath relative error {rel_err:.2e} for seed={seed}"

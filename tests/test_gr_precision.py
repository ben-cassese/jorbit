"""Tests comparing DoubleDouble and mpmath PPN gravity implementations.

Verifies that ppn_gravity_dd agrees with the arbitrary-precision mpmath
reference implementation across a range of particle configurations.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from jorbit.accelerations.gr import ppn_gravity
from jorbit.accelerations.gr_dd import ppn_gravity_dd
from jorbit.accelerations.gr_mpmath import ppn_gravity_mpmath_from_state
from jorbit.data.constants import SPEED_OF_LIGHT
from jorbit.utils.states import SystemState


def _make_state(
    n_perturber: int,
    n_massive: int,
    n_tracer: int,
    seed: int,
    c2: float = SPEED_OF_LIGHT**2,
    pos_scale: float = 10.0,
    vel_scale: float = 1.0,
    gm_low: float = 0.1,
    gm_high: float = 1.0,
) -> SystemState:
    """Create a SystemState with random positions/velocities for testing."""
    rng = np.random.RandomState(seed)

    def _make_arrays(n, rng) -> tuple:
        if n > 0:
            pos = jnp.array(rng.normal(0, pos_scale, (n, 3)))
            vel = jnp.array(rng.normal(0, vel_scale, (n, 3)))
            gms = jnp.array(rng.uniform(gm_low, gm_high, n))
            return pos, vel, gms
        return jnp.empty((0, 3)), jnp.empty((0, 3)), jnp.empty((0,))

    p_pos, p_vel, p_gms = _make_arrays(n_perturber, rng)
    m_pos, m_vel, m_gms = _make_arrays(n_massive, rng)
    t_pos, t_vel, _ = _make_arrays(n_tracer, rng)

    return SystemState(
        massive_positions=m_pos,
        massive_velocities=m_vel,
        tracer_positions=t_pos,
        tracer_velocities=t_vel,
        log_gms=jnp.log(m_gms) if n_massive > 0 else jnp.empty((0,)),
        time=0.0,
        fixed_perturber_positions=p_pos,
        fixed_perturber_velocities=p_vel,
        fixed_perturber_log_gms=(
            jnp.log(p_gms) if n_perturber > 0 else jnp.empty((0,))
        ),
        acceleration_func_kwargs={"c2": c2},
    )


# (n_perturber, n_massive, n_tracer)
PARTICLE_CONFIGS = [
    (2, 0, 1),  # perturbers + single tracer (common case)
    (3, 0, 2),  # perturbers + multiple tracers
    (0, 2, 0),  # massive only
    (0, 3, 1),  # massive + tracer
    (2, 1, 1),  # all three types
    (3, 2, 2),  # larger mixed
    (0, 4, 0),  # four massive bodies
    (5, 0, 3),  # many perturbers, several tracers
    (2, 2, 2),  # equal mix
]


@pytest.mark.parametrize("n_perturber,n_massive,n_tracer", PARTICLE_CONFIGS)
def test_dd_vs_mpmath(n_perturber, n_massive, n_tracer) -> None:
    """Test that DD PPN gravity agrees with mpmath to ~30 digits."""
    state = _make_state(n_perturber, n_massive, n_tracer, seed=42)

    result_dd_obj = ppn_gravity_dd(state)
    result_dd = np.asarray(result_dd_obj.hi + result_dd_obj.lo)
    result_mp = ppn_gravity_mpmath_from_state(state, dps=50)

    assert (
        result_dd.shape == result_mp.shape
    ), f"Shape mismatch: DD {result_dd.shape} vs mpmath {result_mp.shape}"

    # DD should agree with mpmath to ~30 digits (double-double has ~31 digits).
    # We check relative error where the acceleration is non-negligible,
    # and absolute error otherwise.
    scale = np.max(np.abs(result_mp))
    abs_err = np.max(np.abs(result_dd - result_mp))
    rel_err = abs_err / scale if scale > 0 else abs_err

    # DD precision is ~2^-104 ≈ 5e-32, but accumulated error across the
    # computation can lose a few digits. 1e-28 is a comfortable bound.
    assert rel_err < 1e-28, (
        f"DD vs mpmath relative error {rel_err:.2e} exceeds 1e-28 "
        f"for config P={n_perturber}, M={n_massive}, T={n_tracer}"
    )


@pytest.mark.parametrize("n_perturber,n_massive,n_tracer", PARTICLE_CONFIGS)
def test_f64_vs_mpmath(n_perturber, n_massive, n_tracer) -> None:
    """Test that float64 PPN gravity agrees with mpmath to ~15 digits."""
    state = _make_state(n_perturber, n_massive, n_tracer, seed=42)

    result_f64 = np.asarray(ppn_gravity(state))
    result_mp = ppn_gravity_mpmath_from_state(state, dps=50)

    assert result_f64.shape == result_mp.shape

    scale = np.max(np.abs(result_mp))
    abs_err = np.max(np.abs(result_f64 - result_mp))
    rel_err = abs_err / scale if scale > 0 else abs_err

    # float64 has ~15.9 significant digits; PPN involves many operations
    # so we allow some degradation. 1e-13 is a comfortable bound.
    assert rel_err < 1e-13, (
        f"f64 vs mpmath relative error {rel_err:.2e} exceeds 1e-13 "
        f"for config P={n_perturber}, M={n_massive}, T={n_tracer}"
    )


@pytest.mark.parametrize("n_perturber,n_massive,n_tracer", PARTICLE_CONFIGS)
def test_dd_strictly_better_than_f64(n_perturber, n_massive, n_tracer) -> None:
    """Test that DD is closer to mpmath than float64 for every configuration."""
    state = _make_state(n_perturber, n_massive, n_tracer, seed=42)

    result_dd_obj = ppn_gravity_dd(state)
    result_dd = np.asarray(result_dd_obj.hi + result_dd_obj.lo)
    result_f64 = np.asarray(ppn_gravity(state))
    result_mp = ppn_gravity_mpmath_from_state(state, dps=50)

    err_dd = np.max(np.abs(result_dd - result_mp))
    err_f64 = np.max(np.abs(result_f64 - result_mp))

    assert err_dd <= err_f64, (
        f"DD error ({err_dd:.2e}) not better than f64 ({err_f64:.2e}) "
        f"for config P={n_perturber}, M={n_massive}, T={n_tracer}"
    )


def test_solar_system_like_scales() -> None:
    """Test with solar-system-like position/velocity/GM scales."""
    # Sun-like central body + inner planet + outer planet + asteroid tracer
    state = _make_state(
        n_perturber=1,
        n_massive=2,
        n_tracer=1,
        seed=123,
        pos_scale=5.0,  # ~AU scale
        vel_scale=0.01,  # ~AU/day scale
        gm_low=1e-6,
        gm_high=1e-3,
    )
    # Override perturber to be Sun-like
    state = SystemState(
        massive_positions=state.massive_positions,
        massive_velocities=state.massive_velocities,
        tracer_positions=state.tracer_positions,
        tracer_velocities=state.tracer_velocities,
        log_gms=state.log_gms,
        time=0.0,
        fixed_perturber_positions=jnp.array([[0.0, 0.0, 0.0]]),
        fixed_perturber_velocities=jnp.array([[0.0, 0.0, 0.0]]),
        fixed_perturber_log_gms=jnp.log(jnp.array([2.959e-4])),  # ~GM_sun
        acceleration_func_kwargs={"c2": SPEED_OF_LIGHT**2},
    )

    result_dd_obj = ppn_gravity_dd(state)
    result_dd = np.asarray(result_dd_obj.hi + result_dd_obj.lo)
    result_f64 = np.asarray(ppn_gravity(state))
    result_mp = ppn_gravity_mpmath_from_state(state, dps=50)

    scale = np.max(np.abs(result_mp))
    err_dd = np.max(np.abs(result_dd - result_mp)) / scale
    err_f64 = np.max(np.abs(result_f64 - result_mp)) / scale

    assert err_dd < 1e-28, f"DD relative error {err_dd:.2e} at solar system scales"
    assert err_f64 < 1e-13, f"f64 relative error {err_f64:.2e} at solar system scales"
    assert err_dd < err_f64, "DD should be more precise than f64"


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_multiple_seeds(seed) -> None:
    """Test that results are consistent across different random seeds."""
    state = _make_state(n_perturber=2, n_massive=1, n_tracer=1, seed=seed)

    result_dd_obj = ppn_gravity_dd(state)
    result_dd = np.asarray(result_dd_obj.hi + result_dd_obj.lo)
    result_mp = ppn_gravity_mpmath_from_state(state, dps=50)

    scale = np.max(np.abs(result_mp))
    rel_err = np.max(np.abs(result_dd - result_mp)) / scale if scale > 0 else 0.0

    assert rel_err < 1e-28, f"DD vs mpmath relative error {rel_err:.2e} for seed={seed}"

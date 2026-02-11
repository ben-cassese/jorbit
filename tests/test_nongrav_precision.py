"""Tests comparing DoubleDouble and mpmath nongravitational acceleration."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from jorbit.accelerations.nongrav import nongrav_acceleration
from jorbit.accelerations.nongrav_dd import nongrav_acceleration_dd
from jorbit.accelerations.nongrav_mpmath import nongrav_acceleration_mpmath_from_state
from jorbit.utils.states import SystemState


def _make_nongrav_state(
    n_massive: int,
    n_tracer: int,
    seed: int,
    pos_scale: float = 5.0,
    vel_scale: float = 0.5,
) -> SystemState:
    """Create a SystemState with nongrav coefficients for testing."""
    rng = np.random.RandomState(seed)
    N = n_massive + n_tracer

    m_pos = (
        jnp.array(rng.normal(0, pos_scale, (n_massive, 3)))
        if n_massive > 0
        else jnp.empty((0, 3))
    )
    m_vel = (
        jnp.array(rng.normal(0, vel_scale, (n_massive, 3)))
        if n_massive > 0
        else jnp.empty((0, 3))
    )
    m_gms = (
        jnp.array(rng.uniform(0.1, 1.0, n_massive))
        if n_massive > 0
        else jnp.empty((0,))
    )
    t_pos = (
        jnp.array(rng.normal(0, pos_scale, (n_tracer, 3)))
        if n_tracer > 0
        else jnp.empty((0, 3))
    )
    t_vel = (
        jnp.array(rng.normal(0, vel_scale, (n_tracer, 3)))
        if n_tracer > 0
        else jnp.empty((0, 3))
    )

    # Generate nongrav coefficients: small values typical of asteroids
    a1 = jnp.array(rng.uniform(-1e-8, 1e-8, N))
    a2 = jnp.array(rng.uniform(-1e-8, 1e-8, N))
    a3 = jnp.array(rng.uniform(-1e-8, 1e-8, N))

    return SystemState(
        massive_positions=m_pos,
        massive_velocities=m_vel,
        tracer_positions=t_pos,
        tracer_velocities=t_vel,
        log_gms=jnp.log(m_gms) if n_massive > 0 else jnp.empty((0,)),
        time=0.0,
        fixed_perturber_positions=jnp.empty((0, 3)),
        fixed_perturber_velocities=jnp.empty((0, 3)),
        fixed_perturber_log_gms=jnp.empty((0,)),
        acceleration_func_kwargs={"a1": a1, "a2": a2, "a3": a3},
    )


CONFIGS = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 0),
    (2, 0),
    (1, 1),
    (2, 2),
    (0, 5),
    (3, 3),
]


@pytest.mark.parametrize("n_massive,n_tracer", CONFIGS)
def test_dd_vs_mpmath(n_massive, n_tracer) -> None:
    """Test that DD nongrav agrees with mpmath to ~30 digits."""
    state = _make_nongrav_state(n_massive, n_tracer, seed=42)

    result_dd_obj = nongrav_acceleration_dd(state)
    result_dd = np.asarray(result_dd_obj.hi + result_dd_obj.lo)
    result_mp = nongrav_acceleration_mpmath_from_state(state, dps=50)

    assert result_dd.shape == result_mp.shape

    scale = np.max(np.abs(result_mp))
    abs_err = np.max(np.abs(result_dd - result_mp))
    rel_err = abs_err / scale if scale > 0 else abs_err

    assert rel_err < 1e-28, (
        f"DD vs mpmath relative error {rel_err:.2e} exceeds 1e-28 "
        f"for config M={n_massive}, T={n_tracer}"
    )


@pytest.mark.parametrize("n_massive,n_tracer", CONFIGS)
def test_f64_vs_mpmath(n_massive, n_tracer) -> None:
    """Test that float64 nongrav agrees with mpmath to ~15 digits."""
    state = _make_nongrav_state(n_massive, n_tracer, seed=42)

    result_f64 = np.asarray(nongrav_acceleration(state))
    result_mp = nongrav_acceleration_mpmath_from_state(state, dps=50)

    assert result_f64.shape == result_mp.shape

    scale = np.max(np.abs(result_mp))
    abs_err = np.max(np.abs(result_f64 - result_mp))
    rel_err = abs_err / scale if scale > 0 else abs_err

    assert rel_err < 1e-13, (
        f"f64 vs mpmath relative error {rel_err:.2e} exceeds 1e-13 "
        f"for config M={n_massive}, T={n_tracer}"
    )


@pytest.mark.parametrize("n_massive,n_tracer", CONFIGS)
def test_dd_better_than_f64(n_massive, n_tracer) -> None:
    """Test that DD is at least as precise as float64."""
    state = _make_nongrav_state(n_massive, n_tracer, seed=42)

    result_dd_obj = nongrav_acceleration_dd(state)
    result_dd = np.asarray(result_dd_obj.hi + result_dd_obj.lo)
    result_f64 = np.asarray(nongrav_acceleration(state))
    result_mp = nongrav_acceleration_mpmath_from_state(state, dps=50)

    err_dd = np.max(np.abs(result_dd - result_mp))
    err_f64 = np.max(np.abs(result_f64 - result_mp))

    assert err_dd <= err_f64, (
        f"DD error ({err_dd:.2e}) not better than f64 ({err_f64:.2e}) "
        f"for config M={n_massive}, T={n_tracer}"
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_multiple_seeds(seed) -> None:
    """Test consistency across random seeds."""
    state = _make_nongrav_state(n_massive=1, n_tracer=1, seed=seed)

    result_dd_obj = nongrav_acceleration_dd(state)
    result_dd = np.asarray(result_dd_obj.hi + result_dd_obj.lo)
    result_mp = nongrav_acceleration_mpmath_from_state(state, dps=50)

    scale = np.max(np.abs(result_mp))
    rel_err = np.max(np.abs(result_dd - result_mp)) / scale if scale > 0 else 0.0

    assert rel_err < 1e-28, f"DD vs mpmath relative error {rel_err:.2e} for seed={seed}"

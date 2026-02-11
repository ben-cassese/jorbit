"""Tests comparing DoubleDouble and mpmath Newtonian gravity implementations."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from jorbit.accelerations.newtonian import newtonian_gravity
from jorbit.accelerations.newtonian_dd import newtonian_gravity_dd
from jorbit.accelerations.newtonian_mpmath import newtonian_gravity_mpmath_from_state
from jorbit.utils.states import SystemState


def _make_state(
    n_perturber: int,
    n_massive: int,
    n_tracer: int,
    seed: int,
    pos_scale: float = 10.0,
    vel_scale: float = 1.0,
    gm_low: float = 0.1,
    gm_high: float = 1.0,
) -> SystemState:
    """Create a SystemState with random positions/velocities for testing."""
    rng = np.random.RandomState(seed)

    def _make(n) -> tuple:
        if n > 0:
            pos = jnp.array(rng.normal(0, pos_scale, (n, 3)))
            vel = jnp.array(rng.normal(0, vel_scale, (n, 3)))
            gms = jnp.array(rng.uniform(gm_low, gm_high, n))
            return pos, vel, gms
        return jnp.empty((0, 3)), jnp.empty((0, 3)), jnp.empty((0,))

    p_pos, p_vel, p_gms = _make(n_perturber)
    m_pos, m_vel, m_gms = _make(n_massive)
    t_pos, t_vel, _ = _make(n_tracer)

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
        acceleration_func_kwargs={},
    )


PARTICLE_CONFIGS = [
    (2, 0, 1),
    (3, 0, 2),
    (0, 2, 0),
    (0, 3, 1),
    (2, 1, 1),
    (3, 2, 2),
    (0, 4, 0),
    (5, 0, 3),
    (2, 2, 2),
]


@pytest.mark.parametrize("n_perturber,n_massive,n_tracer", PARTICLE_CONFIGS)
def test_dd_vs_mpmath(n_perturber, n_massive, n_tracer) -> None:
    """Test that DD Newtonian gravity agrees with mpmath to ~30 digits."""
    state = _make_state(n_perturber, n_massive, n_tracer, seed=42)

    result_dd_obj = newtonian_gravity_dd(state)
    result_dd = np.asarray(result_dd_obj.hi + result_dd_obj.lo)
    result_mp = newtonian_gravity_mpmath_from_state(state, dps=50)

    assert result_dd.shape == result_mp.shape

    scale = np.max(np.abs(result_mp))
    abs_err = np.max(np.abs(result_dd - result_mp))
    rel_err = abs_err / scale if scale > 0 else abs_err

    assert rel_err < 1e-28, (
        f"DD vs mpmath relative error {rel_err:.2e} exceeds 1e-28 "
        f"for config P={n_perturber}, M={n_massive}, T={n_tracer}"
    )


@pytest.mark.parametrize("n_perturber,n_massive,n_tracer", PARTICLE_CONFIGS)
def test_f64_vs_mpmath(n_perturber, n_massive, n_tracer) -> None:
    """Test that float64 Newtonian gravity agrees with mpmath to ~15 digits."""
    state = _make_state(n_perturber, n_massive, n_tracer, seed=42)

    result_f64 = np.asarray(newtonian_gravity(state))
    result_mp = newtonian_gravity_mpmath_from_state(state, dps=50)

    assert result_f64.shape == result_mp.shape

    scale = np.max(np.abs(result_mp))
    abs_err = np.max(np.abs(result_f64 - result_mp))
    rel_err = abs_err / scale if scale > 0 else abs_err

    assert rel_err < 1e-13, (
        f"f64 vs mpmath relative error {rel_err:.2e} exceeds 1e-13 "
        f"for config P={n_perturber}, M={n_massive}, T={n_tracer}"
    )


@pytest.mark.parametrize("n_perturber,n_massive,n_tracer", PARTICLE_CONFIGS)
def test_dd_better_than_f64(n_perturber, n_massive, n_tracer) -> None:
    """Test that DD is at least as precise as float64."""
    state = _make_state(n_perturber, n_massive, n_tracer, seed=42)

    result_dd_obj = newtonian_gravity_dd(state)
    result_dd = np.asarray(result_dd_obj.hi + result_dd_obj.lo)
    result_f64 = np.asarray(newtonian_gravity(state))
    result_mp = newtonian_gravity_mpmath_from_state(state, dps=50)

    err_dd = np.max(np.abs(result_dd - result_mp))
    err_f64 = np.max(np.abs(result_f64 - result_mp))

    assert err_dd <= err_f64, (
        f"DD error ({err_dd:.2e}) not better than f64 ({err_f64:.2e}) "
        f"for config P={n_perturber}, M={n_massive}, T={n_tracer}"
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_multiple_seeds(seed) -> None:
    """Test consistency across random seeds."""
    state = _make_state(n_perturber=2, n_massive=1, n_tracer=1, seed=seed)

    result_dd_obj = newtonian_gravity_dd(state)
    result_dd = np.asarray(result_dd_obj.hi + result_dd_obj.lo)
    result_mp = newtonian_gravity_mpmath_from_state(state, dps=50)

    scale = np.max(np.abs(result_mp))
    rel_err = np.max(np.abs(result_dd - result_mp)) / scale if scale > 0 else 0.0

    assert rel_err < 1e-28, f"DD vs mpmath relative error {rel_err:.2e} for seed={seed}"

"""Test whether the leapfrog integrator setup functions work correctly."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.integrators.yoshida_leapfrog import create_leapfrog_times


def test_create_leapfrog_times() -> None:
    """Test that create_leapfrog_times creates expanded time series with dt<bound."""
    times = jnp.array([1.0, 2.0, 5.0, -2, 1])
    expanded_times, inds = create_leapfrog_times(
        t0=0.9, times=times, biggest_allowed_dt=0.3
    )
    assert jnp.allclose(expanded_times[inds], times)
    assert jnp.abs(jnp.diff(expanded_times)).max() <= 0.3 + 1e-14

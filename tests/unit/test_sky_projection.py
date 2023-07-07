import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jorbit.engine import icrs_to_barycentricmeanecliptic
from jorbit.engine import barycentricmeanecliptic_to_icrs


def test_reversible():
    """
    A dummy test to get started with the pytest framework
    """
    x = jnp.array(
        [[0.73291537, -1.48253882, -1.24400574], [0.73291537, -1.48253882, -1.24400574]]
    )
    y = icrs_to_barycentricmeanecliptic(x)
    z = barycentricmeanecliptic_to_icrs(y)
    assert jnp.allclose(x, z)

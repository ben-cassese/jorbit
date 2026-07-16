"""Shared pytest configuration."""

import jax
import pytest


@pytest.fixture(autouse=True, scope="module")
def _clear_jax_caches_between_modules() -> None:
    """Free the JIT compilation caches after each test module.

    Compiled executables live in process-global caches for every distinct input
    shape a jitted function sees. The suite exercises hundreds of shapes (many
    Particles/Systems with different observation counts and step schedules), so
    without clearing, the caches grow monotonically over the whole run — enough
    to OOM a 16 GB CI runner by the final modules. Clearing per module caps the
    peak near the most demanding single module at the cost of some
    recompilation.
    """
    yield
    jax.clear_caches()

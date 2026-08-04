"""Shared pytest configuration."""

import jax
import pytest
from astropy.utils.data import conf as astropy_data_conf

# astropy's 10 s default is a per-read socket timeout, not a per-download budget, so a
# few seconds of stall aborts an otherwise healthy transfer -- and the suite pulls
# ~1.4 GB of ephemerides on a cold cache. 60 s is still short enough that a genuinely
# dead server fails promptly. Set at import so it applies to the downloads that
# `import jorbit` triggers while test modules are being collected.
astropy_data_conf.remote_timeout = 60


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

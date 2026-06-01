"""The Particle subpackage: the high-level single-particle interface.

The user-facing :class:`Particle` lives in :mod:`jorbit.particle.particle`; the jitted
helpers that implement each integration branch are split across the sibling modules
(``ephem``, ``ias15_dense``, ``ias15_forced``, ``keplerian``, ``covariance``,
``likelihood``).
"""

import jax

jax.config.update("jax_enable_x64", True)

from jorbit.particle.particle import Particle

__all__ = ["Particle"]

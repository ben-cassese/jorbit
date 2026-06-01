"""The System subpackage: the high-level multi-particle interface.

The user-facing :class:`System` lives in :mod:`jorbit.system.system`; the jitted
helpers that implement each integration branch are split across the sibling modules
(``ephem``, ``ias15_dense``, ``keplerian``).
"""

import jax

jax.config.update("jax_enable_x64", True)

from jorbit.system.system import System

__all__ = ["System"]

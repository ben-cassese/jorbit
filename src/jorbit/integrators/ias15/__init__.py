"""A JAX implementation of the IAS15 integrator.

This is a pythonized/jaxified version of the IAS15 integrator from Rein & Spiegel (2015)
(DOI: 10.1093/mnras/stu2164), currently implemented in REBOUND. It used to follow the
implementation found in the REBOUND source as closely as possible; see < v1.2 for that.

The original code is available on `github <https://github.com/hannorein/rebound/blob/0b5c85d836fec20bc284d1f1bb326f418e11f591/src/integrator_ias15.c>`_.
Accessed Summer 2023, re-visited Fall 2024. Refactored early 2026.

Many thanks to the REBOUND developers for their work on this integrator, and for making it open source!

The implementation is split across sibling modules: ``helpers`` (low-level primitives),
``step_control`` (adaptive step-size controllers), ``interpolation`` (dense-output /
light-travel-time utilities), ``step`` (the single-step predictor-corrector), and
``evolve`` (the driving loops). The names below are re-exported so that
``jorbit.integrators.ias15.<name>`` continues to resolve as before the split.
"""

# This is a pythonized/jaxified version of the IAS15 integrator from
# Rein & Spiegel (2015) (DOI: 10.1093/mnras/stu2164), currently implemented in REBOUND.
# The original code is available at https://github.com/hannorein/rebound/blob/0b5c85d836fec20bc284d1f1bb326f418e11f591/src/integrator_ias15.c
# Accessed Summer 2023, re-visited Fall 2024. Refactored early 2026.

# Many thanks to the REBOUND developers for their work on this integrator,
# and for making it open source!
import jax

jax.config.update("jax_enable_x64", True)

from jorbit.integrators.ias15.evolve import (
    IAS15_MAX_DYNAMIC_STEPS,
    _ias15_evolve_core,
    ias15_evolve,
    ias15_evolve_forced_landing,
    ias15_evolve_with_dense_output,
)
from jorbit.integrators.ias15.helpers import (
    _estimate_x_v_from_b,
    add_cs,
    initialize_ias15_integrator_state,
)
from jorbit.integrators.ias15.interpolation import (
    interpolate_from_dense_output,
    make_ltt_propagator,
    precompute_interpolation_indices,
)
from jorbit.integrators.ias15.step import (
    _predict_next_step,
    _refine_sub_g,
    _update_bs,
    ias15_step,
)
from jorbit.integrators.ias15.step_control import (
    next_proposed_dt_global,
    next_proposed_dt_PRS23,
)

__all__ = [
    "IAS15_MAX_DYNAMIC_STEPS",
    "_estimate_x_v_from_b",
    "_ias15_evolve_core",
    "_predict_next_step",
    "_refine_sub_g",
    "_update_bs",
    "add_cs",
    "ias15_evolve",
    "ias15_evolve_forced_landing",
    "ias15_evolve_with_dense_output",
    "ias15_step",
    "initialize_ias15_integrator_state",
    "interpolate_from_dense_output",
    "make_ltt_propagator",
    "next_proposed_dt_PRS23",
    "next_proposed_dt_global",
    "precompute_interpolation_indices",
]

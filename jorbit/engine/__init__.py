"""
The "engine" behind most of the functionality.

This module contains the functions that do the heavy lifting of when integrating/fitting
orbits. Everything in this module is written in JAX, and tries as closely as possible
to follow JAX best practices (and avoid `"🔪 the sharp bits 🔪" <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html>`_).
This means that all functions are "pure" (no side effects), all functions can be 
Just-In-Time compiled with jax.jit, and all functions can be differentiated.
Functions should not depend on anything besides their input parameters and pre-imported
constants, which must be jax.numpy arrays or other JAX-compatible types like pytrees.

By design, there are very few guardrails in this module. This is again to
appease JAX's JIT compiler, which does not play well with certain pure Python functions.
The functions here are brittle, and often depend on arrays with specific shapes to run
properly, but will still return something nonsensical if the input arrays are
the wrong shape. Procede with caution, and use the Observation, Particle, and System
classes for more user-friendly interfaces to these functions.

Several other quirks of JAX to keep in mind:

- for loops are possible, but lead to excessively long compilation times.
  jax.lax.scan is a better choice for most cases.
- jax.lax.cond will trace both branches of a conditional, but then only execute one at
  runtime. This is why there are so many throughout these module: often, it'd be
  great to only compute something until a condition is met, but JAX doesn't like
  breaking out of loops early. So, often we loop over a larger number of steps than
  needed, but use lax.cond to only perform an expensive computation if necessary, and
  otherwise run a dummy function which returns the unmodified values.
- BUT if you vmap a function that includes a lax.cond, it *will* execute both branches
  since it internally converts it to a lax.select. This can have serious performance
  implications, in general here so much so that it's worth avoiding vmap, as cool as it is.

"""

import jax
from jax.config import config

config.update("jax_enable_x64", True)

from jorbit.engine.accelerations import *
from jorbit.engine.ephemeris import *
from jorbit.engine.gauss_jackson_likelihood import *
from jorbit.engine.sky_projection import *

j_planet_state_helper = jax.jit(planet_state_helper)
j_planet_state = jax.jit(planet_state)
j_gr_helper = jax.jit(gr_helper)
j_gr = jax.jit(gr)
j_newtonian_helper = jax.jit(newtonian_helper)
j_newtonian = jax.jit(newtonian)
j_acceleration = jax.jit(acceleration)
j_on_sky = jax.jit(on_sky)
j_sky_error = jax.jit(sky_error)
j_barycentricmeanecliptic_to_icrs = jax.jit(barycentricmeanecliptic_to_icrs)
j_icrs_to_barycentricmeanecliptic = jax.jit(icrs_to_barycentricmeanecliptic)
j_cart_to_elements = jax.jit(cart_to_elements)
j_elements_to_cart = jax.jit(elements_to_cart)

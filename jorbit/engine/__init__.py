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
import jax.numpy as jnp

from jorbit.engine.sky_projection import cart_to_elements, elements_to_cart

j_cart_to_elements = jax.jit(cart_to_elements)
j_elements_to_cart = jax.jit(elements_to_cart)


def _pad_to_parallelize(ndevices, arr, pad_value):
    """
    Fold arrays where the first index runs over particles into chunks that
    can be parallelized over multiple devices
    """
    nparticles = arr.shape[0]

    if nparticles < ndevices:
        return arr.reshape((nparticles, 1) + arr.shape[1:])

    padding = arr.shape[0] % ndevices  # can't use divmod, that's not jittable

    padded_arr = jnp.ones((arr.shape[0] + padding,) + arr.shape[1:]) * pad_value
    padded_arr = padded_arr.at[:nparticles].set(arr)

    particles_per_device = (
        padded_arr.shape[0] // ndevices
    )  # can't use divmod, that's not jittable

    return padded_arr.reshape((ndevices, particles_per_device) + arr.shape[1:])


pad_to_parallelize = jax.jit(
    jax.tree_util.Partial(_pad_to_parallelize, jax.local_device_count())
)

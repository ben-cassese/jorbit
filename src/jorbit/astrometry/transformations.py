import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.data.constants import (
    ICRS_TO_HORIZONS_ECLIPTIC_ROT_MAT,
    HORIZONS_ECLIPTIC_TO_ICRS_ROT_MAT,
)


def icrs_to_horizons_ecliptic(xs):
    rotated_xs = jnp.dot(xs, ICRS_TO_HORIZONS_ECLIPTIC_ROT_MAT.T)
    return rotated_xs


def horizons_ecliptic_to_icrs(xs):
    rotated_xs = jnp.dot(xs, HORIZONS_ECLIPTIC_TO_ICRS_ROT_MAT.T)
    return rotated_xs

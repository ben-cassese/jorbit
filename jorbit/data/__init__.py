import pkg_resources
import pickle
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

t = pkg_resources.resource_filename("jorbit", "data/observatory_codes.pkl")
with open(t, "rb") as f:
    observatory_codes = pickle.load(f)

# These were created by running jorbit.construct_perturbers.construct_perturbers()
# Caching since they're used the most often
t = pkg_resources.resource_filename("jorbit", "data/standard_ephemeris_params.pkl")
with open(t, "rb") as f:
    tmp = pickle.load(f)
STANDARD_PLANET_PARAMS = tmp["STANDARD_PLANET_PARAMS"]
STANDARD_ASTEROID_PARAMS = tmp["STANDARD_ASTEROID_PARAMS"]
STANDARD_PLANET_GMS = tmp["STANDARD_PLANET_GMS"]
STANDARD_ASTEROID_GMS = tmp["STANDARD_ASTEROID_GMS"]

# Pulling this one out for converting between Cartesian and Keplerian elements
STANDARD_SUN_PARAMS = [jnp.array([STANDARD_PLANET_PARAMS[i][0]]) for i in range(3)]

import os
from astropy.utils.data import download_files_in_parallel, is_url_in_cache


from jorbit.data.constants import planet_ephemeris, asteroid_ephemeris
from jorbit.particle import Particle
from jorbit.observations import Observations
from jorbit.system import System

os.environ["JAX_ENABLE_X64"] = "True"

_root = os.path.dirname(__file__)
DATADIR = os.path.join(_root, "data/")

if (not is_url_in_cache(planet_ephemeris)) or (not is_url_in_cache(asteroid_ephemeris)):
    print("JPL DE440 ephemeris files not found in astropy cache, downloading now...")
    print(
        "Files are approx. 765 MB, may take several minutes but will not be repeated."
    )
    download_files_in_parallel(
        [planet_ephemeris, asteroid_ephemeris], cache=True, show_progress=True
    )

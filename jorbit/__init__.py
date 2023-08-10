import os
from astropy.utils.data import download_files_in_parallel, is_url_in_cache


from jorbit.data.constants import PLANET_EPHEMERIS_URL, ASTEROID_EPHEMERIS_URL
from jorbit.particle import Particle
from jorbit.observations import Observations
from jorbit.system import System

os.environ["JAX_ENABLE_X64"] = "True"

_root = os.path.dirname(__file__)
DATADIR = os.path.join(_root, "data/")

if (not is_url_in_cache(PLANET_EPHEMERIS_URL)) or (
    not is_url_in_cache(ASTEROID_EPHEMERIS_URL)
):
    print("JPL DE440 ephemeris files not found in astropy cache, downloading now...")
    print(
        "Files are approx. 765 MB, may take several minutes but will not be repeated."
    )
    download_files_in_parallel(
        [PLANET_EPHEMERIS_URL, ASTEROID_EPHEMERIS_URL], cache=True, show_progress=True
    )

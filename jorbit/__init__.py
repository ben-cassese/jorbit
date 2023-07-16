from .data.constants import planet_ephemeris, asteroid_ephemeris
from astropy.utils.data import download_files_in_parallel, is_url_in_cache

if (not is_url_in_cache(planet_ephemeris)) or (not is_url_in_cache(asteroid_ephemeris)):
    print("JPL DE440 ephemeris files not found in astropy cache, downloading now...")
    print(
        "Files are approx. 765 MB, may take several minutes but will not be repeated."
    )
    download_files_in_parallel(
        [planet_ephemeris, asteroid_ephemeris], cache=True, show_progress=True
    )

from .particle import Particle
from .observations import Observations
from .system import System

s = "super long string meant to check if the pre-commit hook is working properly, this should be wrapped by black hopefully"

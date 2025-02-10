import warnings

warnings.filterwarnings("ignore", module="erfa")

from astropy.utils.data import download_files_in_parallel, is_url_in_cache

from jorbit.data.constants import (
    DEFAULT_ASTEROID_EPHEMERIS_URL,
    DEFAULT_PLANET_EPHEMERIS_URL,
    JORBIT_EPHEM_URL_BASE,
)
from jorbit.ephemeris.ephemeris import Ephemeris
from jorbit.observation import Observations
from jorbit.particle import Particle

__all__ = ["Ephemeris", "Observations", "Particle", "System"]


if (not is_url_in_cache(DEFAULT_PLANET_EPHEMERIS_URL)) or (
    not is_url_in_cache(DEFAULT_ASTEROID_EPHEMERIS_URL)
):
    print("JPL DE440 ephemeris files not found in astropy cache, downloading now...")
    print(
        "Files are approx. 765 MB, may take several minutes but will not be repeated."
    )
    download_files_in_parallel(
        [DEFAULT_PLANET_EPHEMERIS_URL, DEFAULT_ASTEROID_EPHEMERIS_URL],
        cache=True,
        show_progress=True,
    )

if not is_url_in_cache(JORBIT_EPHEM_URL_BASE + "names.npy"):
    print("Basic jorbit ephemeris files not found in astropy cache, downloading now...")
    print("Files are approx 660 MB, may take several minutes but will not be repeated.")
    download_files_in_parallel(
        [
            JORBIT_EPHEM_URL_BASE + "names.npy",
            JORBIT_EPHEM_URL_BASE + "x0.npy",
            JORBIT_EPHEM_URL_BASE + "v0.npy",
            JORBIT_EPHEM_URL_BASE + "mpcorb.arrow",
        ],
        cache=True,
        show_progress=True,
    )

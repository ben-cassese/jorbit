"""The jorbit package."""

__url__ = "https://github.com/ben-cassese/jorbit"
__license__ = "MIT"
__description__ = "Solar system orbit fitting and integration with JAX"
__version__ = "0.2.0"

import warnings

warnings.filterwarnings("ignore", module="erfa")

from astropy.utils.data import download_file, is_url_in_cache

from jorbit.data.constants import (
    DEFAULT_ASTEROID_EPHEMERIS_URL,
    DEFAULT_PLANET_EPHEMERIS_URL,
    JORBIT_EPHEM_URL_BASE,
)
from jorbit.ephemeris.ephemeris import Ephemeris
from jorbit.observation import Observations
from jorbit.particle import Particle
from jorbit.system import System

__all__ = ["Ephemeris", "Observations", "Particle", "System"]


def initialize_jorbit() -> None:
    """Download and cache the JPL DE440 and jorbit ephemeris files."""
    if (not is_url_in_cache(DEFAULT_PLANET_EPHEMERIS_URL)) or (
        not is_url_in_cache(DEFAULT_ASTEROID_EPHEMERIS_URL)
    ):
        print(
            "JPL DE440 ephemeris files not found in astropy cache, downloading now..."
        )
        print(
            "Files are approx. 800 MB, may take several minutes but will not be repeated."
        )
        download_file(DEFAULT_PLANET_EPHEMERIS_URL, cache=True, show_progress=True)
        download_file(DEFAULT_ASTEROID_EPHEMERIS_URL, cache=True, show_progress=True)

    if not is_url_in_cache(JORBIT_EPHEM_URL_BASE + "names.npy"):
        print(
            "Basic jorbit ephemeris files not found in astropy cache, downloading now..."
        )
        print(
            "Files are approx 660 MB, may take several minutes but will not be repeated."
        )
        download_file(
            JORBIT_EPHEM_URL_BASE + "names.npy", cache=True, show_progress=True
        )
        download_file(JORBIT_EPHEM_URL_BASE + "x0.npy", cache=True, show_progress=True)
        download_file(JORBIT_EPHEM_URL_BASE + "v0.npy", cache=True, show_progress=True)
        download_file(
            JORBIT_EPHEM_URL_BASE + "mpcorb.arrow", cache=True, show_progress=True
        )


initialize_jorbit()

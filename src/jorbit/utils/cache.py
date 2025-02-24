"""Functions for downloading/clearing jorbit-related files from the Astropy cache."""
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

from astropy.utils.data import (
    clear_download_cache,
    download_file,
    get_cached_urls,
    is_url_in_cache,
)

from jorbit.data.constants import (
    DEFAULT_ASTEROID_EPHEMERIS_URL,
    DEFAULT_PLANET_EPHEMERIS_URL,
    HUGE_ASTEROID_EPHEMERIS_URL,
    JORBIT_EPHEM_CACHE_TIMEOUT,
    JORBIT_EPHEM_URL_BASE,
)


def view_jorbit_cache() -> list:
    """View the jorbit-related files in the Astropy cache."""
    fs = get_cached_urls()
    jorb_fs = [f for f in fs if f.startswith(JORBIT_EPHEM_URL_BASE)]
    if DEFAULT_PLANET_EPHEMERIS_URL in fs:
        jorb_fs.append(DEFAULT_PLANET_EPHEMERIS_URL)
    if DEFAULT_ASTEROID_EPHEMERIS_URL in fs:
        jorb_fs.append(DEFAULT_ASTEROID_EPHEMERIS_URL)
    if HUGE_ASTEROID_EPHEMERIS_URL in fs:
        jorb_fs.append(HUGE_ASTEROID_EPHEMERIS_URL)
    return jorb_fs

def clear_jorbit_cache() -> None:
    """Clear the Astropy cache of jorbit-related files."""
    fs = get_cached_urls()
    fs = [f for f in fs if f.startswith(JORBIT_EPHEM_URL_BASE)]
    fs.append(DEFAULT_PLANET_EPHEMERIS_URL)
    fs.append(DEFAULT_ASTEROID_EPHEMERIS_URL)
    fs.append(HUGE_ASTEROID_EPHEMERIS_URL)
    for f in fs:
        clear_download_cache(f)

def download_file_wrapper(url: str) -> str:
    """Check if a file is in the cache and not expired: if not, download it.

    Args:
        url (str):
            The URL of the file to download.

    Returns:
        str:
            The path to the downloaded file.
    """
    present = is_url_in_cache(url)
    if not present:
        request_file = Path(download_file(url, cache=True))
        current_time = datetime.now(timezone.utc)
        cache_time = datetime.fromtimestamp(request_file.stat().st_mtime, timezone.utc)
        expired = current_time - cache_time > timedelta(
            seconds=JORBIT_EPHEM_CACHE_TIMEOUT
        )
        if expired:
            request_file = download_file(url, cache="update")
            warnings.warn(
                f"File {url} was present in the cache but has expired and will be re-downloaded.",
                stacklevel=2,
            )
        return download_file(url)
    else:
        return download_file(url, cache=True)

"""Test that the packed to unpacked designation translator is consistent."""

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time

from jorbit.mpchecker import (
    animate_region,
    load_mpcorb,
    mpchecker,
    nearest_asteroid,
    nearest_asteroid_helper,
)
from jorbit.utils.mpc import (
    packed_to_unpacked_designation,
    unpacked_to_packed_designation,
)


def test_designation_translators() -> None:
    """Test that the designation translators are consistent."""
    mpcorb = load_mpcorb()
    for n in mpcorb["Packed designation"]:
        q = packed_to_unpacked_designation(n)
        m = unpacked_to_packed_designation(q)
        if n != m:
            print(n, q, m)
            raise ValueError


def test_mpchecker_low_res() -> None:
    """Just check that the mpchecker function runs ok- no comparison to anything yet."""
    _ = mpchecker(
        coordinate=SkyCoord(ra=0 * u.deg, dec=0 * u.deg),
        time=Time("2025-01-01"),
        radius=10 * u.arcmin,
        extra_precision=False,
    )


def test_mpchecker_high_res() -> None:
    """Just check that the mpchecker function runs ok- no comparison to anything yet."""
    _ = mpchecker(
        coordinate=SkyCoord(ra=0 * u.deg, dec=0 * u.deg),
        time=Time("2025-01-01"),
        radius=10 * u.arcmin,
        extra_precision=True,
        observer="Palomar",
    )


def test_nearest_asteroid_low_res() -> None:
    """Check that the nearest_asteroid function runs ok- no comparison to anything yet."""
    _, _ = separations, asteroids = nearest_asteroid(
        coordinate=SkyCoord(ra=0 * u.deg, dec=0 * u.deg),
        times=Time("2025-01-01") + np.arange(0, 3, 1 / 24 / 2) * u.day,
        radius=2 * u.arcmin,
    )


def test_nearest_asteroid_high_res() -> None:
    """Check that the nearest_asteroid function runs ok- no comparison to anything yet."""
    _, _, _, _, _ = nearest_asteroid(
        coordinate=SkyCoord(ra=0 * u.deg, dec=0 * u.deg),
        times=Time("2025-01-01") + np.arange(0, 3, 1 / 24 / 2) * u.day,
        radius=2 * u.arcmin,
        compute_contamination=True,
        observer="kitt peak",
    )


def test_nearest_asteroid_precompute() -> None:
    """Check that the nearest_asteroid_helper function runs ok- no comparison to anything yet."""
    precomputed = nearest_asteroid_helper(
        coordinate=SkyCoord(ra=0 * u.deg, dec=0 * u.deg),
        times=Time("2025-01-01") + np.arange(0, 3, 1 / 24 / 2) * u.day,
        observer="kitt peak",
    )
    separations, asteroids, coord_table, mag_table, total_mags = nearest_asteroid(
        coordinate=SkyCoord(ra=0 * u.deg, dec=0 * u.deg),
        times=Time("2025-01-01") + np.arange(0, 3, 1 / 24 / 2) * u.day,
        radius=2 * u.arcmin,
        compute_contamination=True,
        precomputed=precomputed,
        observer="kitt peak",
    )
    _ = animate_region(
        coordinate=SkyCoord(ra=0 * u.deg, dec=0 * u.deg),
        times=Time("2025-01-01") + np.arange(0, 3, 1 / 24 / 2) * u.day,
        coord_table=coord_table,
        radius=2 * u.arcmin,
    )

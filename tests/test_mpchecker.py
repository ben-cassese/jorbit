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
    unpack_epoch,
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
    _, _ = _separations, _asteroids = nearest_asteroid(
        coordinate=SkyCoord(ra=0 * u.deg, dec=0 * u.deg),
        times=Time("2025-01-01") + np.arange(0, 3, 1) * u.day,
        radius=2 * u.arcmin,
    )


def test_nearest_asteroid_high_res() -> None:
    """Check that the nearest_asteroid function runs ok- no comparison to anything yet."""
    _, _, _, _, _ = nearest_asteroid(
        coordinate=SkyCoord(ra=0 * u.deg, dec=0 * u.deg),
        times=Time("2025-01-01") + np.arange(0, 3, 1) * u.day,
        radius=2 * u.arcmin,
        compute_contamination=True,
        observer="kitt peak",
    )


def test_nearest_asteroid_precompute() -> None:
    """Check that the nearest_asteroid_helper function runs ok- no comparison to anything yet."""
    precomputed = nearest_asteroid_helper(
        coordinate=SkyCoord(ra=0 * u.deg, dec=0 * u.deg),
        times=Time("2025-01-01") + np.arange(0, 3, 1) * u.day,
        observer="kitt peak",
    )
    _separations, _asteroids, coord_table, _mag_table, _total_mags = nearest_asteroid(
        coordinate=SkyCoord(ra=0 * u.deg, dec=0 * u.deg),
        times=Time("2025-01-01") + np.arange(0, 3, 1) * u.day,
        radius=2 * u.arcmin,
        compute_contamination=True,
        precomputed=precomputed,
        observer="kitt peak",
    )
    _ = animate_region(
        coordinate=SkyCoord(ra=0 * u.deg, dec=0 * u.deg),
        times=Time("2025-01-01") + np.arange(0, 3, 1) * u.day,
        coord_table=coord_table,
        radius=2 * u.arcmin,
    )


def test_packed_epoch_translator() -> None:
    """Test that the conversion from packed epoch to astropy Time matches online docs."""
    # examples from https://minorplanetcenter.net/iau/info/PackedDates.html
    assert unpack_epoch("J9611") == Time("1996-01-01", format="iso", scale="tt")
    assert unpack_epoch("J961A") == Time("1996-01-10", format="iso", scale="tt")
    assert unpack_epoch("J969U") == Time("1996-09-30", format="iso", scale="tt")
    assert unpack_epoch("J96A1") == Time("1996-10-01", format="iso", scale="tt")
    assert unpack_epoch("K01AM") == Time("2001-10-22", format="iso", scale="tt")


def test_packed_epoch_is_tt() -> None:
    """MPCORB epochs are defined in TT, not UTC.

    The MPC "Export Format for Minor-Planet Orbits" specifies columns 21-25 as
    "Epoch (in packed form, .0 TT)". Tagging the result UTC (as jorbit did through
    1.5.0) mis-places every decoded epoch by TT-UTC, which is ~69 s in the 2020s --
    a pure systematic worth ~1300 km along-track for a main belt object.
    """
    epoch = unpack_epoch("K259M")
    assert epoch.scale == "tt"

    # A calendar reading interpreted in TT is an *earlier* absolute instant than the
    # same reading interpreted in UTC, by TT-UTC. Before the fix this difference was
    # exactly zero. Bound it loosely rather than pinning 69.184 s, so the test does
    # not rot at the next leap second.
    offset = (Time("2025-09-22", format="iso", scale="utc") - epoch).to(u.s).value
    assert 60.0 < offset < 80.0


def test_packed_epoch_trailing_suffix_is_tt() -> None:
    """A packed epoch with a trailing day suffix also comes back in TT.

    Only the scale is asserted here. unpack_epoch adds ``float(epoch_str[5:])`` days
    to the base date, so "K259M5" resolves to 2025-09-27 rather than 2025-09-22.5 --
    the suffix is treated as whole days despite the ``day_frac`` name. That is a
    separate question from the time scale and is deliberately not pinned down here.
    """
    assert unpack_epoch("K259M5").scale == "tt"

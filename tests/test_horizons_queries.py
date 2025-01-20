import jax

jax.config.update("jax_enable_x64", True)
import astropy.units as u
import jax.numpy as jnp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from jorbit.utils.horizons import (
    horizons_bulk_astrometry_query,
    horizons_bulk_vector_query,
)


def test_horizons_vector_single():
    t0 = Time("2024-12-24T00:00:00", scale="utc")

    jorb_table = horizons_bulk_vector_query("274301", "500@0", t0)
    horizons_obj = Horizons(id="274301", location="@0", epochs=t0.tdb.jd)
    horizons_table = horizons_obj.vectors(refplane="earth")

    # Compare x, y, z columns
    np.testing.assert_allclose(
        jorb_table["x"].values, horizons_table["x"].value, rtol=1e-10
    )
    np.testing.assert_allclose(
        jorb_table["y"].values, horizons_table["y"].value, rtol=1e-10
    )
    np.testing.assert_allclose(
        jorb_table["z"].values, horizons_table["z"].value, rtol=1e-10
    )


def test_horizons_vector_bulk():
    t0 = Time("2024-12-24T00:00:00", scale="utc")
    large_times = Time(
        jnp.linspace(t0.tdb.jd, t0.tdb.jd + 365, 1000), format="jd", scale="tdb"
    )

    # Get last 25 entries from a bulk jorbit query of 1000 pts (too many for astroquery)
    jorb_table = horizons_bulk_vector_query("274301", "500@0", large_times).iloc[-25:]

    # Direct Horizons query for comparison, but only the last 25
    horizons_obj = Horizons(id="274301", location="@0", epochs=large_times.tdb.jd[-25:])
    horizons_table = horizons_obj.vectors(refplane="earth")

    # Compare x, y, z columns
    np.testing.assert_allclose(
        jorb_table["x"].values, horizons_table["x"].value, rtol=1e-10
    )
    np.testing.assert_allclose(
        jorb_table["y"].values, horizons_table["y"].value, rtol=1e-10
    )
    np.testing.assert_allclose(
        jorb_table["z"].values, horizons_table["z"].value, rtol=1e-10
    )


def test_horizons_astrometry_single():
    t0 = Time("2024-12-24T00:00:00", scale="utc")

    horizons_obj = Horizons(id="274301", location="@0", epochs=[t0.jd])
    horizons_table = horizons_obj.ephemerides(
        quantities="1,36,37", extra_precision=True
    )
    horizons_coord = SkyCoord(
        ra=horizons_table["RA"][0], dec=horizons_table["DEC"][0], unit=(u.deg, u.deg)
    )

    jorb_table = horizons_bulk_astrometry_query("274301", "500@0", t0)
    jorb_coord = SkyCoord(
        ra=jorb_table["RA"], dec=jorb_table["DEC"], unit=(u.deg, u.deg)
    )

    np.testing.assert_allclose(
        jorb_coord.separation(horizons_coord).to(u.arcsec)[0].value, 0.0, atol=1e-5
    )


def test_horizons_astrometry_bulk():
    t0 = Time("2024-12-24T00:00:00", scale="utc")

    large_times = Time(t0.jd + jnp.linspace(0, 365, 1000), format="jd", scale="utc")
    jorb_table = horizons_bulk_astrometry_query("274301", "500@0", large_times)
    jorb_coord = SkyCoord(
        ra=jorb_table["RA"], dec=jorb_table["DEC"], unit=(u.deg, u.deg)
    )[-10:]
    horizons_obj = Horizons(id="274301", location="@0", epochs=large_times.utc.jd[-10:])
    horizons_table = horizons_obj.ephemerides(
        quantities="1,36,37", extra_precision=True
    )
    horizons_coord = SkyCoord(
        ra=horizons_table["RA"], dec=horizons_table["DEC"], unit=(u.deg, u.deg)
    )

    seps = horizons_coord.separation(jorb_coord).to(u.arcsec).value
    np.testing.assert_allclose(seps, 0.0, atol=1e-5)

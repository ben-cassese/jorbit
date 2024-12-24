import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpy as np

from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.jplhorizons import Horizons
from jorbit.utils.horizons import (
    horizons_bulk_vector_query,
    horizons_bulk_astrometry_query,
    old_horizons_bulk_vector_query,
)


def test_horizons_comparison():
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
        jorb_table["x"].values, horizons_table["x"].values, rtol=1e-10
    )
    np.testing.assert_allclose(
        jorb_table["y"].values, horizons_table["y"].values, rtol=1e-10
    )
    np.testing.assert_allclose(
        jorb_table["z"].values, horizons_table["z"].values, rtol=1e-10
    )

"""Tests for the orbit_fit_seeds module."""
import jax

jax.config.update("jax_enable_x64", True)

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from jorbit import Observations
from jorbit.astrometry.orbit_fit_seeds import gauss_method_orbit, simple_circular


def test_gauss_method_orbit() -> None:
    """Test the Gauss method for orbit determination."""
    np.random.seed(13)

    def percentage_difference(a: float, b: float) -> float:
        """Compute the percentage difference between two values."""
        return np.abs(a - b) / np.max(np.abs([a, b])) * 100

    times = Time(["2020-01-01", "2020-01-02", "2020-02-01"])

    for _i in range(25):
        sso_id = str(np.random.randint(1_000, 200_000))

        obj = Horizons(id=sso_id, location="695@399", epochs=times.utc.jd)
        eph = obj.ephemerides()
        coords = SkyCoord(eph["RA"], eph["DEC"])
        obj = Horizons(id=sso_id, location="500@0", epochs=times.utc.jd)
        elements = obj.elements(refplane="ecliptic")

        obs = Observations(
            observed_coordinates=coords,
            times=times,
            observatories="kitt peak",
            astrometric_uncertainties=1*u.arcsec,
        )
        gauss_orbit = gauss_method_orbit(obs).to_keplerian()

        thresh = 20
        if np.mean(elements["a"]) < 3.0:
            assert np.abs(np.mean(elements["a"]) - gauss_orbit.semi[0]) < 1.0
        else:
            assert (percentage_difference(np.mean(elements["a"]), gauss_orbit.semi[0])) < thresh
        assert np.abs(np.mean(elements["e"]) - gauss_orbit.ecc[0]) < 0.1
        assert np.abs(np.mean(elements["incl"]) - gauss_orbit.inc[0]) < 10.0


def test_simple_circular() -> None:
    """Test the simple circular orbit fit."""
    np.random.seed(13)

    for _i in range(100):
        k = simple_circular(
            ra=np.random.uniform(0, 2*np.pi),
            dec=np.random.uniform(-np.pi/2, np.pi/2),
            semi=np.random.uniform(1, 100),
            time=0.0,
        ).to_keplerian()

        assert np.isfinite(k.semi)
        assert np.isfinite(k.ecc)
        assert np.isfinite(k.inc)
        assert np.isfinite(k.Omega)
        assert np.isfinite(k.omega)
        assert np.isfinite(k.nu)

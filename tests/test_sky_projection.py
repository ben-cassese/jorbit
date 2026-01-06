"""Test the sky_projection module."""

import jax

jax.config.update("jax_enable_x64", True)
import astropy.units as u
import jax.numpy as jnp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from jorbit.accelerations import create_newtonian_ephemeris_acceleration_func
from jorbit.astrometry.sky_projection import on_sky, sky_sep, tangent_plane_projection
from jorbit.ephemeris import Ephemeris
from jorbit.utils.horizons import get_observer_positions


def setup() -> tuple[np.ndarray, np.ndarray]:
    """Gather data from Horizons and compare to Jorbit's on_sky function."""
    # gather positions and observed coordinates for Kitt Peak

    t0_kp = Time("2024-12-01 00:00")
    t1_kp = Time("2025-12-01 00:00")

    obj = Horizons(
        id="274301",
        location="500@0",
        epochs={"start": t0_kp.tdb.iso, "stop": t1_kp.tdb.iso, "step": "2d"},
    )
    vecs_kp = obj.vectors(refplane="earth")
    xs_kp = jnp.array([vecs_kp["x"], vecs_kp["y"], vecs_kp["z"]]).T
    vs_kp = jnp.array([vecs_kp["vx"], vecs_kp["vy"], vecs_kp["vz"]]).T
    times_kp = jnp.array(vecs_kp["datetime_jd"])

    obj = Horizons(
        id="274301",
        location="695@399",
        epochs={"start": t0_kp.utc.iso, "stop": t1_kp.utc.iso, "step": "2d"},
    )
    coords_kp = obj.ephemerides(extra_precision=True, quantities="1")
    ra_kp = jnp.array(coords_kp["RA"])
    dec_kp = jnp.array(coords_kp["DEC"])

    # gather positions and observed coordinates for Palomar

    t0_pal = Time("2024-12-02 00:00")
    t1_pal = Time("2025-12-02 00:00")

    obj = Horizons(
        id="274301",
        location="500@0",
        epochs={"start": t0_pal.tdb.iso, "stop": t1_pal.tdb.iso, "step": "2d"},
    )
    vecs_pal = obj.vectors(refplane="earth")
    xs_pal = jnp.array([vecs_pal["x"], vecs_pal["y"], vecs_pal["z"]]).T
    vs_pal = jnp.array([vecs_pal["vx"], vecs_pal["vy"], vecs_pal["vz"]]).T
    times_pal = jnp.array(vecs_pal["datetime_jd"])

    obj = Horizons(
        id="274301",
        location="675@399",
        epochs={"start": t0_pal.utc.iso, "stop": t1_pal.utc.iso, "step": "2d"},
    )
    coords_pal = obj.ephemerides(extra_precision=True, quantities="1")
    ra_pal = jnp.array(coords_pal["RA"])
    dec_pal = jnp.array(coords_pal["DEC"])

    # combine everything
    xs = jnp.concatenate([xs_kp, xs_pal])
    vs = jnp.concatenate([vs_kp, vs_pal])
    times = jnp.concatenate([times_kp, times_pal])
    ra = jnp.concatenate([ra_kp, ra_pal])
    dec = jnp.concatenate([dec_kp, dec_pal])
    observatories = ["kitt peak"] * len(xs_kp) + ["palomar"] * len(xs_pal)

    order = jnp.argsort(times)
    xs = xs[order]
    vs = vs[order]
    times = Time(times[order], format="jd", scale="tdb")
    ra = ra[order]
    dec = dec[order]
    observatories = np.array(observatories)[order]

    # get the observer positions
    obs_pos = get_observer_positions(times, observatories)

    eph = Ephemeris("default planets")
    acc_func = create_newtonian_ephemeris_acceleration_func(eph.processor)

    calc_ras, calc_decs = jax.vmap(on_sky, in_axes=(0, 0, 0, 0, None))(
        xs, vs, times.tdb.jd, obs_pos, acc_func
    )

    calc_skycoords = SkyCoord(ra=calc_ras, dec=calc_decs, unit=(u.rad, u.rad))
    true_skycoords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

    seps_astropy = calc_skycoords.separation(true_skycoords).to(u.arcsec)
    seps_jorbit = jax.vmap(sky_sep, in_axes=(0, 0, 0, 0))(
        calc_ras, calc_decs, ra * u.deg.to(u.rad), dec * u.deg.to(u.rad)
    )

    return seps_astropy, seps_jorbit


def test_sky_sep() -> None:
    """Test that the sky_sep function agrees w/ Astropy's SkyCoord.separation method."""
    seps_astropy, seps_jorbit = setup()

    np.testing.assert_allclose(seps_astropy.value, seps_jorbit, atol=1e-6)


def test_on_sky() -> None:
    """Test the on_sky function.

    Given the same Cartesian position/velocity of an object and the same position of an
    observer, should agree with the Horizons astrometry to within 0.1 mas.
    """
    _seps_astropy, seps_jorbit = setup()
    np.testing.assert_allclose(seps_jorbit, 0.0, atol=1e-4)  # 0.1 mas


def test_tangent_plane_projection() -> None:
    """Test the tangent_plane_projection function."""
    np.random.seed(0)
    for _i in range(1_000):
        c1 = SkyCoord(
            ra=np.random.uniform(0, 2 * np.pi, 1) * u.rad,
            dec=np.random.uniform(-np.pi / 2, np.pi / 2, 1) * u.rad,
        )
        ang = np.random.uniform(0, 2 * np.pi, 1) * u.rad
        sep = np.random.uniform(0, 1, 1) * u.arcsec
        c2 = c1.directional_offset_by(position_angle=ang, separation=sep)
        sep = sep.to_value(u.arcsec)

        ra_ref, dec_ref = c1.ra.rad, c1.dec.rad
        ra, dec = c2.ra.rad, c2.dec.rad

        xi_eta = tangent_plane_projection(ra, dec, ra_ref, dec_ref)
        projected_sep = np.sqrt(xi_eta[0] ** 2 + xi_eta[1] ** 2)

        np.testing.assert_allclose(projected_sep, sep, rtol=1e-4)  # 0.1 mas agreement

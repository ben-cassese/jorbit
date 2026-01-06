"""Test that transformations between Cartesian and Keplerian are consistent and agree w/ Horizons."""

import jax

jax.config.update("jax_enable_x64", True)
import astropy.units as u
import jax.numpy as jnp
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from jorbit.astrometry.transformations import (
    cartesian_to_elements,
    elements_to_cartesian,
    horizons_ecliptic_to_icrs,
    icrs_to_horizons_ecliptic,
)


def test_elements_to_cartesian() -> None:
    """Test that, given matching elements, we get the same cartesian as Horizons."""
    t0 = Time("2024-12-01 00:00")

    obj = Horizons(
        id="274301",
        location="500@0",
        epochs=t0.tdb.jd,
    )
    vecs = obj.vectors(refplane="earth")
    true_xs = jnp.array([vecs["x"], vecs["y"], vecs["z"]]).T
    # true_vs = jnp.array([vecs["vx"], vecs["vy"], vecs["vz"]]).T

    elements = obj.elements(refplane="ecliptic")

    a_horizons = jnp.array([elements["a"][0]])
    ecc_horizons = jnp.array([elements["e"][0]])
    inc_horizons = jnp.array([elements["incl"][0]])
    Omega_horizons = jnp.array([elements["Omega"][0]])
    omega_horizons = jnp.array([elements["w"][0]])
    nu_horizons = jnp.array([elements["nu"][0]])

    xs, _vs = elements_to_cartesian(
        a=a_horizons,
        ecc=ecc_horizons,
        inc=inc_horizons,
        Omega=Omega_horizons,
        omega=omega_horizons,
        nu=nu_horizons,
    )
    xs = horizons_ecliptic_to_icrs(xs)
    # vs = horizons_ecliptic_to_icrs(vs)

    assert jnp.allclose(xs, true_xs, atol=1e-11)  # 1m


def test_cartesian_to_elements() -> None:
    """Test that, given matching cartesian, we get the same elements as Horizons."""
    t0 = Time("2024-12-01 00:00")

    obj = Horizons(
        id="274301",
        location="500@0",
        epochs=t0.tdb.jd,
    )
    vecs = obj.vectors(refplane="earth")
    true_xs = jnp.array([vecs["x"], vecs["y"], vecs["z"]]).T
    true_vs = jnp.array([vecs["vx"], vecs["vy"], vecs["vz"]]).T

    elements = obj.elements(refplane="ecliptic")
    a_horizons = jnp.array([elements["a"][0]])
    ecc_horizons = jnp.array([elements["e"][0]])
    inc_horizons = jnp.array([elements["incl"][0]])
    Omega_horizons = jnp.array([elements["Omega"][0]])
    omega_horizons = jnp.array([elements["w"][0]])
    nu_horizons = jnp.array([elements["nu"][0]])

    xs = icrs_to_horizons_ecliptic(true_xs)
    vs = icrs_to_horizons_ecliptic(true_vs)
    a, ecc, nu, inc, Omega, omega = cartesian_to_elements(
        x=xs,
        v=vs,
    )

    assert jnp.allclose(a, a_horizons, atol=1e-11)  # 1m
    assert jnp.allclose(ecc, ecc_horizons, atol=1e-9)
    assert jnp.allclose(nu, nu_horizons, atol=1e-6 * u.deg.to(u.rad))
    assert jnp.allclose(inc, inc_horizons, atol=1e-6 * u.deg.to(u.rad))
    assert jnp.allclose(Omega, Omega_horizons, atol=1e-6 * u.deg.to(u.rad))
    assert jnp.allclose(omega, omega_horizons, atol=1e-6 * u.deg.to(u.rad))


def test_inverses() -> None:
    """Test that elements_to_cartesian and cartesian_to_elements are inverses."""
    t0 = Time("2024-12-01 00:00")

    obj = Horizons(
        id="274301",
        location="500@0",
        epochs=t0.tdb.jd,
    )
    vecs = obj.vectors(refplane="earth")
    true_xs = jnp.array([vecs["x"], vecs["y"], vecs["z"]]).T
    converted = horizons_ecliptic_to_icrs(icrs_to_horizons_ecliptic(true_xs))
    assert jnp.allclose(true_xs, converted, atol=1e-15)

    vecs = obj.vectors(refplane="ecliptic")
    true_xs = jnp.array([vecs["x"], vecs["y"], vecs["z"]]).T
    true_vs = jnp.array([vecs["vx"], vecs["vy"], vecs["vz"]]).T
    converted = icrs_to_horizons_ecliptic(horizons_ecliptic_to_icrs(true_xs))
    assert jnp.allclose(true_xs, converted, atol=1e-15)

    a, ecc, nu, inc, Omega, omega = cartesian_to_elements(
        x=true_xs,
        v=true_vs,
    )
    converted_xs, _converted_vs = elements_to_cartesian(
        a=a,
        ecc=ecc,
        nu=nu,
        inc=inc,
        Omega=Omega,
        omega=omega,
    )
    assert jnp.allclose(true_xs, converted_xs, atol=1e-15)

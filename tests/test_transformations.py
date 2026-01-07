"""Test that transformations between Cartesian and Keplerian are consistent and agree w/ Horizons."""

import jax

jax.config.update("jax_enable_x64", True)
import astropy.units as u
import jax.numpy as jnp
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from jorbit import Particle
from jorbit.astrometry.transformations import (
    cartesian_to_elements,
    elements_to_cartesian,
    horizons_ecliptic_to_icrs,
    icrs_to_horizons_ecliptic,
)
from jorbit.utils.kepler import M_from_f
from jorbit.utils.states import barycentric_to_heliocentric, heliocentric_to_barycentric


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


def test_barycentric_to_heliocentric_cartesian() -> None:
    """Test barycentric to heliocentric Cartesian conversion."""
    epoch = Time(61000.0, format="mjd", scale="tdb")
    obj = Horizons(
        id="J35M00J",
        location="500@10",
        epochs=epoch.tdb.jd,
    )
    vecs = obj.vectors(refplane="earth")
    horizons_x = jnp.squeeze(
        jnp.array([vecs["x"].value, vecs["y"].value, vecs["z"].value])
    )
    horizons_v = jnp.squeeze(
        jnp.array([vecs["vx"].value, vecs["vy"].value, vecs["vz"].value])
    )

    p = Particle.from_horizons(name="J35M00J", time=epoch)
    jorbit_helio = barycentric_to_heliocentric(state=p.cartesian_state, time=epoch)

    assert jnp.allclose(jorbit_helio["x_helio"][0], horizons_x, atol=1e-10)
    assert jnp.allclose(jorbit_helio["v_helio"][0], horizons_v, atol=1e-10)


def test_barycentric_to_heliocentric_keplerian() -> None:
    """Test barycentric to heliocentric Keplerian conversion."""
    epoch = Time(61000.0, format="mjd", scale="tdb")

    obj = Horizons(
        id="J35M00J",
        location="500@10",
        epochs=epoch.tdb.jd,
    )
    heliocentric_elements = obj.elements(refplane="ecliptic")
    heliocentric_elements = jnp.array(
        [
            heliocentric_elements["a"],
            heliocentric_elements["e"],
            heliocentric_elements["incl"],
            heliocentric_elements["Omega"],
            heliocentric_elements["w"],
            heliocentric_elements["M"],
        ]
    ).T

    p = Particle.from_horizons(name="J35M00J", time=epoch)
    jorbit_helio = barycentric_to_heliocentric(state=p.keplerian_state, time=epoch)
    jorbit_helio = jnp.squeeze(
        jnp.array(
            [
                jorbit_helio["a_helio"],
                jorbit_helio["ecc_helio"],
                jorbit_helio["inc_helio"],
                jorbit_helio["Omega_helio"],
                jorbit_helio["omega_helio"],
                M_from_f(
                    jorbit_helio["nu_helio"] * jnp.pi / 180.0, jorbit_helio["ecc_helio"]
                )
                * 180.0
                / jnp.pi,
            ]
        )
    )
    difference = jorbit_helio - heliocentric_elements
    assert jnp.allclose(difference, 0, atol=1e-10)


def test_heliocentric_to_barycentric_cartesian() -> None:
    """Test heliocentric to barycentric Cartesian conversion."""
    epoch = Time(61000.0, format="mjd", scale="tdb")
    obj = Horizons(
        id="J35M00J",
        location="500@10",
        epochs=epoch.tdb.jd,
    )
    vecs = obj.vectors(refplane="earth")
    horizons_x = jnp.squeeze(
        jnp.array([vecs["x"].value, vecs["y"].value, vecs["z"].value])
    )
    horizons_v = jnp.squeeze(
        jnp.array([vecs["vx"].value, vecs["vy"].value, vecs["vz"].value])
    )

    horizons_bary = heliocentric_to_barycentric(
        {"x_helio": horizons_x, "v_helio": horizons_v}, epoch
    )

    p = Particle.from_horizons(name="J35M00J", time=epoch)

    assert jnp.allclose(p.cartesian_state.x[0], horizons_bary.x, atol=1e-10)
    assert jnp.allclose(p.cartesian_state.v[0], horizons_bary.v, atol=1e-10)


def test_heliocentric_to_barycentric_keplerian() -> None:
    """Test heliocentric to barycentric Keplerian conversion."""
    epoch = Time(61000.0, format="mjd", scale="tdb")
    obj = Horizons(
        id="J35M00J",
        location="500@10",
        epochs=epoch.tdb.jd,
    )
    horizons_helio_elements = obj.elements(refplane="ecliptic")
    horizons_helio_elements = {
        "a_helio": horizons_helio_elements["a"][0],
        "ecc_helio": horizons_helio_elements["e"][0],
        "inc_helio": horizons_helio_elements["incl"][0],
        "Omega_helio": horizons_helio_elements["Omega"][0],
        "omega_helio": horizons_helio_elements["w"][0],
        "nu_helio": horizons_helio_elements["nu"][0],
    }
    horizons_bary_elements = heliocentric_to_barycentric(
        heliocentric_dict=horizons_helio_elements,
        time=epoch,
    )

    p = Particle.from_horizons(name="J35M00J", time=epoch)
    assert jnp.allclose(p.keplerian_state.semi, horizons_bary_elements.semi, atol=1e-10)
    assert jnp.allclose(p.keplerian_state.ecc, horizons_bary_elements.ecc, atol=1e-10)
    assert jnp.allclose(p.keplerian_state.inc, horizons_bary_elements.inc, atol=1e-10)
    assert jnp.allclose(
        p.keplerian_state.Omega, horizons_bary_elements.Omega, atol=1e-10
    )
    assert jnp.allclose(
        p.keplerian_state.omega, horizons_bary_elements.omega, atol=1e-10
    )
    assert jnp.allclose(p.keplerian_state.nu, horizons_bary_elements.nu, atol=1e-10)

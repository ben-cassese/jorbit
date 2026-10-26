"""Perturber refinement against an independent SPK/light-time calculation."""

import logging
from collections.abc import Iterator
from pathlib import Path

import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.utils.data import download_file
from jplephem.spk import SPK

from jorbit.data.constants import (
    DEFAULT_ASTEROID_EPHEMERIS_URL,
    DEFAULT_PLANET_EPHEMERIS_URL,
    INV_SPEED_OF_LIGHT,
    PERTURBER_PACKED_DESIGNATIONS,
)
from jorbit.mpchecker import mpchecker, nearest_asteroid
from jorbit.mpchecker import parse_jorbit_ephem as core


@pytest.fixture(scope="module", autouse=True)
def _clear_ephemeris_cache() -> Iterator[None]:
    """Release retained ephemeris arrays when this module finishes."""
    core._mpchecker_ephemeris.cache_clear()
    yield
    core._mpchecker_ephemeris.cache_clear()


@pytest.fixture
def catalogue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[np.ndarray, pl.DataFrame]:
    """Use a compact, deliberately interleaved catalogue and two plausible tracers."""
    names = np.array([*PERTURBER_PACKED_DESIGNATIONS[::-1], "00006", "00008"])
    names[[1, 17]] = names[[17, 1]]
    metadata = pl.DataFrame(
        {
            "Packed designation": names,
            "Unpacked Name": names,
            "H": np.linspace(6.0, 10.0, len(names)),
            "G": np.full(len(names), 0.15),
        }
    )
    x0 = np.zeros((len(names), 3))
    v0 = np.zeros_like(x0)
    ordinary = ~np.isin(names, PERTURBER_PACKED_DESIGNATIONS)
    x0[ordinary] = [[2.4, 0.3, 0.1], [3.0, -0.4, 0.3]]
    v0[ordinary] = [[0.0, 0.011, 0.0], [0.0, 0.01, 0.0]]
    for name, data in [("names", names), ("x0", x0), ("v0", v0)]:
        np.save(tmp_path / f"{name}.npy", data)
    monkeypatch.setattr(
        core,
        "download_file_wrapper",
        lambda url: str(tmp_path / url.rsplit("/", 1)[-1]),
    )
    monkeypatch.setattr(core, "load_mpcorb", lambda: metadata)
    return names, metadata


@pytest.fixture(scope="module")
def kernels() -> Iterator[tuple[SPK, SPK]]:
    """Read the same JPL files through jplephem rather than Jorbit's processor."""
    planets = SPK.open(download_file(DEFAULT_PLANET_EPHEMERIS_URL, cache=True))
    asteroids = SPK.open(download_file(DEFAULT_ASTEROID_EPHEMERIS_URL, cache=True))
    yield planets, asteroids
    planets.close()
    asteroids.close()


def _vector(kernel: SPK, center: int, target: int, jd: float) -> np.ndarray:
    """Select the SPK segment covering this time and return its position in AU."""
    segment = next(
        s
        for s in kernel.segments
        if s.center == center and s.target == target and s.start_jd <= jd <= s.end_jd
    )
    return segment.compute(jd) / u.au.to(u.km)


def _position(kernels: tuple[SPK, SPK], packed: str, jd: float) -> np.ndarray:
    """Get barycentric positions, including the Sun-to-barycenter translation."""
    planets, asteroids = kernels
    if packed == "D4340":
        return _vector(planets, 0, 9, jd)
    return _vector(planets, 0, 10, jd) + _vector(
        asteroids, 10, 2000000 + int(packed), jd
    )


def _observers(
    kernels: tuple[SPK, SPK], times: Time, offset: float = 0.0
) -> np.ndarray:
    """Return Earth positions, optionally shifted by an Earth radius along ICRS x."""
    planets, _ = kernels
    return np.array(
        [
            _vector(planets, 0, 3, jd) + _vector(planets, 3, 399, jd)
            for jd in np.atleast_1d(times.tdb.jd)
        ]
    ) + np.array([offset, 0.0, 0.0])


def _reference(
    kernels: tuple[SPK, SPK], names: np.ndarray, times: Time, observers: np.ndarray
) -> tuple[SkyCoord, np.ndarray]:
    """Solve light travel time independently with eight iterations per target/time."""
    positions, directions = [], []
    for name in names:
        body_positions, body_directions = [], []
        for jd, observer in zip(np.atleast_1d(times.tdb.jd), observers, strict=True):
            position = _position(kernels, name, jd)
            body_positions.append(position)
            for _ in range(8):
                dt = np.linalg.norm(position - observer) * INV_SPEED_OF_LIGHT
                position = _position(kernels, name, jd - dt)
            body_directions.append(position - observer)
        positions.append(body_positions)
        directions.append(body_directions)
    directions = np.array(directions)
    coords = SkyCoord(
        x=directions[..., 0],
        y=directions[..., 1],
        z=directions[..., 2],
        representation_type="cartesian",
    ).spherical
    return SkyCoord(ra=coords.lon, dec=coords.lat), np.swapaxes(positions, 0, 1)


@pytest.mark.parametrize(
    "scalar,offset", [(True, 0.0), (False, 6378.0 / u.au.to(u.km))]
)
def test_perturbers_match_spk(
    catalogue: tuple[np.ndarray, pl.DataFrame],
    kernels: tuple[SPK, SPK],
    monkeypatch: pytest.MonkeyPatch,
    scalar: bool,
    offset: float,
) -> None:
    """All 17 bodies have correct identities, light-time, shapes, and photometry."""
    names, _ = catalogue
    flags = np.isin(names, PERTURBER_PACKED_DESIGNATIONS)
    metadata = core.get_relevant_mpcorb(flags)
    times = Time("2025-01-01", scale="tdb")
    if not scalar:
        times = times + np.array([0.0, 1.0, 32.0]) * u.day
    observers = _observers(kernels, times, offset)
    reference, positions = _reference(kernels, names[flags], times, observers)
    coordinate = reference[0, 0]
    radius = 600.0
    queries = []

    def get_observers(t: Time, observer: str, version: str) -> jnp.ndarray:
        queries.append((observer, version))
        np.testing.assert_allclose(np.atleast_1d(t.tdb.jd), np.atleast_1d(times.tdb.jd))
        return jnp.array(observers)

    def no_integration(**kwargs: dict) -> None:
        pytest.fail("A perturber-only field must not construct a System")

    monkeypatch.setattr(core, "get_observer_positions", get_observers)
    monkeypatch.setattr(core, "System", no_integration)
    coords, seps, coord_table, mags, mag_table, total = core.extra_precision_calcs(
        flags,
        times,
        radius,
        "geocentric" if scalar else "test site",
        coordinate,
        metadata,
    )
    assert queries == [("500@399" if scalar else "test site", "440")]
    assert coords.shape == reference.shape
    assert np.all(coords.separation(reference).to_value(u.mas) < 0.1)
    expected_seps = reference.separation(coordinate).to_value(u.arcsec)
    np.testing.assert_allclose(seps, expected_seps, atol=1e-6)
    expected_mags = np.array(
        [
            [
                core.apparent_mag(h, g, position, observer)
                for position, observer in zip(positions[:, i], observers, strict=True)
            ]
            for i, (h, g) in enumerate(zip(metadata["H"], metadata["G"], strict=True))
        ]
    )
    np.testing.assert_allclose(mags, expected_mags, atol=1e-10)
    assert list(coord_table["name"]) == list(names[flags])
    assert list(mag_table["name"]) == list(names[flags])
    np.testing.assert_allclose(mag_table["mag"], expected_mags, atol=1e-10)
    flux = np.sum(
        np.where(expected_seps < radius, 10 ** (-0.4 * expected_mags), 0), axis=0
    )
    with np.errstate(divide="ignore"):
        expected_total = -2.5 * np.log10(flux)
    np.testing.assert_allclose(total, expected_total, atol=1e-10)


def test_mixed_field_preserves_integrated_asteroids(
    catalogue: tuple[np.ndarray, pl.DataFrame],
    kernels: tuple[SPK, SPK],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Splitting a mixed field preserves ordinary orbits and the catalogue order."""
    names, _ = catalogue
    ordinary = ~np.isin(names, PERTURBER_PACKED_DESIGNATIONS)
    times = Time("2020-01-01") + np.array([0.0, 0.25]) * u.day
    observers = jnp.array(_observers(kernels, times))
    coordinate = SkyCoord(0 * u.deg, 0 * u.deg)
    original_system = core.System
    tracer_counts = []
    processors = []

    def make_system(**kwargs: dict) -> original_system:
        tracer_counts.append(len(kwargs["state"].tracer_positions))
        processors.append(kwargs["gravity"].args[0])
        return original_system(**kwargs)

    monkeypatch.setattr(core, "System", make_system)
    mixed = core.extra_precision_calcs(
        np.ones(len(names), dtype=bool),
        times,
        648000.0,
        "geocentric",
        coordinate,
        core.get_relevant_mpcorb(np.ones(len(names), dtype=bool)),
        observer_positions=observers,
    )
    isolated = core.extra_precision_calcs(
        ordinary,
        times,
        648000.0,
        "geocentric",
        coordinate,
        core.get_relevant_mpcorb(ordinary),
        observer_positions=observers,
    )
    assert tracer_counts == [2, 2]
    assert processors[0] is processors[1]
    assert (
        processors[0]
        is core._mpchecker_ephemeris("440", "default solar system").processor
    )
    assert np.all(mixed[0][ordinary].separation(isolated[0]).to_value(u.mas) < 0.001)
    np.testing.assert_allclose(mixed[1][ordinary], isolated[1], atol=1e-8)
    np.testing.assert_allclose(mixed[3][ordinary], isolated[3], atol=1e-10)
    reference, _ = _reference(kernels, names[~ordinary], times, np.asarray(observers))
    assert np.all(mixed[0][~ordinary].separation(reference).to_value(u.mas) < 0.1)
    assert list(mixed[2]["name"]) == list(names)


@pytest.mark.parametrize("search", ["mpchecker", "nearest_asteroid"])
def test_public_search_with_europa(
    catalogue: tuple[np.ndarray, pl.DataFrame],
    kernels: tuple[SPK, SPK],
    monkeypatch: pytest.MonkeyPatch,
    search: str,
) -> None:
    """Both public searches refine Europa, including automatic precision for a site."""
    names, _ = catalogue
    times = Time("2025-01-01", scale="tdb") + np.array([0.0, 0.01]) * u.day
    observers = _observers(kernels, times, 6378.0 / u.au.to(u.km))
    reference, _ = _reference(kernels, np.array(["00052"]), times, observers)
    coordinate = reference[0, 0]
    # A tiny synthetic coarse catalogue isolates Europa; refinement uses real JPL data.
    coefficients = jnp.zeros((len(names), 2, 2))
    coefficients = coefficients.at[:, -1, 0].set(coordinate.ra.rad + np.pi)
    coefficients = coefficients.at[names == "00052", -1].set(
        jnp.array([coordinate.ra.rad, coordinate.dec.rad])
    )
    monkeypatch.setattr(
        core, "get_observer_positions", lambda *args: jnp.array(observers[:1])
    )
    if search == "mpchecker":
        result = mpchecker(
            coordinate,
            times[0],
            radius=10 * u.arcmin,
            observer="test site",
            chunk_coefficients=coefficients,
        )
        assert list(result["Packed designation"]) == ["00052"]
        coords = SkyCoord(result["ra"], result["dec"])
        assert coords.separation(reference[0, :1]).to_value(u.mas)[0] < 0.1
        assert np.all(np.isfinite(result["est. Vmag"]))
    else:
        precomputed = (
            core.setup_checks(coordinate, times, 0 * u.arcsec),
            [coefficients],
            jnp.array(observers),
        )
        _, metadata, coords, mags, total = nearest_asteroid(
            coordinate,
            times,
            radius=10 * u.arcmin,
            compute_contamination=True,
            observer="test site",
            precomputed=precomputed,
        )
        assert list(metadata["Packed designation"]) == ["00052"]
        assert np.all(coords["coord"].separation(reference).to_value(u.mas) < 0.1)
        np.testing.assert_allclose(total, mags["mag"][0], atol=1e-10)


def test_ephemeris_cache_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain one object for each version/body-group combination, without downloads."""
    constructed = []

    def make_ephemeris(**kwargs: dict) -> object:
        constructed.append(kwargs)
        return object()

    core._mpchecker_ephemeris.cache_clear()
    monkeypatch.setattr(core, "Ephemeris", make_ephemeris)
    try:
        objects = []
        for version in ["440", "430"]:
            for group in ["default solar system", "default planets"]:
                first = core._mpchecker_ephemeris(version, group)
                assert first is core._mpchecker_ephemeris(version, group)
                objects.append(first)
        assert len({id(obj) for obj in objects}) == 4
        assert len(constructed) == 4
        assert core._mpchecker_ephemeris.cache_info().maxsize == 4
        for arguments in constructed:
            assert arguments["earliest_time"] == Time("1999-12-30")
            assert arguments["latest_time"] == Time("2040-01-02")
    finally:
        core._mpchecker_ephemeris.cache_clear()


@pytest.mark.parametrize("field", ["ordinary", "perturber", "mixed"])
def test_repeated_refinement_reuses_compiled_code(
    catalogue: tuple[np.ndarray, pl.DataFrame],
    kernels: tuple[SPK, SPK],
    caplog: pytest.LogCaptureFixture,
    field: str,
) -> None:
    """Reuse numerical kernels; new step counts may compile buffer preparation once."""
    names, _ = catalogue
    selected = {
        "ordinary": ["00006"],
        "perturber": ["00052"],
        "mixed": ["00006", "00052"],
    }
    flags = np.isin(names, selected[field])
    metadata = core.get_relevant_mpcorb(flags)
    times = Time("2020-01-01") + np.array([0.0, 0.25]) * u.day
    observers = jnp.array(_observers(kernels, times))
    changed_observers = observers + jnp.array([1e-5, 0.0, 0.0])
    coordinate = SkyCoord(0 * u.deg, 0 * u.deg)

    def refine(t: Time, observer_positions: jnp.ndarray) -> None:
        result = core.extra_precision_calcs(
            flags,
            t,
            648000.0,
            "geocentric",
            coordinate,
            metadata,
            observer_positions=observer_positions,
        )
        jax.block_until_ready((result[1], result[3], result[5]))
        assert np.all(np.isfinite(result[0].ra.rad))

    def compilations(t: Time, observer_positions: jnp.ndarray) -> list[str]:
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="jax"), jax.log_compiles(True):
            refine(t, observer_positions)
        return [
            r.getMessage() for r in caplog.records if "Compiling jit(" in r.getMessage()
        ]

    refine(times, observers)
    repeated = compilations(times, observers)
    assert not repeated, "\n".join(repeated)

    changed_times = times + 0.125 * u.day
    # The stitched trajectory is sliced and padded according to its accepted step count.
    preparation = ("dynamic_slice", "broadcast_in_dim", "concatenate")
    unexpected = [
        message
        for message in compilations(changed_times, changed_observers)
        if not any(f"Compiling jit({name})" in message for name in preparation)
    ]
    assert not unexpected, "\n".join(unexpected)
    repeated = compilations(changed_times, changed_observers)
    assert not repeated, "\n".join(repeated)


@pytest.mark.parametrize(
    "gravity",
    [
        "newtonian planets",
        "newtonian solar system",
        "gr planets",
        "gr solar system",
        "default solar system",
    ],
)
def test_cached_gravity_preserves_accelerations(
    catalogue: tuple[np.ndarray, pl.DataFrame],
    kernels: tuple[SPK, SPK],
    monkeypatch: pytest.MonkeyPatch,
    gravity: str,
) -> None:
    """Cached gravity agrees with System's original model, including planet-only models."""
    names, _ = catalogue
    flags = names == "00006"
    time = Time("2020-01-01")
    observers = jnp.array(_observers(kernels, time))
    original_system = core.System
    systems = []

    def make_system(**kwargs: dict) -> original_system:
        system = original_system(**kwargs)
        systems.append(system)
        return system

    monkeypatch.setattr(core, "System", make_system)
    core.extra_precision_calcs(
        flags,
        time,
        648000.0,
        "geocentric",
        SkyCoord(0 * u.deg, 0 * u.deg),
        core.get_relevant_mpcorb(flags),
        gravity=gravity,
        observer_positions=observers,
    )
    cached = systems[0]
    reference = original_system(
        state=cached._state,
        gravity=gravity,
        earliest_time=Time("1999-12-30"),
        latest_time=Time("2040-01-02"),
    )
    np.testing.assert_allclose(
        cached.gravity(cached._state),
        reference.gravity(cached._state),
        rtol=1e-14,
        atol=1e-18,
    )
    group = "default planets" if gravity.endswith("planets") else "default solar system"
    assert cached.gravity.args[0] is core._mpchecker_ephemeris("440", group).processor


def test_custom_gravity_bypasses_cache(
    catalogue: tuple[np.ndarray, pl.DataFrame],
    kernels: tuple[SPK, SPK],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ordinary-only field with custom gravity does not load a built-in ephemeris."""
    names, _ = catalogue
    flags = names == "00006"
    time = Time("2020-01-01")

    def no_ephemeris(*args: tuple) -> None:
        pytest.fail("Custom gravity must not request the cached ephemeris")

    def acceleration(state: core.SystemState) -> jnp.ndarray:
        return jnp.zeros_like(state.tracer_positions)

    monkeypatch.setattr(core, "_mpchecker_ephemeris", no_ephemeris)
    result = core.extra_precision_calcs(
        flags,
        time,
        648000.0,
        "geocentric",
        SkyCoord(0 * u.deg, 0 * u.deg),
        core.get_relevant_mpcorb(flags),
        gravity=jax.tree_util.Partial(acceleration),
        observer_positions=jnp.array(_observers(kernels, time)),
    )
    assert np.all(np.isfinite(result[0].ra.rad))

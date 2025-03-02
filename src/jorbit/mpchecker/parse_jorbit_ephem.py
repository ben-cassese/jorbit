"""The helper functions for the mpchecker."""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", module="erfa")

import jax

jax.config.update("jax_enable_x64", True)

import astropy.units as u
import jax.numpy as jnp
import numpy as np
import polars as pl
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time

from jorbit.astrometry.sky_projection import sky_sep
from jorbit.data.constants import JORBIT_EPHEM_URL_BASE, PERTURBER_PACKED_DESIGNATIONS
from jorbit.system import System
from jorbit.utils.cache import download_file_wrapper
from jorbit.utils.horizons import get_observer_positions
from jorbit.utils.states import SystemState


@jax.jit
def get_chunk_index(
    time: Time,
    t0: float = Time("2020-01-01").tdb.jd,
    tf: float = Time("2040-01-01").tdb.jd,
    chunk_size: int = 30,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Get the index of given piecewise chunk of Chebyshev coefficients and the offset within that chunk.

    Args:
        time (Time):
            The time in question.
        t0 (float):
            The start time of the ephemeris, in JD TDB.
        tf (float):
            The end time of the ephemeris, in JD TDB.
        chunk_size (int):
            The size of each chunk, in days.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]:
            The index of the chunk and the offset within that chunk.
    """
    # 2451545.0 is the J2000 epoch in TDB
    init = (t0 - 2451545.0) * 86400.0
    intlen = chunk_size * 86400.0
    num_chunks = (jnp.ceil((tf - t0) / chunk_size)).astype(int)

    tdb2 = 0.0  # leaving in case we ever decide to increase the time precision and use 2 floats
    index1, offset1 = jnp.divmod((time - 2451545.0) * 86400.0 - init, intlen)
    index2, offset2 = jnp.divmod(tdb2 * 86400.0, intlen)
    index3, offset = jnp.divmod(offset1 + offset2, intlen)
    index = (index1 + index2 + index3).astype(int)

    omegas = index == num_chunks
    index = jnp.where(omegas, index - 1, index)
    offset = jnp.where(omegas, offset + intlen, offset)

    index = jnp.where(
        t0 < 2451545, -index, index
    )  # negative index for backwards ephemeris
    return index, offset


@jax.jit
def eval_cheby(
    coefficients: jnp.ndarray, x: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate the pair of Chebyshev polynomials describing RA, Dec at a given point.

    Similar to eval_cheby in the EphemerisProcessor class, but instead of evaluating
    three polynomials to give a cartesian position (and their derivatives to get
    velocity), this evaluates two polynomials to reconstruct the geocentric RA and Dec.

    Args:
        coefficients (jnp.ndarray):
            The Chebyshev coefficients.
        x (jnp.ndarray):
            The point at which to evaluate the polynomials.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]:
            The RA and Dec at the given point.
    """
    b_ii = jnp.zeros(2)
    b_i = jnp.zeros(2)

    def scan_func(X: tuple, a: jnp.ndarray) -> tuple:
        b_i, b_ii = X
        tmp = b_i
        b_i = a + 2 * x * b_i - b_ii
        b_ii = tmp
        return (b_i, b_ii), b_i

    (b_i, b_ii), s = jax.lax.scan(scan_func, (b_i, b_ii), coefficients[:-1])
    return coefficients[-1] + x * b_i - b_ii, s


@jax.jit
def individual_state(
    coefficients: jnp.ndarray, offset: float, t0: float, chunk_size: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Get the state of a single particle at a given time.

    Args:
        coefficients (jnp.ndarray):
            The Chebyshev coefficients.
        offset (float):
            The offset within the chunk.
        t0 (float):
            The start time of the ephemeris, in JD TDB.
        chunk_size (int):
            The size of each chunk, in days.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]:
            The RA and Dec at the given time.
    """
    intlen = chunk_size * 86400.0

    s = 2.0 * offset / intlen - 1.0

    (approx_ra, approx_dec), _ = eval_cheby(coefficients, s)
    return approx_ra % (2 * jnp.pi), approx_dec


@jax.jit
def multiple_states(
    coefficients: jnp.ndarray, offset: float, t0: float, chunk_size: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Get the state of multiple particles at a given time.

    Just a vmapping of individual_state.

    Args:
        coefficients (jnp.ndarray):
            The Chebyshev coefficients, (nparticles, N_coeffs, 2)
        offset (float):
            The offset within the chunk.
        t0 (float):
            The start time of the ephemeris, in JD TDB.
        chunk_size (int):
            The size of each chunk, in days.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]:
            The RAs and Decs of each particle at the given time.
    """
    return jax.vmap(individual_state, in_axes=(0, None, None, None))(
        coefficients, offset, t0, chunk_size
    )


def setup_checks(coordinate: SkyCoord, time: Time, radius: u.Quantity) -> tuple:
    """Check that inputs are valid for the ephemeris, convert to standard forms.

    Args:
        coordinate (SkyCoord):
            The coordinate of the target.
        time (Time):
            The time of the observation.
        radius (u.Quantity):
            The radius of the field of view. Must be a unit of angle.

    Returns:
        tuple[SkyCoord, u.Quantity, float, float, int, np.ndarray]:
            The coordinate, radius, start time, end time, chunk size, and names.
    """
    assert np.all(
        time > Time("2000-01-01 00:05")
    ), "All times must be after 2000-01-01 00:05"
    assert np.all(time < Time("2040-01-01")), "All times must be before 2040-01-01"
    coordinate = coordinate.transform_to("icrs")
    radius = radius.to(u.arcsec).value

    # hard-coded:
    t0 = np.where(
        time > Time("2020-01-01"),
        Time("2020-01-01").tdb.jd,
        Time("2000-01-01 00:00:05.000").tdb.jd,
    )
    tf = np.where(
        time > Time("2020-01-01"), Time("2040-01-01").tdb.jd, Time("2020-01-01").tdb.jd
    )
    chunk_size = 30

    # get the names of all particles- this file is < 40 MB
    names = np.load(download_file_wrapper(JORBIT_EPHEM_URL_BASE + "names.npy"))

    return coordinate, radius, t0, tf, chunk_size, names


def load_mpcorb() -> pl.DataFrame:
    """Load the mpcorb file used to generate the latest Jorbit ephemeris.

    Returns:
        pl.DataFrame:
            The mpcorb file.
    """
    df = pl.read_ipc(download_file_wrapper(JORBIT_EPHEM_URL_BASE + "mpcorb.arrow"))
    return df


def nearest_asteroid_helper(
    coordinate: SkyCoord, times: Time, observer: str | None = None
) -> tuple:
    """Pre-compute and load material for the nearest_asteroid function.

    Args:
        coordinate (SkyCoord):
            The coordinate of the target.
        times (Time):
            The times of the observation.
        observer (str):
            The observatory observing the target. Optional, defaults to None.

    Returns:
        tuple[tuple, jnp.ndarray, jnp.ndarray|None]:
            The coordinate, radius, start time, end time, chunk size, and names, then
            a merged array of all the relevant Chebyshev coefficients, then the observer
            positions.
    """
    coordinate, _, t0, tf, chunk_size, names = setup_checks(
        coordinate, times, radius=0 * u.arcsec
    )
    indices, offsets = jax.vmap(get_chunk_index, in_axes=(0, 0, 0, None))(
        times.tdb.jd, t0, tf, chunk_size
    )
    unique_indices = jnp.unique(indices)

    if len(unique_indices) > 2:
        warnings.warn(
            f"Requested times span {len(unique_indices)} chunks of the jorbit ephemeris. "
            "Beware of memory issues, as each chunk is ~250 MB and all will be  "
            "downloaded, cached, and loaded into memory. ",
            stacklevel=2,
        )

    coeffs = []
    for ind in unique_indices:
        if jnp.sign(ind) == -1:
            chunk = jnp.load(
                download_file_wrapper(
                    JORBIT_EPHEM_URL_BASE + f"chebyshev_coeffs_rev_{-ind:03d}.npy"
                )
            )
        else:
            chunk = jnp.load(
                download_file_wrapper(
                    JORBIT_EPHEM_URL_BASE + f"chebyshev_coeffs_fwd_{ind:03d}.npy"
                )
            )
        coeffs.append(chunk)

    coeffs = jnp.array(coeffs)

    if observer is not None:
        if observer == "geocentric":
            observer = "500@399"
        observer_positions = get_observer_positions(times, observer)
    else:
        observer_positions = None

    return (coordinate, _, t0, tf, chunk_size, names), coeffs, observer_positions


@jax.jit
def apparent_mag(
    h: float, g: float, target_position: jnp.ndarray, observer_position: jnp.ndarray
) -> float:
    r"""Calculate the apparent magnitude of an asteroid at a certain position from a certain observer.

    Implements the same formula as JPL Horizons,

    .. math::
        \text{Ap. mag} = H + 5*\log_{10}(\Delta) + 5*\log_{10}(r) -2.5*\log_{10}((1-G)*\phi_1 + G*\phi_2)

    where :math:`\Delta` is the distance from the observer to the target, :math:`r` is
    the distance from the target to the Sun, and :math:`\phi_1` and :math:`\phi_2` are
    phase functions that depend on the phase angle :math:`\alpha`.

    Note a minor inconsistency here: the coordinates are defined in barycentric
    coordinates, but here we assume they're heliocentric just to avoid having to query
    the position of the sun. Shouldn't matter much.

    Args:
        h (float):
            The absolute magnitude of the asteroid.
        g (float):
            The slope parameter of the asteroid.
        target_position (jnp.ndarray):
            The position of the target in barycentric coordinates.
        observer_position (jnp.ndarray):
            The position of the observer in barycentric coordinates.

    Returns:
        float:
            The apparent magnitude of the asteroid.
    """
    # APmag= H + 5*log10(delta) + 5*log10(r) -2.5*log10((1-G)*phi1 + G*phi2)

    delta_vec = target_position - observer_position
    cos_phase = jnp.dot(target_position, delta_vec) / (
        jnp.linalg.norm(target_position) * jnp.linalg.norm(delta_vec)
    )
    cos_phase = jnp.clip(cos_phase, -1.0, 1.0)
    phase_angle = jnp.arccos(cos_phase)

    tan_half_alpha = jnp.tan(phase_angle / 2)
    phi1 = jnp.exp(-3.33 * tan_half_alpha**0.63)
    phi2 = jnp.exp(-1.87 * tan_half_alpha**1.22)
    phase_function = -2.5 * jnp.log10((1 - g) * phi1 + g * phi2)

    r = jnp.linalg.norm(target_position)
    delta = jnp.linalg.norm(delta_vec)
    return h + 5 * jnp.log10(r * delta) + phase_function


def extra_precision_calcs(
    asteroid_flags: jnp.ndarray,
    times: Time,
    radius: u.Quantity,
    observer: str,
    coordinate: SkyCoord,
    relevant_mpcorb: pl.DataFrame,
    observer_positions: jnp.ndarray | None = None,
) -> tuple:
    """Helper function for running N-body ephemeris calculations.

    Args:
        asteroid_flags (jnp.ndarray):
            A boolean array indicating which asteroids to include.
        times (Time):
            The times of the observation.
        radius (u.Quantity):
            The radius of the field of view. Must be a unit of angle.
        observer (str):
            The observer to use. Must be a valid Horizons observer code.
        coordinate (SkyCoord):
            The coordinate of the target.
        relevant_mpcorb (pl.DataFrame):
            The mpcorb file used to generate the latest Jorbit ephemeris.
        observer_positions (jnp.ndarray):
            The observer positions. If None, they will be retrieved from Horizons.

    Returns:
        tuple:
            The ephemeris, separations, coordinate table, magnitudes, magnitude table,
            and total magnitudes.
    """
    names = jnp.load(download_file_wrapper(JORBIT_EPHEM_URL_BASE + "names.npy"))
    names = names[asteroid_flags]
    if np.isin(names, PERTURBER_PACKED_DESIGNATIONS).sum() > 0:
        relavant_perturbers = names[np.isin(names, PERTURBER_PACKED_DESIGNATIONS)]
        warnings.warn(
            f"Of the objects found nearby the target, {relavant_perturbers} are "
            "massive perturbers that are included in jorbit ephemeris calculations by "
            "default. Consequently, these will be dropped to avoid complications with "
            "infinite accelerations. Use the `Ephemeris` class to query the positions "
            "of these objects.",
            stacklevel=2,
        )
    x0 = jnp.load(download_file_wrapper(JORBIT_EPHEM_URL_BASE + "x0.npy"))
    x0 = x0[asteroid_flags]
    v0 = jnp.load(download_file_wrapper(JORBIT_EPHEM_URL_BASE + "v0.npy"))
    v0 = v0[asteroid_flags]

    # tested breaking this into chunks and running in serial instead of all particles
    # at once, no major speedup
    state = SystemState(
        tracer_positions=x0,
        tracer_velocities=v0,
        massive_positions=jnp.empty((0, 3)),
        massive_velocities=jnp.empty((0, 3)),
        log_gms=jnp.empty((0,)),
        acceleration_func_kwargs={},
        time=Time("2020-01-01").tdb.jd,
    )

    sy = System(
        state=state,
        gravity="default solar system",
        earliest_time=Time("1999-12-30"),
        latest_time=Time("2040-01-02"),
    )

    # might as well only query once, will need both for the ephemeris and phase function
    if observer_positions is None:
        if observer == "geocentric":
            observer = "500@399"
        observer_positions = get_observer_positions(times, observer)
    coords = sy.ephemeris(times=times, observer=observer_positions)
    coord_table = Table(
        [[str(i) for i in list(relevant_mpcorb["Unpacked Name"])], coords],
        names=["name", "coord"],
    )

    positions, _ = sy.integrate(times=times)

    hs = jnp.array([float(i) for i in list(relevant_mpcorb["H"])])
    gs = jnp.array([float(i) for i in list(relevant_mpcorb["G"])])

    mags = jax.vmap(
        jax.vmap(apparent_mag, in_axes=(None, None, 0, 0)), in_axes=(0, 0, 1, None)
    )(hs, gs, positions, observer_positions)
    mag_table = Table(
        [[str(i) for i in list(relevant_mpcorb["Unpacked Name"])], mags],
        names=["name", "mag"],
    )

    c_ra = coordinate.ra.rad
    c_dec = coordinate.dec.rad
    coords_ra = coords.ra.rad
    coords_dec = coords.dec.rad

    seps = jax.vmap(
        jax.vmap(sky_sep, in_axes=(None, None, 0, 0)), in_axes=(None, None, 0, 0)
    )(
        c_ra, c_dec, coords_ra, coords_dec
    )  # (n_particles, times)

    m_ref = jnp.min(mags)
    fluxes = jnp.power(10, -0.4 * (mags - m_ref))

    fluxes = jnp.where(seps < radius, fluxes, 0.0)
    fluxes = jnp.sum(fluxes, axis=0)

    total_mags = -2.5 * jnp.log10(fluxes) + m_ref

    return coords, seps, coord_table, mags, mag_table, total_mags


def get_relevant_mpcorb(asteroid_flags: jnp.ndarray) -> pl.DataFrame:
    """Filter an MPCORB file to only include relevant asteroids.

    Given a boolean array corresponding to the asteroid in the Jorbit ephemeris, filter
    the MPCORB file to only include those asteroids. Note that not all asteroids in the
    MPCORB file are in the Jorbit ephemeris since not all had Horizons data available.

    Args:
        asteroid_flags (jnp.ndarray):
            A boolean array indicating which asteroids to include.

    Returns:
        pl.DataFrame:
            The filtered MPCORB file.
    """
    all_names = jnp.load(download_file_wrapper(JORBIT_EPHEM_URL_BASE + "names.npy"))
    names = all_names[asteroid_flags]
    names = [str(n) for n in names]
    relevant_mpcorb = load_mpcorb()
    names_df = pl.DataFrame({"Packed designation": names})
    relevant_mpcorb = names_df.join(
        relevant_mpcorb, on="Packed designation", how="left"
    )
    relevant_mpcorb = relevant_mpcorb.select(
        [pl.col("Unpacked Name"), pl.exclude("Unpacked Name")]
    )
    assert len(relevant_mpcorb) == len(names)
    return relevant_mpcorb

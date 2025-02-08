import warnings

warnings.filterwarnings("ignore", module="erfa")

import jax

jax.config.update("jax_enable_x64", True)

import astropy.units as u
import jax.numpy as jnp
import numpy as np
import pandas as pd
import polars as pl
from astropy.table import Table
from astropy.time import Time
from astropy.utils.data import download_file, is_url_in_cache

from jorbit.astrometry.sky_projection import sky_sep
from jorbit.data.constants import JORBIT_EPHEM_URL_BASE


@jax.jit
def get_chunk_index(
    time, t0=Time("2020-01-01").tdb.jd, tf=Time("2040-01-01").tdb.jd, chunk_size=30
):

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
    return index, offset


@jax.jit
def eval_cheby(coefficients, x):
    b_ii = jnp.zeros(2)
    b_i = jnp.zeros(2)

    def scan_func(X, a):
        b_i, b_ii = X
        tmp = b_i
        b_i = a + 2 * x * b_i - b_ii
        b_ii = tmp
        return (b_i, b_ii), b_i

    (b_i, b_ii), s = jax.lax.scan(scan_func, (b_i, b_ii), coefficients[:-1])
    return coefficients[-1] + x * b_i - b_ii, s


@jax.jit
def individual_state(coefficients, offset, t0, chunk_size):
    intlen = chunk_size * 86400.0

    s = 2.0 * offset / intlen - 1.0

    (approx_ra, approx_dec), _ = eval_cheby(coefficients, s)
    return approx_ra % (2 * jnp.pi), approx_dec


@jax.jit
def multiple_states(coefficients, offset, t0, chunk_size):
    return jax.vmap(individual_state, in_axes=(0, None, None, None))(
        coefficients, offset, t0, chunk_size
    )


def setup_checks(coordinate, time, radius):
    assert np.all(time > Time("2020-01-01")), "All times must be after 2020-01-01"
    assert np.all(time < Time("2040-01-01")), "All times must be before 2040-01-01"
    coordinate = coordinate.transform_to("icrs")
    radius = radius.to(u.arcsec).value

    # hard-coded:
    t0 = Time("2020-01-01").tdb.jd
    tf = Time("2040-01-01").tdb.jd
    chunk_size = 30

    # get the names of all particles- this file is < 40 MB
    names = np.load(download_file(JORBIT_EPHEM_URL_BASE + "names.npy"))

    return coordinate, radius, t0, tf, chunk_size, names


def load_mpcorb():
    column_spans = {
        "Packed designation": (0, 7),
        "H": (8, 13),
        "G": (14, 19),
        "Epoch": (20, 25),
        "M": (26, 36),
        "Peri": (37, 47),
        "Node": (48, 58),
        "Incl.": (59, 69),
        "e": (70, 79),
        "n": (80, 91),
        "a": (92, 104),
        "U": (105, 106),
        "Reference": (107, 116),
        "#Obs": (117, 122),
        "#Opp": (123, 126),
        "Arc": (127, 136),
        "rms": (137, 141),
        "Coarse Perts": (142, 145),
        "Precise Perts": (146, 149),
        "Computer": (150, 160),
        "Flags": (161, 165),
        "Unpacked Name": (166, 193),
        "last obs": (194, 201),
    }

    col_names = list(column_spans.keys())
    col_widths = [end - start + 1 for start, end in column_spans.values()]

    file_path = download_file(JORBIT_EPHEM_URL_BASE + "MPCORB.DAT", cache=True)
    df = pd.read_fwf(
        file_path, widths=col_widths, names=col_names, dtype=str, skiprows=43
    )
    df = pl.DataFrame(df)
    return df


def mpchecker(coordinate, time, radius=20 * u.arcmin, chunk_coefficients=None):

    coordinate, radius, t0, tf, chunk_size, names = setup_checks(
        coordinate, time, radius
    )

    index, offset = get_chunk_index(time.tdb.jd, t0, tf, chunk_size)

    # figure out what chunk you're in
    if chunk_coefficients is None:
        file_name = JORBIT_EPHEM_URL_BASE + f"chebyshev_coeffs_fwd_{index:03d}.npy"
        if not is_url_in_cache(file_name):
            warnings.warn(
                "The requested time falls in an ephemeris chunk that is not found in "
                "astropy cache. Downloading now, file is approx. 250 MB. Be aware of "
                "system memory constraints if checking many well-separated times.",
                stacklevel=2,
            )
        file_name = download_file(file_name, cache=True)
        coefficients = jnp.load(file_name)

    # get the ra and dec of every minor planet (!)
    ras, decs = multiple_states(coefficients, offset, t0, chunk_size)

    # get the separation in arcsec
    separations = jax.vmap(sky_sep, in_axes=(0, 0, None, None))(
        ras, decs, coordinate.ra.rad, coordinate.dec.rad
    )

    # filter down to just those within the radius
    mask = separations < radius
    names = names[mask]
    ras = ras[mask]
    decs = decs[mask]
    separations = separations[mask]

    order = np.argsort(separations)
    names = names[order]
    ras = ras[order]
    decs = decs[order]
    separations = separations[order]

    t = Table(
        [names, separations, ras * u.rad.to(u.deg), decs * u.rad.to(u.deg)],
        names=["name", "separation", "ra", "dec"],
        units=[None, u.arcsec, u.deg, u.deg],
    )
    return t


def nearest_asteroid_helper(coordinate, times):

    coordinate, _, t0, tf, chunk_size, names = setup_checks(
        coordinate, times, radius=0 * u.arcsec
    )
    indices, offsets = jax.vmap(get_chunk_index, in_axes=(0, None, None, None))(
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
        chunk = jnp.load(
            download_file(JORBIT_EPHEM_URL_BASE + f"chebyshev_coeffs_fwd_{ind:03d}.npy")
        )
        coeffs.append(chunk)

    coeffs = jnp.array(coeffs)
    return (coordinate, _, t0, tf, chunk_size, names), coeffs


def nearest_asteroid(coordinate, times, precomputed=None):

    if precomputed is None:
        coordinate, _, t0, tf, chunk_size, names = setup_checks(
            coordinate, times, radius=0 * u.arcsec
        )
    else:
        coordinate, _, t0, tf, chunk_size, names = precomputed[0]

    indices, offsets = jax.vmap(get_chunk_index, in_axes=(0, None, None, None))(
        times.tdb.jd, t0, tf, chunk_size
    )
    unique_indices = jnp.unique(indices)

    if (len(unique_indices) > 2) and (precomputed is None):
        warnings.warn(
            f"Requested times span {len(unique_indices)} chunks of the jorbit "
            "ephemeris, each of which is ~250MB. Although only one of these will be "
            "loaded into memory at a time, beware that all will be downloaded "
            "and cached. ",
            stacklevel=2,
        )
    if precomputed is not None:
        coefficients = precomputed[1]
        assert len(coefficients) == len(unique_indices), (
            "The number of ephemeris chunk coefficients provided does not match the "
            "number of unique chunks implied by the requested times."
        )
    else:
        coefficients = None

    separations = np.zeros(len(times))
    for i, ind in enumerate(unique_indices):
        if coefficients is None:
            chunk_coefficients = jnp.load(
                download_file(
                    JORBIT_EPHEM_URL_BASE + f"chebyshev_coeffs_fwd_{ind:03d}.npy"
                )
            )
        else:
            chunk_coefficients = coefficients[i]

        # do an initial calculation of *all* asteroids
        mid_ind = (len(offsets[indices == ind]) - 1) // 2
        offset = offsets[indices == ind][mid_ind]
        ras, decs = multiple_states(chunk_coefficients, offset, t0, chunk_size)
        seps = jax.vmap(sky_sep, in_axes=(0, 0, None, None))(
            ras, decs, coordinate.ra.rad, coordinate.dec.rad
        )
        mask = seps < 108000.0  # 30 degrees
        smol_coefficients = chunk_coefficients[mask]

        def scan_func(carry, scan_over):
            coeffs = carry
            offset = scan_over
            ras, decs = multiple_states(coeffs, offset, t0, chunk_size)
            seps = jax.vmap(sky_sep, in_axes=(0, 0, None, None))(
                ras, decs, coordinate.ra.rad, coordinate.dec.rad
            )
            return coeffs, jnp.min(seps)

        _, seps = jax.lax.scan(scan_func, smol_coefficients, offsets[indices == ind])

        separations[indices == ind] = seps

    return separations

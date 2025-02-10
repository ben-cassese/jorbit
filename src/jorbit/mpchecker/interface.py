import warnings

warnings.filterwarnings("ignore", module="erfa")

import jax

jax.config.update("jax_enable_x64", True)

import astropy.units as u
import jax.numpy as jnp
import numpy as np
import polars as pl
from astropy.table import Table, hstack
from astropy.utils.data import download_file, is_url_in_cache

from jorbit.astrometry.sky_projection import sky_sep
from jorbit.data.constants import JORBIT_EPHEM_URL_BASE
from jorbit.mpchecker.parse_jorbit_ephem import (
    get_chunk_index,
    load_mpcorb,
    multiple_states,
    setup_checks,
    unpacked_to_packed_designation,
)


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

    x = [unpacked_to_packed_designation(i) for i in names]
    mpcorb = load_mpcorb()
    reference_df = pl.DataFrame({"Packed designation": x}).with_row_index("ordering")
    mpcorb = (
        mpcorb.join(reference_df, on="Packed designation", how="inner")
        .sort("ordering")
        .drop("ordering")
    )
    names = mpcorb["Unpacked Name"].to_numpy()
    mpcorb = Table.from_pandas(mpcorb.drop("Unpacked Name").to_pandas())

    t = Table(
        [names, separations, ras * u.rad.to(u.deg), decs * u.rad.to(u.deg)],
        names=["name", "separation", "ra", "dec"],
        units=[None, u.arcsec, u.deg, u.deg],
    )
    t = hstack([t, mpcorb])
    return t


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
                    JORBIT_EPHEM_URL_BASE + f"chebyshev_coeffs_fwd_{ind:03d}.npy",
                    cache=True,
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

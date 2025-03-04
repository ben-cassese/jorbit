"""The public interface to the mpchecker functions."""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", module="erfa")

import jax

jax.config.update("jax_enable_x64", True)

import astropy.units as u
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, hstack
from astropy.time import Time
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

from jorbit.astrometry.sky_projection import sky_sep, tangent_plane_projection
from jorbit.data.constants import JORBIT_EPHEM_URL_BASE
from jorbit.mpchecker.parse_jorbit_ephem import (
    extra_precision_calcs,
    get_chunk_index,
    get_relevant_mpcorb,
    multiple_states,
    setup_checks,
)
from jorbit.utils.cache import download_file_wrapper


def mpchecker(
    coordinate: SkyCoord,
    time: Time,
    radius: u.Quantity = 20 * u.arcmin,
    extra_precision: bool = False,
    observer: str = "geocentric",
    extra_precision_gravity: str | callable = "newtonian solar system",
    chunk_coefficients: jnp.ndarray | None = None,
) -> Table:
    """Find the minor planets within a given radius of a coordinate at a given time.

    This is a local implementation of the MPC's 'mpchecker' service. It uses a cached
    integration that began with particle states taken from JPL Horizons and were
    evolved using Jorbit and Newtonian gravity (but all major solar system perturbers).
    Instead of saving the cartesian positions, we saved the on-sky coordinates of each
    particle at each time step as seen by a geocentric observer. Between this assumption
    and the neglect of GR effects, the cached positions are only accurate to within
    ~an arcsec. However, if using "extra_precision", this will first run the
    quick/coarse search, figure out which minor planets fell within the radius, then
    actually run an N-body integration using Jorbit to get their positions as seen from
    a specific observer. These positions should agree with Horizons to ~1 mas if using
    extra_precision_gravity="default solar system".

    Args:
        coordinate (SkyCoord):
            The coordinate to search around.
        time (Time):
            The time to search at.
        radius (u.Quantity):
            The radius to search within. Note that for the coarse search, speed
            shouldn't depend too strongly on radius, but the speed of the extra
            precision search will depend on the number of particles that need to be
            integrated. Must be a unit of angle (e.g. u.arcsec, u.deg, etc.).
        extra_precision (bool):
            Whether to run the extra precision search. This will be slower, but more
            accurate. Will be "true" if the observer is not geocentric.
        observer (str):
            The observatory from which the observations are made. Can be a string name
            or Horizons-style @399 code.
        extra_precision_gravity (str | callable):
            The gravity model to use for the extra precision search. Must be a valid
            argument for "gravity" in System. Default is "newtonian solar system".
        chunk_coefficients (jnp.ndarray | None):
            Optionally pass the relevant chunk coefficients to avoid I/O operations if
            running this repeatedly.

    Returns:
        Table:
            A table of the minor planets within the given radius of the coordinate at
            the given time. The exact columns depend on whether extra precision was
            requested.
    """
    if observer != "geocentric":
        extra_precision = True

    coordinate, radius, t0, tf, chunk_size, names = setup_checks(
        coordinate, time, radius
    )

    index, offset = get_chunk_index(time.tdb.jd, t0, tf, chunk_size)

    # figure out what chunk you're in
    if chunk_coefficients is None:
        if jnp.sign(index) == -1:
            file_name = JORBIT_EPHEM_URL_BASE + f"chebyshev_coeffs_rev_{-index:03d}.npy"
        else:
            file_name = JORBIT_EPHEM_URL_BASE + f"chebyshev_coeffs_fwd_{index:03d}.npy"
        file_name = download_file_wrapper(file_name)
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

    relevant_mpcorb = get_relevant_mpcorb(mask)

    if extra_precision:
        coords, seps, coord_table, mags, mag_table, total_mags = extra_precision_calcs(
            asteroid_flags=mask,
            times=time,
            radius=radius,
            observer=observer,
            coordinate=coordinate,
            gravity=extra_precision_gravity,
            relevant_mpcorb=relevant_mpcorb,
        )
        separations = seps[:, 0]
        ras = coords[:, 0].ra.rad
        decs = coords[:, 0].dec.rad

        t = Table(
            [separations, ras * u.rad.to(u.deg), decs * u.rad.to(u.deg), mags],
            names=["separation", "ra", "dec", "est. Vmag"],
            units=[u.arcsec, u.deg, u.deg, u.mag],
        )

    else:
        t = Table(
            [separations, ras * u.rad.to(u.deg), decs * u.rad.to(u.deg)],
            names=["separation", "ra", "dec"],
            units=[u.arcsec, u.deg, u.deg],
        )

    relevant_mpcorb = Table.from_pandas(relevant_mpcorb.to_pandas())
    t = hstack([t, relevant_mpcorb])
    col_order = ["Unpacked Name"] + [
        col for col in t.colnames if col != "Unpacked Name"
    ]
    t = t[col_order]
    t.sort("separation")
    return t


def nearest_asteroid(
    coordinate: SkyCoord,
    times: Time,
    precomputed: tuple | None = None,
    radius: u.Quantity = 2 * u.arcmin,
    compute_contamination: bool = False,
    observer: str = "geocentric",
    extra_precision_gravity: str | callable = "newtonian solar system",
) -> tuple:
    """Identify minor planets passing through a region of the sky at a series of times.

    This is a more dynamic version of the mpchecker function that's designed to find
    the nearest minor planet to a given coordinate at a series of times. If one wants to
    do a quick check to see if any minor planets got close to a given coordinate over a
    series of times, they can leave compute_contamination as False. This will return a
    limited amount of information about the nearest minor planet at each time, but will
    not compute their magnitudes or re-integrate their orbits for higher precision. If
    compute_contamination is True, it will first run the coarse search, then
    re-integrate the orbits of all minor planets that fell within the search radius at
    any time, compute their on-sky coordinates and Vmags as seen from a specific
    observer, and produce more detailed tables of the results.

    Args:
        coordinate (SkyCoord):
            The coordinate to search around.
        times (Time):
            The times to search at. Can handle arbitrary length Times, but speed will
            depend on the total length.
        precomputed (tuple | None):
            Optionally pass the relevant chunk coefficients to avoid I/O operations if
            running this repeatedly.
        radius (u.Quantity):
            The radius to search within when computing total magntiude/flagging
            individual asteroids. Must be a unit of angle (e.g. u.arcsec, u.deg, etc.).
        compute_contamination (bool):
            Whether to compute the total magntiude of all asteroids within the search
            radius at each time. Uses the same formula as Horizons for converting H and
            G to Vmags, and each asteroids' individual H and G values from the
            MPCORB.DAT table that the cached ephemeris was built from.
        observer (str):
            The observatory from which the observations are made. Can be a string name
            or Horizons-style @399 code.
        extra_precision_gravity (str | callable):
            The gravity model to use for the extra precision search. Must be a valid
            argument for "gravity" in System. Default is "newtonian solar system".

    Returns:
        tuple:
            If compute_contamination is False, returns a tuple of the distance to the
            nearest minor planet at each time and a table of the minor planets that
            fell within the search radius at any time. If compute_contamination is True,
            returns a tuple of the distance to the nearest minor planet at each time, a
            table of the minor planets that fell within the search radius at any time,
            a table of the coordinates of all minor planets within the search radius at
            each time, a table of the Vmags of all minor planets within the search
            radius at each time, and a table of the total Vmags of all minor planets
            within the search radius at each time.
    """
    radius = radius.to(u.arcsec).value
    if times.shape == ():
        times = Time([times])

    if (times[-1] - times[0]) > (30 * u.day):
        warnings.warn(
            "The requested time span is greater than 30 days. Long time spans can "
            "result in missing rapid minor planets, since we only consider object that "
            "fell within 30 degrees of the reference point at any the midpoint.",
            stacklevel=2,
        )

    if precomputed is None:
        coordinate, _, t0, tf, chunk_size, names = setup_checks(
            coordinate, times, radius=0 * u.arcsec
        )
        observer_positions = None
    else:
        coordinate, _, t0, tf, chunk_size, names = precomputed[0]
        observer_positions = precomputed[2]

    indices, offsets = jax.vmap(get_chunk_index, in_axes=(0, 0, 0, None))(
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
        asteroid_flags = np.zeros(len(coefficients[0]), dtype=bool)
    else:
        # load the first chunk to get the number of asteroids
        if jnp.sign(unique_indices[0]) == -1:
            tmp = jnp.load(
                download_file_wrapper(
                    JORBIT_EPHEM_URL_BASE
                    + f"chebyshev_coeffs_rev_{-unique_indices[0]:03d}.npy",
                )
            )
        else:
            tmp = jnp.load(
                download_file_wrapper(
                    JORBIT_EPHEM_URL_BASE
                    + f"chebyshev_coeffs_fwd_{unique_indices[0]:03d}.npy",
                )
            )
        asteroid_flags = np.zeros(len(tmp), dtype=bool)
        coefficients = None
    separations = np.zeros(len(times))
    for i, ind in enumerate(unique_indices):
        if coefficients is None:
            if jnp.sign(ind) == -1:
                file_name = (
                    JORBIT_EPHEM_URL_BASE + f"chebyshev_coeffs_rev_{-ind:03d}.npy"
                )
            else:
                file_name = (
                    JORBIT_EPHEM_URL_BASE + f"chebyshev_coeffs_fwd_{ind:03d}.npy"
                )
            chunk_coefficients = jnp.load(download_file_wrapper(file_name))
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

        def scan_func(carry: tuple, scan_over: float) -> tuple:
            coeffs, flags = carry
            offset = scan_over
            ras, decs = multiple_states(coeffs, offset, t0, chunk_size)
            seps = jax.vmap(sky_sep, in_axes=(0, 0, None, None))(
                ras, decs, coordinate.ra.rad, coordinate.dec.rad
            )
            flags = jnp.where(seps < radius, True, flags)
            return (coeffs, flags), jnp.min(seps)

        (_, flags), seps = jax.lax.scan(
            scan_func,
            (smol_coefficients, jnp.zeros(len(smol_coefficients), dtype=bool)),
            offsets[indices == ind],
        )

        separations[indices == ind] = seps
        asteroid_flags[mask] = jnp.where(flags, True, asteroid_flags[mask])

    relevant_mpcorb = get_relevant_mpcorb(asteroid_flags)
    relevant_mpcorb = Table.from_pandas(relevant_mpcorb.to_pandas())

    if not compute_contamination:
        return separations * u.arcsec, relevant_mpcorb

    coords, seps, coord_table, mags, mag_table, total_mags = extra_precision_calcs(
        asteroid_flags=asteroid_flags,
        times=times,
        radius=radius,
        observer=observer,
        coordinate=coordinate,
        relevant_mpcorb=relevant_mpcorb,
        gravity=extra_precision_gravity,
        observer_positions=observer_positions,
    )

    return separations * u.arcsec, relevant_mpcorb, coord_table, mag_table, total_mags


def animate_region(
    coordinate: SkyCoord,
    times: Time,
    coord_table: Table,
    radius: u.Quantity,
    frame_interval: int = 50,
) -> FuncAnimation:
    """Animate the results of nearest_asteroid.

    Args:
        coordinate (SkyCoord):
            The coordinate to search around.
        times (Time):
            The times to search at.
        coord_table (Table):
            The table of minor planets that fell within the search radius at any time.
            Computed via nearest_asteroid.
        radius (u.Quantity):
            The radius to search within when computing total magntiude/flagging
            individual asteroids. Must be a unit of angle (e.g. u.arcsec, u.deg, etc.).
        frame_interval (int):
            The interval between frames in milliseconds.

    Returns:
        FuncAnimation:
            The animation of the minor planets passing through the region of the sky.
            If running in a Jupyter notebook, should render in the cell. Can be saved as
            a gif via animate_region(...).save('animation.gif', writer='pillow').
    """
    radius = radius.to(u.arcsec).value
    tmp = jax.vmap(
        jax.vmap(tangent_plane_projection, in_axes=(None, None, 0, 0)),
        in_axes=(None, None, 0, 0),
    )(
        coordinate.ra.rad,
        coordinate.dec.rad,
        coord_table["coord"].ra.rad,
        coord_table["coord"].dec.rad,
    )
    xs, ys = tmp[..., 0], tmp[..., 1]
    xs = xs.T
    ys = ys.T

    particle_names = [str(i) for i in list(coord_table["name"])]

    fig, ax = plt.subplots(figsize=(6, 6))

    # Add the reference circle
    circle = Circle((0, 0), radius, fill=False, linestyle="--", color="gray")
    ax.add_patch(circle)
    ax.scatter(0, 0, s=100, c="k", marker="x", label="Reference point")

    title = ax.set_title("")

    # Initialize scatter plot
    scatter = ax.scatter(xs[0], ys[0], s=np.ones(xs.shape[1]) * 100)

    # Initialize text annotations if names are provided
    texts = []
    texts = [
        ax.text(
            xs[0][i],
            ys[0][i],
            name,
            horizontalalignment="center",
            verticalalignment="bottom",
            animated=True,
            fontsize=8,
            color="k",
            clip_on=True,
        )
        for i, name in enumerate(particle_names)
    ]

    ax.set(
        xlim=(-radius * 4, radius * 4),
        ylim=(-radius * 4, radius * 4),
        xlabel=r"$\Delta$ RA [arcsec]",
        ylabel=r"$\Delta$ Dec [arcsec]",
        aspect="equal",
    )
    ax.grid(True)
    ax.autoscale(False)
    plt.tight_layout()

    def update(frame: int) -> tuple:
        scatter.set_offsets(np.c_[xs[frame], ys[frame]])
        title.set_text(times[frame].iso)
        for i, text in enumerate(texts):
            text.set_position((xs[frame][i], ys[frame][i] + radius * 0.15))
        return (scatter, *texts)

    n_frames = len(xs)
    anim = FuncAnimation(
        fig, update, frames=n_frames, interval=frame_interval, blit=True
    )

    plt.close()

    try:
        get_ipython()
        display(HTML(anim.to_jshtml()))
    except NameError:
        pass

    return anim

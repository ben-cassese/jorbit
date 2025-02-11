import warnings

warnings.filterwarnings("ignore", module="erfa")

import jax

jax.config.update("jax_enable_x64", True)

import astropy.units as u
import jax.numpy as jnp
import numpy as np
import polars as pl
from astropy.table import Table
from astropy.time import Time
from astropy.utils.data import download_file

from jorbit import System
from jorbit.astrometry.sky_projection import sky_sep
from jorbit.data.constants import JORBIT_EPHEM_URL_BASE
from jorbit.utils.horizons import get_observer_positions
from jorbit.utils.states import SystemState


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
    names = np.load(download_file(JORBIT_EPHEM_URL_BASE + "names.npy", cache=True))

    return coordinate, radius, t0, tf, chunk_size, names


def load_mpcorb():
    df = pl.read_ipc(download_file(JORBIT_EPHEM_URL_BASE + "mpcorb.arrow", cache=True))
    return df


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
            download_file(
                JORBIT_EPHEM_URL_BASE + f"chebyshev_coeffs_fwd_{ind:03d}.npy",
                cache=True,
            )
        )
        coeffs.append(chunk)

    coeffs = jnp.array(coeffs)
    return (coordinate, _, t0, tf, chunk_size, names), coeffs


def unpacked_to_packed_designation(number_str):
    # If it's a provisional designation (7 characters), return as is
    if len(number_str) == 7:
        return number_str

    # Convert to integer for numerical comparisons
    num = int(number_str)

    # Low numbers (purely numeric) - return as is
    if num < 100000:
        return number_str

    # Medium numbers (10000-619999) - convert to letter + 4 digits
    if num < 620000:
        # Calculate the letter prefix and remaining digits
        prefix_num = num // 10000
        remaining = num % 10000

        # Convert prefix number to letter (matching the original letter_to_number function)
        if prefix_num >= 36:  # a-z for 36+
            prefix = chr(ord("a") + (prefix_num - 36))
        else:  # A-Z for 10-35
            prefix = chr(ord("A") + (prefix_num - 10))

        # Format the remaining digits with leading zeros
        return f"{prefix}{remaining:04d}"

    # High numbers (620000+) - convert to tilde + base62
    def decimal_to_base62(n):
        """Convert decimal number to base62 string."""
        chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        if n == 0:
            return "0"

        result = ""
        while n > 0:
            n, remainder = divmod(n, 62)
            result = chars[remainder] + result
        return result

    # Subtract the offset and convert to base62
    base62_num = decimal_to_base62(num - 620000)
    # Pad to ensure total length of 5 characters (including the tilde)
    return f"~{base62_num:0>4}"


def packed_to_unpacked_designation(code):

    # if it's a provisional designation, just return it
    if len(code) == 7:
        return code

    # if it's a numbered object, it could be written 3 forms:

    # low numbered objects are just numbers
    if code.isdigit():
        return code

    # medium-numbered objects are a letter followed by 4 digits
    def letter_to_number(char):
        if char.isupper():
            return ord(char) - ord("A") + 10
        else:
            return ord(char) - ord("a") + 36

    if code[0].isalpha() and code[1:].isdigit():
        prefix_value = letter_to_number(code[0])
        num = (prefix_value * 10000) + int(code[1:])
        return str(num)

    # high-numbered objects are a tilde followed by a base-62 number
    def base62_to_decimal(char):
        if char.isdigit():
            return int(char)
        elif char.isupper():
            return ord(char) - ord("A") + 10
        else:
            return ord(char) - ord("a") + 36

    if code.startswith("~"):
        # Convert each character to its decimal value and calculate total
        total = 0
        for position, char in enumerate(reversed(code[1:])):
            decimal_value = base62_to_decimal(char)
            total += decimal_value * (62**position)
        num = total + 620000
        return str(num)

    raise ValueError(f"Invalid MPC code format: {code}")


@jax.jit
def apparent_mag(h, g, target_position, observer_position):
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
    asteroid_flags, times, radius, observer, coordinate, relevant_mpcorb
):
    x0 = jnp.load(download_file(JORBIT_EPHEM_URL_BASE + "x0.npy", cache=True))
    x0 = x0[asteroid_flags]
    v0 = jnp.load(download_file(JORBIT_EPHEM_URL_BASE + "v0.npy", cache=True))
    v0 = v0[asteroid_flags]

    if observer == "geocentric":
        observer = "500@399"

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


def get_relevant_mpcorb(asteroid_flags):
    all_names = jnp.load(download_file(JORBIT_EPHEM_URL_BASE + "names.npy", cache=True))
    names = all_names[asteroid_flags]
    names = [unpacked_to_packed_designation(i) for i in names]
    names = [str(n) for n in names]
    relevant_mpcorb = load_mpcorb()
    relevant_mpcorb = relevant_mpcorb.filter(pl.col("Packed designation").is_in(names))
    relevant_mpcorb = relevant_mpcorb.select(
        [pl.col("Unpacked Name"), pl.exclude("Unpacked Name")]
    )
    assert len(relevant_mpcorb) == len(names)
    return relevant_mpcorb

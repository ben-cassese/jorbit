import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: I001
import sqlite3
import sys

import astropy.units as u
import numpy as np
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from numpy.polynomial import chebyshev
from tqdm import tqdm

from jorbit import Particle


def generate_ephem(particle_name, chunk_size, degree):
    # chunk size in days
    # print(f"beginning for {particle_name}")
    obj = Horizons(
        id=particle_name, location="500@0", epochs=t0.tdb.jd, id_type="smallbody"
    )
    vecs = obj.vectors(refplane="earth")
    # print("horizons vectors acquired")
    x0 = jnp.array([vecs["x"], vecs["y"], vecs["z"]]).T[0]
    v0 = jnp.array([vecs["vx"], vecs["vy"], vecs["vz"]]).T[0]
    # print("creating particle")
    particle = Particle(x=x0, v=v0, time=t0, gravity="newtonian solar system")

    t = forward_times.tdb.jd

    # print("generating ephemeris")
    eph = particle.ephemeris(t, observer=forward_pos)

    # print("forming coefficients")
    r = jnp.unwrap(eph.ra.rad)
    d = eph.dec.rad

    num_chunks = int(jnp.ceil((t[-1] - t[0]) / chunk_size))

    init = (t[0] - 2451545.0) * 86400.0
    intlen = chunk_size * 86400.0

    coeffs = jnp.zeros((degree + 1, 2, num_chunks))
    for i in range(num_chunks):
        inds = (t >= t[0] + i * chunk_size) & (t < t[0] + (i + 1) * chunk_size)
        t_chunk = t[inds]
        r_chunk = r[inds]
        d_chunk = d[inds]

        # Scale time to [-1, 1] domain
        t_min, t_max = t0.tdb.jd + i * chunk_size, t0.tdb.jd + (i + 1) * chunk_size
        t_scaled = 2 * (t_chunk - t_min) / (t_max - t_min) - 1

        # Fit Chebyshev polynomials
        coefficients = chebyshev.chebfit(t_scaled, r_chunk, degree)
        coefficients = coefficients[::-1]
        coeffs = coeffs.at[:, 0, i].set(coefficients)

        coefficients = chebyshev.chebfit(t_scaled, d_chunk, degree)
        coefficients = coefficients[::-1]
        coeffs = coeffs.at[:, 1, i].set(coefficients)

    return (init, intlen, coeffs), x0, v0


# def test_ephem(particle_name, ephem, t):
#     obj = Horizons(id=particle_name, location="500@399", epochs=t.utc.jd, id_type="smallbody")
#     coord = obj.ephemerides(quantities=1, extra_precision=True)
#     s = SkyCoord(coord["RA"], coord["DEC"], unit=(u.deg, u.deg))

#     ra, dec = _individual_state(*ephem, t.tdb.jd)

#     return sky_sep(ra, dec, s.ra.rad, s.dec.rad)


def mpc_code_to_number(code):

    def base62_to_decimal(char):
        """
        Convert a single base-62 character to its decimal value.
        Base-62 uses:
        - 0-9 for values 0-9
        - A-Z for values 10-35
        - a-z for values 36-61
        """
        if char.isdigit():
            return int(char)
        elif char.isupper():
            return ord(char) - ord("A") + 10
        else:
            return ord(char) - ord("a") + 36

    # if it's unnumbered, just return the code
    if len(code) == 7:
        return code

    # if it's numbered, now it should only be 5 characters long
    assert len(code) == 5

    # low numbers are just numbers
    if "~" not in code:
        return code

    # higher ones start with a tilde and are base 62
    assert code[0] == "~"

    base62_part = code[1:]

    total = 0
    for position, char in enumerate(reversed(base62_part)):
        decimal_value = base62_to_decimal(char)
        total += decimal_value * (62**position)

    # Add back the offset of 620,000
    final_number = total + 620000

    return final_number


def setup_database():
    with sqlite3.connect("ephem_results.db") as conn:
        conn.execute("PRAGMA journal_mode=WAL")

        # Create our table if it doesn't exist
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                target_name TEXT PRIMARY KEY,
                chebyshev_coefficients BLOB,
                x0 BLOB,
                v0 BLOB
            )
        """
        )


def adapt_array(arr):
    """Convert numpy array to binary for SQLite storage"""
    return arr.tobytes()


def convert_array(blob):
    """Convert binary blob back to numpy array"""
    return np.frombuffer(blob)


def write_result(target_name, chebyshev_coefficients, x0, v0):
    with sqlite3.connect("ephem_results.db") as conn:
        # Convert the numpy array to binary
        cheby_binary = adapt_array(chebyshev_coefficients)
        x0_binary = adapt_array(x0)
        v0_binary = adapt_array(v0)

        # Insert the result
        conn.execute(
            "INSERT OR REPLACE INTO results (target_name, chebyshev_coefficients, x0, v0) VALUES (?, ?, ?, ?)",
            (target_name, cheby_binary, x0_binary, v0_binary),
        )


def contribute_to_ephem(line_start, line_stop):
    with open("MPCORB.DAT") as f:
        lines = f.readlines()[line_start : line_stop + 1]

    targets = [line.split()[0] for line in lines]
    targets = [mpc_code_to_number(target) for target in targets]

    # the asteroids that we use as perturbers are included in MPCORB.DAT
    # if we try to integrate them the accelerations will be huge, and the step sizes
    # will be so small they'll never finish
    forbidden_targets = [
        "00001",
        "00002",
        "00003",
        "00004",
        "00007",
        "00010",
        "00015",
        "00016",
        "00031",
        "00052",
        "00065",
        "00087",
        "00088",
        "00107",
        "00511",
        "00704",
    ]
    targets = [target for target in targets if target not in forbidden_targets]

    print(
        f"Processing {len(targets)} targets between line_start={line_start} and line_stop={line_stop}"
    )
    setup_database()

    for target in tqdm(targets):
        (_, _, coeffs), x0, v0 = generate_ephem(
            particle_name=target, chunk_size=30, degree=10
        )
        write_result(target, coeffs, x0, v0)

    return targets


line_start, line_stop = int(sys.argv[1]), int(sys.argv[2])

t0 = Time("2020-01-01")
forward_times = t0 + jnp.arange(0, 20.001, 10 * u.hour.to(u.year)) * u.year
reverse_times = t0 - jnp.arange(0, 20.001, 10 * u.hour.to(u.year)) * u.year

forward_pos = jnp.load("forward_pos.npy")
reverse_pos = jnp.load("reverse_pos.npy")

contribute_to_ephem(line_start, line_stop)

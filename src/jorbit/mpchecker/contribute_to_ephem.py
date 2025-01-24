import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: I001
import time
import os
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
    print(f"beginning for {particle_name}")
    for _i in range(5):
        try:
            obj = Horizons(
                id=particle_name,
                location="500@0",
                epochs=t0.tdb.jd,
                id_type="smallbody",
            )
            vecs = obj.vectors(refplane="earth")
            break
        except Exception as e:
            print(f"error getting vectors for {particle_name}, retrying")
            if _i == 4:
                print(f"failed to get vectors for {particle_name}")
                raise e
            time.sleep(2 + np.random.uniform(0, 1))
            pass
    print("horizons vectors acquired")
    x0 = jnp.array([vecs["x"], vecs["y"], vecs["z"]]).T[0]
    v0 = jnp.array([vecs["vx"], vecs["vy"], vecs["vz"]]).T[0]
    print("creating particle")
    particle = Particle(x=x0, v=v0, time=t0, gravity="newtonian solar system")

    t = forward_times.tdb.jd

    print("generating ephemeris")
    eph = particle.ephemeris(t, observer=forward_pos)

    print("forming coefficients")
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

    print("done")
    return (init, intlen, coeffs), x0, v0


def mpc_code_to_number(code):

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


def adapt_array(arr):
    """Convert numpy array to binary for SQLite storage"""
    return arr.tobytes()


def convert_array(blob):
    """Convert binary blob back to numpy array"""
    return np.frombuffer(blob)


def write_result(target_name, chebyshev_coefficients, x0, v0):

    with sqlite3.connect(TEMP_DB, timeout=30.0) as conn:
        # Create the table if it doesn't exist
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS results
            (target_name TEXT PRIMARY KEY,
             chebyshev_coefficients BLOB,
             x0 BLOB,
             v0 BLOB)
        """
        )

        # Convert arrays to binary
        cheby_binary = adapt_array(chebyshev_coefficients)
        x0_binary = adapt_array(x0)
        v0_binary = adapt_array(v0)

        # Insert into temporary database
        conn.execute(
            "INSERT OR REPLACE INTO results VALUES (?, ?, ?, ?)",
            (target_name, cheby_binary, x0_binary, v0_binary),
        )


def result_exists(target_name):
    with sqlite3.connect(TEMP_DB) as conn:
        cursor = conn.execute(
            "SELECT 1 FROM results WHERE target_name = ?", (target_name,)
        )
        return cursor.fetchone() is not None


def setup_db():
    with sqlite3.connect(TEMP_DB, timeout=30.0) as conn:
        # Create the table if it doesn't exist
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS results
            (target_name TEXT PRIMARY KEY,
             chebyshev_coefficients BLOB,
             x0 BLOB,
             v0 BLOB)
        """
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

    for target in tqdm(targets):
        if result_exists(target):
            print(f"Skipping target {target} because it already exists in the database")
            continue
        try:
            (_, _, coeffs), x0, v0 = generate_ephem(
                particle_name=target, chunk_size=30, degree=10
            )
            print("writing result to database\n")
            write_result(target, coeffs, x0, v0)
        except Exception as e:
            print(f"Error processing target {target}: {e}")
            continue

    return targets


line_start, line_stop = int(sys.argv[1]), int(sys.argv[2])

# this is silly, but all of the jobs writing to the same database simultaneously
# made it infeasibly slow. so, write each job to its own database and then merge them
# probably don't need to check for existing records anymore, kinda a bummer if
# restarting the array
print("setting up database")
arr_id = os.environ.get("SLURM_ARRAY_TASK_ID", "ARRAY_ID_NOT_FOUND")
job_id = os.environ.get("SLURM_JOB_ID", "JOB_ID_NOT_FOUND")
if arr_id == "ARRAY_ID_NOT_FOUND" or job_id == "JOB_ID_NOT_FOUND":
    raise ValueError("SLURM environment variables not found")

TEMP_DB = f"db_results/temp_results_{arr_id}_{job_id}.db"

setup_db()

print("reading in times/positions")
t0 = Time("2020-01-01")
forward_times = t0 + jnp.arange(0, 20.001, 10 * u.hour.to(u.year)) * u.year

forward_pos = jnp.load("forward_pos.npy")

print("beginning integrations")
contribute_to_ephem(line_start, line_stop)

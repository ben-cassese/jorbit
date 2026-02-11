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
from astropy.utils.data import download_file
from astroquery.jplhorizons import Horizons
from numpy.polynomial import chebyshev
from tqdm import tqdm

from jorbit import Particle
from jorbit.utils.horizons import horizons_bulk_vector_query
from jorbit.utils.mpc import packed_to_unpacked_designation
from jorbit.data.constants import JORBIT_EPHEM_URL_BASE, PERTURBER_PACKED_DESIGNATIONS


def generate_ephem(target_index, chunk_size, degree):
    print(f"beginning for {target_index}")

    particle = Particle(
        x=x0_full[target_index],
        v=v0_full[target_index],
        time=t0,
        gravity="newtonian solar system",
    )

    t = reverse_times.tdb.jd

    print("generating ephemeris")
    eph = particle.ephemeris(t, observer=reverse_pos)
    eph = eph[::-1]
    t = t[::-1]

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
        t_min, t_max = t[0] + i * chunk_size, t[0] + (i + 1) * chunk_size
        t_scaled = 2 * (t_chunk - t_min) / (t_max - t_min) - 1

        # Fit Chebyshev polynomials
        coefficients = chebyshev.chebfit(t_scaled, r_chunk, degree)
        coefficients = coefficients[::-1]
        coeffs = coeffs.at[:, 0, i].set(coefficients)

        coefficients = chebyshev.chebfit(t_scaled, d_chunk, degree)
        coefficients = coefficients[::-1]
        coeffs = coeffs.at[:, 1, i].set(coefficients)

    print("done")
    return (init, intlen, coeffs), x0_full[target_index], v0_full[target_index]


def adapt_array(arr):
    """Convert numpy array to binary for SQLite storage"""
    return arr.tobytes()


def convert_array(blob):
    """Convert binary blob back to numpy array"""
    return np.frombuffer(blob)


def write_result(target_name, chebyshev_coefficients, x0, v0):

    with sqlite3.connect(TEMP_DB, timeout=30.0) as conn:
        # Create the table if it doesn't exist
        conn.execute("""
            CREATE TABLE IF NOT EXISTS results
            (target_name TEXT PRIMARY KEY,
             chebyshev_coefficients BLOB,
             x0 BLOB,
             v0 BLOB)
        """)

        # Convert arrays to binary
        cheby_binary = adapt_array(chebyshev_coefficients)
        x0_binary = adapt_array(x0)
        v0_binary = adapt_array(v0)

        # Insert into temporary database
        conn.execute(
            "INSERT OR REPLACE INTO results VALUES (?, ?, ?, ?)",
            (target_name, cheby_binary, x0_binary, v0_binary),
        )


def setup_db():
    with sqlite3.connect(TEMP_DB, timeout=30.0) as conn:
        # Create the table if it doesn't exist
        conn.execute("""
            CREATE TABLE IF NOT EXISTS results
            (target_name TEXT PRIMARY KEY,
             chebyshev_coefficients BLOB,
             x0 BLOB,
             v0 BLOB)
        """)


def contribute_to_ephem(target_indecies):

    print(f"Processing {len(target_indecies)} targets")

    for i, target_index in tqdm(enumerate(target_indecies)):
        try:
            target = str(names[target_index])

            if target in PERTURBER_PACKED_DESIGNATIONS:
                print(f"{target} is a perturber, skipping")
                continue

            (_, _, coeffs), x0, v0 = generate_ephem(
                target_index=target_index, chunk_size=30, degree=10
            )
            print("writing result to database\n")
            write_result(target, coeffs, x0, v0)
        except Exception as e:
            print(f"Error processing target {target_index}: {e}")
            continue

    return target_indecies


target_indecies = [int(i) for i in sys.argv[1:]]

print("loading data")
t0 = Time("2020-01-01")
reverse_times = t0 - jnp.arange(0, 20.001, 10 * u.hour.to(u.year)) * u.year
names = jnp.load(download_file(JORBIT_EPHEM_URL_BASE + "names.npy"))
x0_full = jnp.load(download_file(JORBIT_EPHEM_URL_BASE + "x0.npy"))
v0_full = jnp.load(download_file(JORBIT_EPHEM_URL_BASE + "v0.npy"))
reverse_pos = jnp.load(
    download_file(JORBIT_EPHEM_URL_BASE + "reverse_pos.npy", cache=True)
)

print("setting up database")
arr_id = os.environ.get("SLURM_ARRAY_TASK_ID", "ARRAY_ID_NOT_FOUND")
job_id = os.environ.get("SLURM_JOB_ID", "JOB_ID_NOT_FOUND")
if arr_id == "ARRAY_ID_NOT_FOUND" or job_id == "JOB_ID_NOT_FOUND":
    raise ValueError("SLURM environment variables not found")

TEMP_DB = f"db_results/temp_results_rev_{arr_id}_{job_id}.db"

setup_db()

print("beginning integrations")
contribute_to_ephem(target_indecies=target_indecies)

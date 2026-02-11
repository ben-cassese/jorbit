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
from jorbit.utils.mpc import packed_to_unpacked_designation


def generate_ephem(particle_name, chunk_size, degree):
    # chunk size in days
    print(f"beginning for {particle_name}")
    for _i in range(20):
        try:
            vecs = horizons_bulk_vector_query(
                target=particle_name,
                center="500@0",
                times=t0,
                disable_astroquery=True,
            )
            break

        except Exception as e:
            if _i == 0:
                print(
                    f"error getting vectors for {particle_name}, going to use astroquery"
                )
                try:
                    p = packed_to_unpacked_designation(particle_name)
                    obj = Horizons(
                        id=p,
                        location="500@0",
                        epochs=t0.tdb.jd,
                        id_type="smallbody",
                    )
                    vecs = obj.vectors(refplane="earth")
                except ValueError as e:
                    if ("Unknown target" in str(e)) or (
                        "Horizons Error: No ephemeris for target" in str(e)
                    ):
                        print(f"target {particle_name} is not in Horizons")
                        file_name = TEMP_DB.replace(".db", "_not_in_horizons.txt")
                        with open(file_name, "a") as f:
                            f.write(f"{particle_name}\n")
                    raise

            print(f"error getting vectors for {particle_name}, retrying")
            if _i == 19:
                print(f"failed to get vectors for {particle_name}\n*****\n\n")
                raise e
            time.sleep(2 * _i + np.random.uniform(0, 10))
            pass
    print("horizons vectors acquired")
    x0 = jnp.array([vecs["x"], vecs["y"], vecs["z"]]).T[0]
    v0 = jnp.array([vecs["vx"], vecs["vy"], vecs["vz"]]).T[0]

    # since we're running this for every sso, it's trying to cache way too many files
    # in our home directory for the cluster to be happy with
    try:  # noqa: SIM105
        Horizons.clear_cache()
    except Exception:
        pass

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


def contribute_to_ephem(targets):

    print(f"Processing {len(targets)}")

    for target in tqdm(targets):
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


targets = sys.argv[1:]

print("setting up database")
arr_id = os.environ.get("SLURM_ARRAY_TASK_ID", "ARRAY_ID_NOT_FOUND")
job_id = os.environ.get("SLURM_JOB_ID", "JOB_ID_NOT_FOUND")
if arr_id == "ARRAY_ID_NOT_FOUND" or job_id == "JOB_ID_NOT_FOUND":
    raise ValueError("SLURM environment variables not found")

TEMP_DB = f"db_results/FINAL_temp_results_{arr_id}_{job_id}.db"

setup_db()

print("reading in times/positions")
t0 = Time("2020-01-01")
forward_times = t0 + jnp.arange(0, 20.001, 10 * u.hour.to(u.year)) * u.year

forward_pos = jnp.load("forward_pos.npy")

print("beginning integrations")
contribute_to_ephem(targets=targets)

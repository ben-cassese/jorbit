import jax

jax.config.update("jax_enable_x64", True)
import sqlite3

import astropy.units as u
import jax.numpy as jnp
import numpy as np
import requests
from astropy.coordinates import SkyCoord
from astropy.time import Time
from numpy.polynomial import chebyshev
from tqdm import tqdm

from jorbit import Ephemeris
from jorbit.accelerations import create_newtonian_ephemeris_acceleration_func
from jorbit.astrometry.sky_projection import on_sky
from jorbit.utils.horizons import get_observer_positions

##########
# get the positions of the geocenter from 20 years ago to 20 years in the future
##########

t0 = Time("2020-01-01")
forward_times = t0 + jnp.arange(0, 20.001, 10 * u.hour.to(u.year)) * u.year
reverse_times = t0 - jnp.arange(0, 20.001, 10 * u.hour.to(u.year)) * u.year

chunk_size = 10_000
forward_pos = []
for i in tqdm(range(int(len(forward_times) / chunk_size) + 1)):
    start = i * chunk_size
    end = (i + 1) * chunk_size
    if end > len(forward_times):
        end = len(forward_times)
    forward_pos.append(get_observer_positions(forward_times[start:end], "500@399"))

reverse_pos = []
for i in tqdm(range(int(len(reverse_times) / chunk_size) + 1)):
    start = i * chunk_size
    end = (i + 1) * chunk_size
    if end > len(reverse_times):
        end = len(reverse_times)
    reverse_pos.append(get_observer_positions(reverse_times[start:end], "500@399"))

forward_pos = jnp.concatenate(forward_pos, axis=0)
reverse_pos = jnp.concatenate(reverse_pos, axis=0)

np.save("forward_pos.npy", forward_pos)
np.save("reverse_pos.npy", reverse_pos)

##########
# get the current mpcorb.dat file
##########

response = requests.get(
    "https://www.minorplanetcenter.net/iau/MPCORB/MPCORB.DAT", stream=True
)
total_size = int(response.headers.get("content-length", 0))
progress_bar = tqdm(
    total=total_size, unit="B", unit_scale=True, unit_divisor=1024, desc="Downloading"
)

with open("MPCORB.DAT", "wb") as f:
    chunk_size = 1024 * 1024  # 1 MiB chunks
    for chunk in response.iter_content(chunk_size=chunk_size):
        f.write(chunk)
        progress_bar.update(len(chunk))  # Update progress bar by chunk size

    progress_bar.close()


##########
# we aren't going to actually integrate the particles we've included as perturbers
# in the integration, but we still want them included in the final list of ssos
# so, add them in manually here
##########

t0 = Time("2020-01-01")
forward_times = t0 + jnp.arange(0, 20.001, 10 * u.hour.to(u.year)) * u.year

t = forward_times.tdb.jd
forward_pos = jnp.load("forward_pos.npy")


on_sky_eph = Ephemeris(
    ssos="default planets",
    earliest_time=Time("2019-01-01"),
    latest_time=Time("2041-01-01"),
)
acc_func = create_newtonian_ephemeris_acceleration_func(on_sky_eph.processor)


perturber_eph = Ephemeris(
    ssos="default solar system",
    earliest_time=Time("2019-01-01"),
    latest_time=Time("2041-01-01"),
)

perturbers = {
    "ceres": "00001",
    "pallas": "00002",
    "juno": "00003",
    "vesta": "00004",
    "iris": "00007",
    "hygiea": "00010",
    "eunomia": "00015",
    "psyche": "00016",
    "euphrosyne": "00031",
    "europa": "00052",
    "cybele": "00065",
    "sylvia": "00087",
    "thisbe": "00088",
    "camilla": "00107",
    "davida": "00511",
    "interamnia": "00704",
}


def write_result(target_name, chebyshev_coefficients, x0, v0):
    with sqlite3.connect("perturbers.db", timeout=30.0) as conn:
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
        cheby_binary = chebyshev_coefficients.tobytes()
        x0_binary = x0.tobytes()
        v0_binary = v0.tobytes()

        # Insert into temporary database
        conn.execute(
            "INSERT OR REPLACE INTO results VALUES (?, ?, ?, ?)",
            (target_name, cheby_binary, x0_binary, v0_binary),
        )


def get_coeffs_for_perturber(target, chunk_size, degree):
    # get the index of this specific target
    d = perturber_eph.state(forward_times[0])
    d_fast = perturber_eph.processor.state(forward_times[0].tdb.jd)[0]
    ind = int(jnp.argwhere(jnp.all((d[target]["x"].value - d_fast) == 0, axis=1))[0, 0])

    # get the x, v positions of the target
    def scan_func(carry, scan_over):
        time = scan_over
        state = perturber_eph.processor.state(time)
        x = state[0][ind]
        v = state[1][ind]
        return None, (x, v)

    _, (x, v) = jax.lax.scan(scan_func, None, forward_times.tdb.jd)

    # convert those positions to on-sky coordinates
    def scan_func(carry, scan_over):
        position, velocity, time, observer_position = scan_over
        ra, dec = on_sky(position, velocity, time, observer_position, acc_func)
        return None, (ra, dec)

    _, (ras, decs) = jax.lax.scan(
        scan_func,
        None,
        (x, v, forward_times.tdb.jd, forward_pos),
    )
    eph = SkyCoord(ra=ras, dec=decs, unit=u.rad)

    # from here on out it's the same as contribute_to_ephem- should probably break
    # this out/refactor
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

    x0 = x[0]
    v0 = v[0]
    return (init, intlen, coeffs), x0, v0


for target in tqdm(perturbers):
    (_, _, coeffs), x0, v0 = get_coeffs_for_perturber(target, 30, 10)
    write_result(perturbers[target], coeffs, x0, v0)

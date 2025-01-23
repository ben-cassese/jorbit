import jax

jax.config.update("jax_enable_x64", True)
import astropy.units as u
import jax.numpy as jnp
import numpy as np
import requests
from astropy.time import Time
from tqdm import tqdm

from jorbit.utils.horizons import get_observer_positions

# get the positions of the geocenter from 20 years ago to 20 years in the future
t0 = Time("2020-01-01")
forward_times = t0 + jnp.arange(0, 20.001, 1 * u.hour.to(u.year)) * u.year
reverse_times = t0 - jnp.arange(0, 20.001, 1 * u.hour.to(u.year)) * u.year

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


# get the currect mpcorb.dat file
response = requests.get(
    "https://www.minorplanetcenter.net/iau/MPCORB/MPCORB.DAT", stream=True
)
total_size = int(response.headers.get("content-length", 0))
progress_bar = tqdm(
    total=total_size, unit="B", unit_scale=True, unit_divisor=1024, desc="Downloading"
)

with open("MPCORB.DAT", "wb") as f:
    chunk_size = 1024 * 1024  # 1 MB chunks
    for chunk in response.iter_content(chunk_size=chunk_size):
        f.write(chunk)
        progress_bar.update(len(chunk))  # Update progress bar by chunk size

    progress_bar.close()

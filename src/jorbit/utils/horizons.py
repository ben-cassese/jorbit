import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import warnings

warnings.filterwarnings("ignore", module="erfa")
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord, ICRS

import requests
import pandas as pd
import io
import os
from tqdm import tqdm


from jorbit.data.observatory_codes import observatory_codes


def horizons_bulk_vector_query(target, center, times):
    if type(times) == type(Time("2023-01-01")):
        times = times.tdb.jd
    if isinstance(times, float):
        times = [times]
    assert (
        len(times) < 10_000
    ), "Horizons batch api can only accept less than 10,000 timesteps at a time"

    def construct_horizons_query(target, center, times):
        query = io.StringIO()
        query.write("!$$SOF\n")
        query.write(f'COMMAND= "{target}"\n')
        query.write("OBJ_DATA='NO'\n")
        query.write("MAKE_EPHEM='YES'\n")
        query.write("TABLE_TYPE='VECTOR'\n")
        query.write(f"CENTER='{center}'\n")
        query.write("REF_PLANE='FRAME'\n")
        query.write("CSV_FORMAT='YES'\n")
        query.write("OUT_UNITS='AU-D'\n")
        query.write("CAL_FORMAT='JD'\n")
        query.write("TLIST=\n")
        for t in times:
            query.write(f"'{t}'\n")
        query.seek(0)
        return query

    try:
        # Construct the query and get an in-memory file object
        query_file = construct_horizons_query(target, center, times)

        # Use the in-memory file with requests
        url = "https://ssd.jpl.nasa.gov/api/horizons_file.api"
        r = requests.post(url, data={"format": "text"}, files={"input": query_file})
        l = r.text.split("\n")
        start = l.index("$$SOE")
        end = l.index("$$EOE")
        data = pd.read_csv(
            io.StringIO("\n".join(l[start + 1 : end])),
            header=None,
            names=[
                "JDTDB",
                "Cal",
                "x",
                "y",
                "z",
                "vx",
                "vy",
                "vz",
                "LT",
                "RG",
                "RR",
                "_",
            ],
        )
        query_file.close()
        return data
    except:
        try:
            query_file.close()
        except:
            pass
        raise ValueError("Vectors query failed, check inputs.")


def horizons_bulk_astrometry_query(target, center, times, skip_daylight=False):
    def construct_horizons_query(target, center, times):
        with open("horizons_query.txt", "w") as f:
            f.write("!$$SOF\n")
            f.write(f'COMMAND= "{target}"\n')
            f.write("OBJ_DATA='NO'")
            f.write("MAKE_EPHEM='YES'\n")
            f.write("TABLE_TYPE='OBSERVER'\n")
            f.write(f"CENTER='{center}'\n")
            f.write("REF_PLANE='FRAME'\n")
            f.write("CSV_FORMAT='YES'\n")
            f.write("OUT_UNITS='AU-D'\n")
            f.write("QUANTITIES='1,36,37'\n")
            if skip_daylight:
                f.write("SKIP_DAYLT = 'YES'\n")
            f.write("ANG_FORMAT='DEG'\n")
            f.write("EXTRA_PREC = 'YES'\n")
            f.write("CAL_FORMAT='JD'\n")
            f.write("TLIST=\n")
            for t in times:
                f.write(f"'{t.jd}'\n")
            return "horizons_query.txt"

    try:
        query = construct_horizons_query(target=target, center=center, times=times)
        with open(query) as f:
            url = "https://ssd.jpl.nasa.gov/api/horizons_file.api"
            r = requests.post(url, data={"format": "text"}, files={"input": f})
            l = r.text.split("\n")
        start = l.index("$$SOE")
        end = l.index("$$EOE")

        cleaned = []
        for line in l[start + 1 : end]:
            if "Daylight Cut-off Requested" in line:
                continue
            elif line == "":
                continue
            cleaned.append(line)
        cleaned

        data = pd.read_csv(
            io.StringIO("\n".join(cleaned)),
            header=None,
            names=[
                "Cal",
                "twilight_flag",
                "moon_flag",
                "RA",
                "Dec",
                "RA 3sig",
                "DEC 3sig",
                "SMAA_3sig",
                "SMIA_3sig",
                "Theta",
                "Area_3sig",
                "_",
            ],
        )
        os.remove("horizons_query.txt")

        if len(data) != len(times):
            raise ValueError(
                "Some requested times were skipped, check skip_daylight flag"
            )
        return data
    except:
        try:
            os.remove("horizons_query.txt")
        except:
            pass
        raise ValueError("Astrometry query failed, check inputs.")


def get_observer_positions(times, observatory_codes, verbose):
    assert len(times) == len(observatory_codes)

    if verbose:
        print("Downloading observer positions from Horizons...")
    emb_from_ssb = horizons_bulk_vector_query("3", "500@0", times)
    emb_from_ssb = jnp.array(emb_from_ssb[["x", "y", "z"]].values)

    if len(set(observatory_codes)) == 1:
        emb_from_observer = horizons_bulk_vector_query("3", observatory_codes[0], times)
        emb_from_observer = jnp.array(emb_from_observer[["x", "y", "z"]].values)

    else:
        # maybe switch this back to astroquery- equally slow with lots of requests,
        # but astroquery will at least cache results
        emb_from_observer = jnp.zeros((len(times), 3))
        if verbose:
            iter = enumerate(tqdm(times))
        else:
            iter = enumerate(times)
        for i, t in iter:
            emb_from_observer = horizons_bulk_vector_query("3", observatory_codes[i], t)
            emb_from_observer = jnp.array(emb_from_observer[["x", "y", "z"]].values)

    postions = emb_from_ssb - emb_from_observer
    return postions

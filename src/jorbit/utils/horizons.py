import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import warnings

warnings.filterwarnings("ignore", module="erfa")

from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord, ICRS
from astroquery.jplhorizons import Horizons

import requests
import pandas as pd
import io
import os
from tqdm import tqdm
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Union, List


from jorbit.data.observatory_codes import observatory_codes


@dataclass
class HorizonsQueryConfig:
    """Configuration for Horizons API queries."""

    HORIZONS_API_URL = "https://ssd.jpl.nasa.gov/api/horizons_file.api"
    MAX_TIMESTEPS = 10_000

    VECTOR_COLUMNS = [
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
    ]

    ASTROMETRY_COLUMNS = [
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
    ]


def horizons_query_string(
    target: str, center: str, query_type: str, times: Time
) -> str:

    if isinstance(times.jd, float):
        times = [times]

    if len(times) > HorizonsQueryConfig.MAX_TIMESTEPS:
        raise ValueError(
            f"Horizons batch API can only accept less than {HorizonsQueryConfig.MAX_TIMESTEPS} timesteps"
        )

    lines = [
        "!$$SOF",
        f'COMMAND= "{target}"',
        "OBJ_DATA='NO'",
        "MAKE_EPHEM='YES'",
        f"CENTER='{center}'",
        "REF_PLANE='FRAME'",
        "CSV_FORMAT='YES'",
        "OUT_UNITS='AU-D'",
        "CAL_FORMAT='JD'",
        "TLIST_TYPE='JD'",
    ]

    if query_type == "VECTOR":
        lines.append("TABLE_TYPE='VECTOR'")
    elif query_type == "OBSERVER":
        lines.extend(
            [
                "TABLE_TYPE='OBSERVER'",
                "QUANTITIES='1,36,37'",
                "ANG_FORMAT='DEG'",
                "EXTRA_PREC = 'YES'",
            ]
        )
        if skip_daylight:
            lines.append("SKIP_DAYLT = 'YES'")

    lines.append("TLIST=")
    for t in times:
        if query_type == "VECTOR":
            time_value = t.tdb.jd if isinstance(t, Time) else t
        elif query_type == "OBSERVER":
            time_value = t.utc.jd if isinstance(t, Time) else t
        lines.append(f"'{time_value}'")

    query = "\n".join(lines)
    return query


@contextmanager
def horizons_query_context(query_string: str) -> io.StringIO:
    """Creates and manages the query content in memory."""
    query = io.StringIO(query_string)
    try:
        yield query
    finally:
        query.close()


def parse_horizons_response(
    response_text: str, columns: List[str], skip_empty: bool = False
) -> pd.DataFrame:
    """Parses the Horizons API response into a DataFrame."""
    lines = response_text.split("\n")
    try:
        start = lines.index("$$SOE")
        end = lines.index("$$EOE")

        if skip_empty:
            cleaned = [
                line
                for line in lines[start + 1 : end]
                if line and "Daylight Cut-off Requested" not in line
            ]
        else:
            cleaned = lines[start + 1 : end]

        df = pd.read_csv(io.StringIO("\n".join(cleaned)), header=None, names=columns)
        df = df.drop(columns="_")
        return df
    except ValueError as e:
        raise ValueError("Failed to parse Horizons response: invalid format") from e


def make_horizons_request(query_content: io.StringIO) -> str:
    """Makes the HTTP request to Horizons API."""
    try:
        response = requests.post(
            HorizonsQueryConfig.HORIZONS_API_URL,
            data={"format": "text"},
            files={"input": query_content},
        )
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        raise ValueError(f"Failed to query Horizons API: {str(e)}")


def horizons_bulk_vector_query(
    target: str,
    center: str,
    times: Time,
) -> pd.DataFrame:

    if isinstance(times.jd, float):
        times = [times]
    if len(times) < 25:
        horizons_obj = Horizons(
            id=target, location=center, epochs=[t.tdb.jd for t in times]
        )
        vec_table = horizons_obj.vectors(refplane="earth")
        vec_table = vec_table[
            [
                "datetime_jd",
                "x",
                "y",
                "z",
                "vx",
                "vy",
                "vz",
                "lighttime",
                "range",
                "range_rate",
            ]
        ].to_pandas()
        vec_table.rename(
            columns={
                "datetime_jd": "JDTDB",
                "lighttime": "LT",
                "range": "RG",
                "range_rate": "RR",
            },
            inplace=True,
        )
        return vec_table

    try:
        # Build query
        query = horizons_query_string(target, center, "VECTOR", times)

        # Execute query
        with horizons_query_context(query) as query_content:
            response_text = make_horizons_request(query_content)
            return parse_horizons_response(
                response_text, HorizonsQueryConfig.VECTOR_COLUMNS
            )

    except Exception as e:
        raise ValueError(f"Vector query failed: {str(e)}")


# def horizons_bulk_astrometry_query(
#     target: str, center: str, times: Time, skip_daylight: bool = False
# ) -> pd.DataFrame:
#     """
#     Query the JPL Horizons system for astrometric data of a celestial body.

#     Args:
#         target: The target body identifier
#         center: The center body identifier
#         times: List of Time objects for the query
#         skip_daylight: Whether to skip daylight observations

#     Returns:
#         pd.DataFrame: DataFrame containing the astrometric data
#     """

#     try:
#         # Build query
#         query = horizons_query_string(target, center, "OBSERVER", times)

#         # Execute query using StringIO
#         with horizons_query_context(query) as query_content:
#             response_text = make_horizons_request(query_content)
#             data = parse_horizons_response(
#                 response_text, HorizonsQueryConfig.ASTROMETRY_COLUMNS, skip_empty=True
#             )

#         if len(data) != len(times):
#             raise ValueError(
#                 "Some requested times were skipped, check skip_daylight flag"
#             )

#         return data

#     except Exception as e:
#         raise ValueError(f"Astrometry query failed: {str(e)}")


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
        for obs in set(observatory_codes):
            idxs = [i for i, x in enumerate(observatory_codes) if x == obs]
            _emb_from_observer = horizons_bulk_vector_query("3", obs, times[idxs])
            _emb_from_observer = jnp.array(_emb_from_observer[["x", "y", "z"]].values)
            if obs == observatory_codes[0]:
                emb_from_observer_all = _emb_from_observer
            else:
                emb_from_observer_all = jnp.concatenate(
                    [emb_from_observer_all, _emb_from_observer]
                )
        emb_from_observer = jnp.array(emb_from_observer_all)[jnp.argsort(times)]

    postions = emb_from_ssb - emb_from_observer
    return postions


def old_horizons_bulk_vector_query(target, center, times):
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

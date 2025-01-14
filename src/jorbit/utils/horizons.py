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


from jorbit.data.observatory_codes import OBSERVATORY_CODES


@dataclass
class HorizonsQueryConfig:
    """Configuration for Horizons API queries."""

    HORIZONS_API_URL = "https://ssd.jpl.nasa.gov/api/horizons_file.api"
    # hard limit from the Horizons api
    MAX_TIMESTEPS = 10_000
    # kinda arbitrary, have gotten it to work with ~50 but seems like it can be finicky
    ASTROQUERY_MAX_TIMESTEPS = 25

    VECTOR_COLUMNS = [
        "JD_TDB",
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
        "JD_UTC",
        "twilight_flag",
        "moon_flag",
        "RA",
        "DEC",
        "RA_3sigma",
        "DEC_3sigma",
        "SMAA_3sigma",
        "SMIA_3sigma",
        "Theta_3sigma",
        "Area_3sigma",
        "_",
    ]


def horizons_query_string(
    target: str, center: str, query_type: str, times: Time, skip_daylight: bool = False
) -> str:

    assert len(times) > HorizonsQueryConfig.ASTROQUERY_MAX_TIMESTEPS

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
        if "twilight_flag" in df.columns:
            df = df.drop(columns="twilight_flag")
        if "moon_flag" in df.columns:
            df = df.drop(columns="moon_flag")
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
    if len(times) < HorizonsQueryConfig.ASTROQUERY_MAX_TIMESTEPS:
        # note that astrometry queries use utc, vector use tdb...
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


def horizons_bulk_astrometry_query(
    target: str, center: str, times: Time, skip_daylight: bool = False
) -> pd.DataFrame:

    if isinstance(times.jd, float):
        times = [times]
    if len(times) < HorizonsQueryConfig.ASTROQUERY_MAX_TIMESTEPS:
        # note that astrometry queries use utc, vector use tdb...
        horizons_obj = Horizons(
            id=target, location=center, epochs=[t.utc.jd for t in times]
        )
        horizons_table = horizons_obj.ephemerides(
            quantities="1,36,37", extra_precision=True
        )
        horizons_table = horizons_table[
            [
                "datetime_jd",
                "RA",
                "DEC",
                "RA_3sigma",
                "DEC_3sigma",
                "SMAA_3sigma",
                "SMIA_3sigma",
                "Theta_3sigma",
                "Area_3sigma",
            ]
        ].to_pandas()
        horizons_table.rename(
            columns={
                "datetime_jd": "JD_UTC",
            },
            inplace=True,
        )
        return horizons_table

    try:
        # Build query
        query = horizons_query_string(
            target, center, "OBSERVER", times, skip_daylight=skip_daylight
        )

        # Execute query using StringIO
        with horizons_query_context(query) as query_content:
            response_text = make_horizons_request(query_content)
            data = parse_horizons_response(
                response_text, HorizonsQueryConfig.ASTROMETRY_COLUMNS, skip_empty=True
            )

        return data

    except Exception as e:
        raise ValueError(f"Astrometry query failed: {str(e)}")


def get_observer_positions(times, observatories):
    if isinstance(times.jd, float):
        times = [times]
    if isinstance(observatories, str):
        observatories = [observatories]
    # allow either a single observatory, or a list of observatories
    # w/ the same length as times
    if len(observatories) == 1:
        observatories = observatories * len(times)
    assert len(times) == len(observatories)
    # just to standardize:
    # the vector/astrometry queries automatically convert to utc/tdb as appropriate
    times = Time([t.utc.jd for t in times], format="jd", scale="utc")

    emb_from_ssb = horizons_bulk_vector_query("3", "500@0", times)
    emb_from_ssb = jnp.array(emb_from_ssb[["x", "y", "z"]].values)

    _times = []
    emb_from_observer_all = jnp.empty((0, 3))
    for obs in set(observatories):
        idxs = [i for i, x in enumerate(observatories) if x == obs]
        if "@" not in obs:
            if obs.lower() in OBSERVATORY_CODES.keys():
                obs = OBSERVATORY_CODES[obs.lower()]
            else:
                raise ValueError(
                    "Observer location '{}' is not a recognized observatory. Please"
                    " refer to"
                    " https://minorplanetcenter.net/iau/lists/ObsCodesF.html".format(
                        obs
                    )
                )

        _emb_from_observer = horizons_bulk_vector_query("3", obs, times[idxs])
        _emb_from_observer = jnp.array(_emb_from_observer[["x", "y", "z"]].values)

        emb_from_observer_all = jnp.concatenate(
            [emb_from_observer_all, _emb_from_observer]
        )
        _times.extend(times[idxs])
    _times = jnp.array([t.tdb.jd for t in _times])
    emb_from_observer = jnp.array(emb_from_observer_all)[jnp.argsort(_times)]

    postions = emb_from_ssb - emb_from_observer
    return postions

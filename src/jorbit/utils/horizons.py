"""Functions for interacting with the Horizons API."""

import jax

jax.config.update("jax_enable_x64", True)

import io
from contextlib import contextmanager
from typing import NamedTuple

import jax.numpy as jnp
import pandas as pd
import requests
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from jorbit.data.observatory_codes import OBSERVATORY_CODES


class HorizonsQueryConfig(NamedTuple):
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
    """Constructs the query string for the Horizons API.

    Args:
        target (str):
            The target object identifier.
        center (str):
            The center object identifier.
        query_type (str):
            The type of query, either 'VECTOR' or 'OBSERVER'.
        times (Time):
            The times for the query. Note it just needs to be an Astropy Time object-
            we'll handle the different tdb/utc conversions internally.
        skip_daylight (bool):
            Whether to skip daylight in the query.

    Returns:
        str:
            The constructed query string.
    """
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
    response_text: str, columns: list[str], skip_empty: bool = False
) -> pd.DataFrame:
    """Parses the Horizons API response into a DataFrame.

    Args:
        response_text (str):
            The response text from the Horizons API.
        columns (list[str]):
            The column names for the DataFrame.
        skip_empty (bool):
            Whether to skip empty lines in the response.

    Returns:
        pd.DataFrame:
            The parsed DataFrame.

    """
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
    """Makes the HTTP request to Horizons API.

    Args:
        query_content (io.StringIO):
            The query content to send in the request.

    Returns:
        str:
            The response text from the Horizons API.

    Raises:
        ValueError:
            If the request fails.
    """
    try:
        response = requests.post(
            HorizonsQueryConfig.HORIZONS_API_URL,
            data={"format": "text"},
            files={"input": query_content},
        )
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        raise ValueError(f"Failed to query Horizons API: {e!s}") from e


def horizons_bulk_vector_query(
    target: str,
    center: str,
    times: Time,
) -> pd.DataFrame:
    """The main query function for our retrieval of vectors from the Horizons API.

    This function creates the appropriate query string, executes the query, and parses
    the response into a DataFrame. If we're requesting a small number of timesteps,
    it'll use astroquery to retrieve the data, which allows for easy caching. However,
    if we're requesting > 50 unique timesteps, astroquery will fail, so we instead fall
    back to a manual API query. These results will not be cached.

    Note: The Horizons API has a hard limit of 10,000 timesteps per query.

    Args:
        target (str):
            The target object identifier.
        center (str):
            The center object identifier.
        times (Time):
            The times for the query. Note it just needs to be an Astropy Time object-
            we'll handle the different tdb/utc conversions internally.

    Returns:
        pd.DataFrame:
            A pandas DataFrame containing the vector data.
    """
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
        raise ValueError(f"Vector query failed: {e!s}") from e


def horizons_bulk_astrometry_query(
    target: str, center: str, times: Time, skip_daylight: bool = False
) -> pd.DataFrame:
    """The main query function for our retrieval of astrometry from the Horizons API.

    This function creates the appropriate query string, executes the query, and parses
    the response into a DataFrame. If we're requesting a small number of timesteps,
    it'll use astroquery to retrieve the data, which allows for easy caching. However,
    if we're requesting > 50 unique timesteps, astroquery will fail, so we instead fall
    back to a manual API query. These results will not be cached.

    Note: The Horizons API has a hard limit of 10,000 timesteps per query.

    Args:
        target (str):
            The target object identifier.
        center (str):
            The center object identifier.
        times (Time):
            The times for the query. Note it just needs to be an Astropy Time object-
            we'll handle the different tdb/utc conversions internally.
        skip_daylight (bool):
            Whether to skip daylight in the query.

    Returns:
        pd.DataFrame:
            A pandas DataFrame containing the astrometry data.
    """
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
        raise ValueError(f"Astrometry query failed: {e!s}") from e


def get_observer_positions(times: Time, observatories: str | list[str]) -> jnp.ndarray:
    """A wrapper to retrieve the barycentric positions of an observer from the Horizons API.

    Args:
        times (Time):
            The times for the query. Can be a single Time, or a Time object with length
            > 1. If length < 50, positions will be retrieved using astroquery, otherwise
            a manual API query will be used. Note that length must be < 10,000.
        observatories (str | list[str]):
            The observatory name for the query. If '@' is included in the query, it's
            assumed to be a Horizons-interpretable code. Otherwise it will be compared
            to the list of observatory names in the jorbit.data.observatory_codes module
            and mapped to its appropriate code.

    Returns:
        jnp.ndarray:
            The barycentric positions of the observer in AU.
    """
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
            if obs.lower() in OBSERVATORY_CODES:
                obs = OBSERVATORY_CODES[obs.lower()]
            else:
                raise ValueError(
                    f"Observer location '{obs}' is not a recognized observatory. Please"
                    " refer to"
                    " https://minorplanetcenter.net/iau/lists/ObsCodesF.html"
                )

        _emb_from_observer = horizons_bulk_vector_query("3", obs, times[idxs])
        _emb_from_observer = jnp.array(_emb_from_observer[["x", "y", "z"]].values)

        emb_from_observer_all = jnp.concatenate(
            [emb_from_observer_all, _emb_from_observer]
        )
        _times.extend(times[idxs])
    _times = jnp.array([t.tdb.jd for t in _times])
    emb_from_observer = jnp.array(emb_from_observer_all)[jnp.argsort(_times)]

    positions = emb_from_ssb - emb_from_observer
    return positions

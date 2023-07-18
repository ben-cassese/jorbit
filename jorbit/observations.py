import jax
from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import warnings

warnings.filterwarnings("ignore", module="erfa")
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord, ICRS
import astropy.units as u
from astroquery.jplhorizons import Horizons

import matplotlib.pyplot as plt
import pickle
import requests
import pandas as pd
import io
import os
import pkg_resources
from tqdm import tqdm

from .construct_perturbers import (
    construct_perturbers,
    STANDARD_PLANET_PARAMS,
    STANDARD_ASTEROID_PARAMS,
    STANDARD_PLANET_GMS,
    STANDARD_ASTEROID_GMS,
)

# from . import data
# codes = (resources.files(data) / 'observatory_codes.pkl')
codes = pkg_resources.resource_filename("jorbit", "data/observatory_codes.pkl")
with open(codes, "rb") as f:
    observatory_codes = pickle.load(f)

# with open("./data/observatory_codes.pkl", "rb") as f:
#     observatory_codes = pickle.load(f)


class Observations:
    def __init__(
        self,
        observed_coordinates=None,
        times=None,
        observatory_locations=None,
        astrometric_uncertainties=None,
        verbose_downloading=False,
        mpc_file=None,
    ):
        self.verbose_downloading = verbose_downloading

        if mpc_file is None:
            assert (
                (observed_coordinates is not None)
                and (times is not None)
                and (observatory_locations is not None)
                and (astrometric_uncertainties is not None)
            ), (
                "If no MPC file is provided, observed_coordinates, times,"
                " observatory_locations, and astrometric_uncertainties must be given"
                " manually."
            )
        else:
            assert (
                (observed_coordinates is None)
                and (times is None)
                and (observatory_locations is None)
                and (astrometric_uncertainties is None)
            ), (
                "If an MPC file is provided, observed_coordinates, times,"
                " observatory_locations, and astrometric_uncertainties must be None."
            )
            cols = [
                (0, 5),
                (5, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 32),
                (32, 44),
                (44, 56),
                (65, 70),
                (70, 71),
                (77, 80),
            ]

            names = [
                "Packed number",
                "Packed provisional designation",
                "Discovery asterisk",
                "Note 1",
                "Note 2",
                "Date of observation",
                "Observed RA (J2000.0)",
                "Observed Decl. (J2000.0)",
                "Observed magnitude",
                "Band",
                "Observatory code",
            ]

            data = pd.read_fwf(mpc_file, colspecs=cols, names=names)

            def parse_time(mpc_time):
                t = mpc_time.replace(" ", "-").split(".")
                return (
                    Time(t[0], format="iso", scale="utc") + float(f"0.{t[1]}") * u.day
                )

            def parse_uncertainty(dec_coord):
                if len(dec_coord.split(".")) == 1:
                    return 1 * u.arcsec
                return 10 ** (-len(dec_coord.split(".")[1])) * u.arcsec

            observed_coordinates = SkyCoord(
                data["Observed RA (J2000.0)"],
                data["Observed Decl. (J2000.0)"],
                unit=(u.hourangle, u.deg),
            )
            times = list(map(parse_time, data["Date of observation"]))
            observatory_locations = [s + "@399" for s in list(data["Observatory code"])]
            astrometric_uncertainties = list(
                map(parse_uncertainty, data["Observed Decl. (J2000.0)"])
            )

        # POSITIONS
        if isinstance(observed_coordinates, type(SkyCoord(0 * u.deg, 0 * u.deg))):
            s = observed_coordinates.transform_to(
                ICRS
            )  # in case they're barycentric, etc
            self.ra = s.ra.rad
            self.dec = s.dec.rad
        elif isinstance(observed_coordinates, list):
            ras = []
            decs = []
            for s in observed_coordinates:
                s = s.transform_to(ICRS)
                ras.append(s.ra.rad)
                decs.append(s.dec.rad)
            self.ra = jnp.array(ras)
            self.dec = jnp.array(decs)
        if self.ra.shape == ():
            self.ra = jnp.array([self.ra])
            self.dec = jnp.array([self.dec])

        # TIMES
        if isinstance(times, type(Time("2023-01-01"))):
            self.times = jnp.array(times.tdb.jd)
        elif isinstance(times, list):
            self.times = jnp.array([t.tdb.jd for t in times])
        elif isinstance(times, jnp.ndarray):
            self.times = times
        else:
            raise ValueError(
                "times must be either astropy.time.Time, list of astropy.time.Time, or"
                " jax.numpy.ndarray (interpreted as JD in TDB)"
            )
        if self.times.shape == ():
            self.times = jnp.array([self.times])

        # OBSERVER POSITIONS
        if isinstance(observatory_locations, str):
            observatory_locations = [observatory_locations] * len(self.times)

        for i, loc in enumerate(observatory_locations):
            loc = loc.lower()
            if loc in observatory_codes.keys():
                observatory_locations[i] = observatory_codes[loc]
            elif "@" in loc:
                pass
            else:
                raise ValueError(
                    "Observer location '{}' is not a recognized observatory. Please"
                    " refer to https://minorplanetcenter.net/iau/lists/ObsCodesF.html"
                    .format(loc)
                )

        self.observatory_locations = observatory_locations
        self.observer_positions = self.get_observer_positions(
            times=Time(self.times, format="jd", scale="tdb"),
            observatory_codes=observatory_locations,
        )

        # UNCERTAINTIES
        if isinstance(astrometric_uncertainties, type(u.Quantity(1 * u.arcsec))):
            self.astrometric_uncertainties = (
                jnp.ones(len(self.times)) * astrometric_uncertainties.to(u.arcsec).value
            )
        elif isinstance(astrometric_uncertainties, list):
            self.astrometric_uncertainties = jnp.array(
                [p.to(u.arcsec).value for p in astrometric_uncertainties]
            )

        # This shouldn't be possible if only dealing with SkyCoords
        assert (
            len(self.ra)
            == len(self.dec)
            == len(self.times)
            == len(self.observer_positions)
            == len(self.astrometric_uncertainties)
        ), (
            "Inputs must have the same length. Currently: ra={}, dec={}, times={},"
            " observer_positions={}, astrometric_uncertainties={}".format(
                len(self.ra),
                len(self.dec),
                len(self.times),
                len(self.observer_positions),
                len(self.astrometric_uncertainties),
            )
        )
        # assert len(self.ra) > 1, "Must include at least two observations"

    def __repr__(self):
        return f"Observations with {len(self.ra)} set(s) of observations"

    def __len__(self):
        return len(self.ra)

    def get_observer_positions(self, times, observatory_codes):
        assert len(times) == len(observatory_codes)

        if self.verbose_downloading:
            print("Downloading observer positions from Horizons...")
        emb_from_ssb = Observations.horizons_vector_query("3", "500@0", times)
        emb_from_ssb = jnp.array(emb_from_ssb[["x", "y", "z"]].values)

        if len(set(observatory_codes)) == 1:
            emb_from_observer = Observations.horizons_vector_query(
                "3", observatory_codes[0], times
            )
            emb_from_observer = jnp.array(emb_from_observer[["x", "y", "z"]].values)

        else:
            emb_from_observer = jnp.zeros((len(times), 3))
            if self.verbose_downloading:
                iter = enumerate(tqdm(times))
            else:
                iter = enumerate(times)
            for i, t in iter:
                # might as well switch back to astroquery since the batch is too slow
                # here, and at least astroquery caches the results

                emb_from_observer = Observations.horizons_vector_query(
                    "3", observatory_codes[i], t
                )
                emb_from_observer = jnp.array(emb_from_observer[["x", "y", "z"]].values)

                # horizons_query = Horizons(
                #     id="3", location=observatory_codes[i], epochs=[t.tdb.jd]
                # )
                # tmp = horizons_query.vectors(refplane="earth")
                # emb_from_observer = emb_from_observer.at[i, :].set(
                #     [
                #         tmp["x"].value.data[0],
                #         tmp["y"].value.data[0],
                #         tmp["z"].value.data[0],
                #     ]
                # )

        postions = emb_from_ssb - emb_from_observer
        return postions

    @staticmethod
    def horizons_vector_query(target, center, times):
        times = times.tdb.jd
        if isinstance(times, float):
            times = [times]
        assert (
            len(times) < 10_000
        ), "Horizons batch api can only accept less than 10,000 timesteps at a time"

        def construct_horizons_query(target, center, times):
            with open("horizons_query.txt", "w") as f:
                f.write("!$$SOF\n")
                f.write(f'COMMAND= "{target}"\n')
                f.write("OBJ_DATA='NO'")
                f.write("MAKE_EPHEM='YES'\n")
                f.write("TABLE_TYPE='VECTOR'\n")
                f.write(f"CENTER='{center}'\n")
                f.write("REF_PLANE='FRAME'\n")
                f.write("CSV_FORMAT='YES'\n")
                f.write("OUT_UNITS='AU-D'\n")
                f.write("CAL_FORMAT='JD'\n")
                f.write("TLIST=\n")
                for t in times:
                    f.write(f"'{t}'\n")
                return "horizons_query.txt"

        query = construct_horizons_query(target, center, times)
        with open(query) as f:
            url = "https://ssd.jpl.nasa.gov/api/horizons_file.api"
            r = requests.post(url, data={"format": "text"}, files={"input": f})
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
        os.remove("horizons_query.txt")
        return data

    @staticmethod
    def horizons_astrometry_query(target, center, times, skip_daylight=False):
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

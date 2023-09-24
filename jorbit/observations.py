import jax
from jax.config import config

config.update("jax_enable_x64", True)
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


from jorbit.data import (
    observatory_codes,
)


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
        self._observed_coordinates = observed_coordinates
        self._times = times
        self._observatory_locations = observatory_locations
        self._astrometric_uncertainties = astrometric_uncertainties
        self._verbose_downloading = verbose_downloading
        self._mpc_file = mpc_file

        self._input_checks()

        (
            self._ra,
            self._dec,
            self._times,
            self._observatory_locations,
            self._astrometric_uncertainties,
            self._observer_positions,
        ) = self._parse_astrometry()

        self._final_init_checks()

    def __repr__(self):
        return f"Observations with {len(self._ra)} set(s) of observations"

    def __len__(self):
        return len(self._ra)

    def __add__(self, newobs):
        old_times = [t for t in self._times]
        old_ra = [r for r in self._ra]
        old_dec = [d for d in self._dec]
        old_obs_precision = [p for p in self.astrometric_uncertainties]
        old_observer_positions = [o for o in self.observer_positions]

        new_times = [t for t in newobs.times]
        new_ra = [r for r in newobs.ra]
        new_dec = [d for d in newobs.dec]
        new_obs_precision = [p for p in newobs.astrometric_uncertainties]
        new_observer_positions = [o for o in newobs.observer_positions]

        times = []
        ras = []
        decs = []
        obs_precision = []
        observer_positions = []
        for i in range(len(old_times) + len(new_times)):
            if len(old_times) == 0:
                times.append(new_times.pop(0))
                ras.append(new_ra.pop(0))
                decs.append(new_dec.pop(0))
                obs_precision.append(new_obs_precision.pop(0))
                observer_positions.append(new_observer_positions.pop(0))
            elif len(new_times) == 0:
                times.append(old_times.pop(0))
                ras.append(old_ra.pop(0))
                decs.append(old_dec.pop(0))
                obs_precision.append(old_obs_precision.pop(0))
                observer_positions.append(old_observer_positions.pop(0))
            elif old_times[0] < new_times[0]:
                times.append(old_times.pop(0))
                ras.append(old_ra.pop(0))
                decs.append(old_dec.pop(0))
                obs_precision.append(old_obs_precision.pop(0))
                observer_positions.append(old_observer_positions.pop(0))
            else:
                times.append(new_times.pop(0))
                ras.append(new_ra.pop(0))
                decs.append(new_dec.pop(0))
                obs_precision.append(new_obs_precision.pop(0))
                observer_positions.append(new_observer_positions.pop(0))

        s = SkyCoord(ra=ras, dec=decs, unit=u.rad)
        return Observations(
            observed_coordinates=s,
            times=jnp.array(times),
            observatory_locations=jnp.array(observer_positions),
            astrometric_uncertainties=jnp.array(obs_precision),
            verbose_downloading=self._verbose_downloading,
            mpc_file=None,
        )

    ####################################################################################
    # Initialization helpers
    def _input_checks(self):
        if self._mpc_file is None:
            assert (
                (self._observed_coordinates is not None)
                and (self._times is not None)
                and (self._observatory_locations is not None)
                and (self._astrometric_uncertainties is not None)
            ), (
                "If no MPC file is provided, observed_coordinates, times,"
                " observatory_locations, and astrometric_uncertainties must be given"
                " manually."
            )
        else:
            assert (
                (self._observed_coordinates is None)
                and (self._times is None)
                and (self._observatory_locations is None)
                and (self._astrometric_uncertainties is None)
            ), (
                "If an MPC file is provided, observed_coordinates, times,"
                " observatory_locations, and astrometric_uncertainties must be None."
            )

        if (
            not isinstance(self._times, type(Time("2023-01-01")))
            and not isinstance(self._times, list)
            and not isinstance(self._times, jnp.ndarray)
        ):
            raise ValueError(
                "times must be either astropy.time.Time, list of astropy.time.Time, or"
                " jax.numpy.ndarray (interpreted as JD in TDB)"
            )

        assert (
            isinstance(self._observatory_locations, str)
            or isinstance(self._observatory_locations, list)
            or isinstance(self._observatory_locations, jnp.ndarray)
        ), (
            "observatory_locations must be either a string (interpreted as an MPC"
            " observatory code), a list of observatory codes, or a jax.numpy.ndarray"
        )
        if isinstance(self._observatory_locations, list):
            assert len(self._observatory_locations) == len(self._times), (
                "If observatory_locations is a list, it must be the same length as"
                " the number of observations."
            )
        elif isinstance(self._observatory_locations, jnp.ndarray):
            assert len(self._observatory_locations) == len(self._times), (
                "If observatory_locations is a jax.numpy.ndarray, it must be the same"
                " length as the number of observations."
            )

    def _parse_astrometry(self):
        def read_mpc_file(mpc_file):
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
            return (
                observed_coordinates,
                times,
                observatory_locations,
                astrometric_uncertainties,
            )

        if self._mpc_file is None:
            (
                observed_coordinates,
                times,
                observatory_locations,
                astrometric_uncertainties,
            ) = (
                self._observed_coordinates,
                self._times,
                self._observatory_locations,
                self._astrometric_uncertainties,
            )

        else:
            (
                observed_coordinates,
                times,
                observatory_locations,
                astrometric_uncertainties,
            ) = read_mpc_file(self._mpc_file)

        # POSITIONS
        if isinstance(observed_coordinates, type(SkyCoord(0 * u.deg, 0 * u.deg))):
            # in case they're barycentric, etc
            s = observed_coordinates.transform_to(ICRS)
            ra = s.ra.rad
            dec = s.dec.rad
        elif isinstance(observed_coordinates, list):
            ras = []
            decs = []
            for s in observed_coordinates:
                s = s.transform_to(ICRS)
                ras.append(s.ra.rad)
                decs.append(s.dec.rad)
            ra = jnp.array(ras)
            dec = jnp.array(decs)
        if ra.shape == ():
            ra = jnp.array([ra])
            dec = jnp.array([dec])

        # TIMES
        if isinstance(times, type(Time("2023-01-01"))):
            times = jnp.array(times.tdb.jd)
        elif isinstance(times, list):
            times = jnp.array([t.tdb.jd for t in times])
        if times.shape == ():
            times = jnp.array([times])

        # OBSERVER POSITIONS
        if isinstance(observatory_locations, str):
            observatory_locations = [observatory_locations] * len(times)
        if isinstance(observatory_locations, list):
            for i, loc in enumerate(observatory_locations):
                loc = loc.lower()
                if loc in observatory_codes.keys():
                    observatory_locations[i] = observatory_codes[loc]
                elif "@" in loc:
                    pass
                else:
                    raise ValueError(
                        "Observer location '{}' is not a recognized observatory. Please"
                        " refer to"
                        " https://minorplanetcenter.net/iau/lists/ObsCodesF.html"
                        .format(loc)
                    )

            observatory_locations = observatory_locations
            observer_positions = self._get_observer_positions(
                times=Time(times, format="jd", scale="tdb"),
                observatory_codes=observatory_locations,
            )
        else:
            observer_positions = observatory_locations

        # UNCERTAINTIES
        if isinstance(astrometric_uncertainties, type(u.Quantity(1 * u.arcsec))):
            astrometric_uncertainties = (
                jnp.ones(len(times)) * astrometric_uncertainties.to(u.arcsec).value
            )
        elif isinstance(astrometric_uncertainties, list):
            astrometric_uncertainties = jnp.array(
                [p.to(u.arcsec).value for p in astrometric_uncertainties]
            )

        return (
            ra,
            dec,
            times,
            observatory_locations,
            astrometric_uncertainties,
            observer_positions,
        )

    def _get_observer_positions(self, times, observatory_codes):
        assert len(times) == len(observatory_codes)

        if self._verbose_downloading:
            print("Downloading observer positions from Horizons...")
        emb_from_ssb = Observations.horizons_bulk_vector_query("3", "500@0", times)
        emb_from_ssb = jnp.array(emb_from_ssb[["x", "y", "z"]].values)

        if len(set(observatory_codes)) == 1:
            emb_from_observer = Observations.horizons_bulk_vector_query(
                "3", observatory_codes[0], times
            )
            emb_from_observer = jnp.array(emb_from_observer[["x", "y", "z"]].values)

        else:
            # maybe switch this back to astroquery- equally slow with lots of requests,
            # but astroquery will at least cache results
            emb_from_observer = jnp.zeros((len(times), 3))
            if self._verbose_downloading:
                iter = enumerate(tqdm(times))
            else:
                iter = enumerate(times)
            for i, t in iter:
                emb_from_observer = Observations.horizons_bulk_vector_query(
                    "3", observatory_codes[i], t
                )
                emb_from_observer = jnp.array(emb_from_observer[["x", "y", "z"]].values)

        postions = emb_from_ssb - emb_from_observer
        return postions

    def _final_init_checks(self):
        assert (
            len(self._ra)
            == len(self._dec)
            == len(self._times)
            == len(self.observer_positions)
            == len(self.astrometric_uncertainties)
        ), (
            "Inputs must have the same length. Currently: ra={}, dec={}, times={},"
            " observer_positions={}, astrometric_uncertainties={}".format(
                len(self._ra),
                len(self._dec),
                len(self._times),
                len(self.observer_positions),
                len(self.astrometric_uncertainties),
            )
        )

        t = self._times[0]
        for i in range(1, len(self._times)):
            assert (
                self._times[i] > t
            ), "Observations must be in ascending chronological order."
            t = self._times[i]

    ####################################################################################

    @staticmethod
    def horizons_bulk_vector_query(target, center, times):
        if type(times) == type(Time("2023-01-01")):
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

        try:
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
        except:
            try:
                os.remove("horizons_query.txt")
            except:
                pass
            raise ValueError("Vectors query failed, check inputs.")

    @staticmethod
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

    @property
    def ra(self):
        return self._ra

    @property
    def dec(self):
        return self._dec

    @property
    def times(self):
        # return Time(self._times, format='jd', scale='tdb').utc
        return self._times

    @property
    def observer_positions(self):
        return self._observer_positions

    @property
    def astrometric_uncertainties(self):
        return self._astrometric_uncertainties

    @property
    def observatory_locations(self):
        return self._observatory_locations

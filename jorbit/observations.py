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
        self, positions, times, observatory_locations, astrometry_uncertainties
    ):
        # POSITIONS
        if isinstance(positions, type(SkyCoord(0 * u.deg, 0 * u.deg))):
            s = positions.transform_to(ICRS)  # in case they're barycentric, etc
            self.ra = s.ra.rad
            self.dec = s.dec.rad
        elif isinstance(positions, list):
            ras = []
            decs = []
            for s in positions:
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
                "times must be either astropy.time.Time, list of astropy.time.Time, or jax.numpy.ndarray (interpreted as JD in TDB)"
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
                    "Observer location '{}' is not a recognized observatory. Please refer to https://minorplanetcenter.net/iau/lists/ObsCodesF.html".format(
                        loc
                    )
                )

        self.observatory_locations = observatory_locations
        self.observer_positions = self.get_observer_positions(
            times=Time(self.times, format='jd', scale='tdb'),
              observatory_codes=observatory_locations
        )

        # UNCERTAINTIES
        if isinstance(astrometry_uncertainties, type(u.Quantity(1 * u.arcsec))):
            self.astrometry_uncertainties = (
                jnp.ones(len(self.times)) * astrometry_uncertainties.to(u.arcsec).value
            )
        elif isinstance(astrometry_uncertainties, list):
            self.astrometry_uncertainties = jnp.array(
                [p.to(u.arcsec).value for p in astrometry_uncertainties]
            )

        # This shouldn't be possible if only dealing with SkyCoords
        assert (
            len(self.ra)
            == len(self.dec)
            == len(self.times)
            == len(self.observer_positions)
            == len(self.astrometry_uncertainties)
        ), "Inputs must have the same length. Currently: ra={}, dec={}, times={}, observer_positions={}, astrometry_uncertainties={}".format(
            len(self.ra),
            len(self.dec),
            len(self.times),
            len(self.observer_positions),
            len(self.astrometry_uncertainties),
        )
        # assert len(self.ra) > 1, "Must include at least two observations"

    def __repr__(self):
        return f"Observations with {len(self.ra)} set(s) of observations"

    def __len__(self):
        return len(self.ra)

    def get_observer_positions(self, times, observatory_codes):
        assert len(times) == len(observatory_codes)

        emb_from_ssb = Observations.horizons_vector_query(
            "3", "500@0", times
        )
        emb_from_ssb = jnp.array(emb_from_ssb[["x", "y", "z"]].values)

        if len(set(observatory_codes)) == 1:
            emb_from_observer = Observations.horizons_vector_query(
                "3", observatory_codes[0], times
            )
            emb_from_observer = jnp.array(emb_from_observer[["x", "y", "z"]].values)

        else:
            emb_from_observer = jnp.zeros((len(times), 3))
            for i, t in enumerate(times):
                tmp = Observations.horizons_vector_query(
                    "3", observatory_codes[i], t
                ).iloc[0]
                emb_from_observer = emb_from_observer.at[i, :].set(
                    jnp.array([tmp.x, tmp.y, tmp.z])
                )

        postions = emb_from_ssb - emb_from_observer
        return postions

    @staticmethod
    def horizons_vector_query(target, center, times):
        times = times.tdb.jd
        if isinstance(times, float):
            times = [times]
        assert len(times) < 10_000, "Horizons batch api can only accept less than 10,000 timesteps at a time"

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
                f.write('CAL_FORMAT=\'JD\'\n')
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
            with open('horizons_query.txt', 'w') as f:
                f.write('!$$SOF\n')
                f.write(f'COMMAND= \"{target}\"\n')
                f.write('OBJ_DATA=\'NO\'')
                f.write('MAKE_EPHEM=\'YES\'\n')
                f.write('TABLE_TYPE=\'OBSERVER\'\n')
                f.write(f'CENTER=\'{center}\'\n')
                f.write('REF_PLANE=\'FRAME\'\n')
                f.write('CSV_FORMAT=\'YES\'\n')
                f.write('OUT_UNITS=\'AU-D\'\n')
                f.write('QUANTITIES=\'1,36,37\'\n')
                if skip_daylight:
                    f.write('SKIP_DAYLT = \'YES\'\n')
                f.write('ANG_FORMAT=\'DEG\'\n')
                f.write('EXTRA_PREC = \'YES\'\n')
                f.write('CAL_FORMAT=\'JD\'\n')
                f.write('TLIST=\n')
                for t in times:
                    f.write(f'\'{t.jd}\'\n')
                return 'horizons_query.txt'

        query = construct_horizons_query(target=target, center=center, times=times)
        with open(query) as f:
            url = 'https://ssd.jpl.nasa.gov/api/horizons_file.api'
            r = requests.post(url, data={'format':'text'}, files={'input': f})
            l = r.text.split('\n')
        start = l.index('$$SOE')
        end = l.index('$$EOE')

        cleaned = []
        for line in l[start+1:end]:
            if 'Daylight Cut-off Requested' in line:
                continue
            elif line == '':
                continue
            cleaned.append(line)
        cleaned

        data = pd.read_csv(io.StringIO('\n'.join(cleaned)), header=None,
                    names=['Cal', 'twilight_flag', 'moon_flag', 'RA', 'Dec', 'RA 3sig', 'DEC 3sig', 
                        'SMAA_3sig', 'SMIA_3sig', 'Theta', 'Area_3sig', '_'])
        os.remove('horizons_query.txt')

        if len(data) != len(times):
            raise ValueError('Some requested times were skipped, check skip_daylight flag')
        return data

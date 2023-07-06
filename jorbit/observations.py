import jax
from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import warnings
warnings.filterwarnings('ignore', module='erfa')
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord, ICRS
import astropy.units as u
from astroquery.jplhorizons import Horizons
import matplotlib.pyplot as plt
import pickle

# import importlib.resources as resources
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
codes = pkg_resources.resource_filename('jorbit', 'data/observatory_codes.pkl')
with open(codes, "rb") as f:
    observatory_codes = pickle.load(f)

# with open("./data/observatory_codes.pkl", "rb") as f:
#     observatory_codes = pickle.load(f)


class Observations:
    def __init__(self, positions, times, observatory_locations, astrometry_uncertainties):
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

        # TIMES
        if isinstance(times, type(Time("2023-01-01"))):
            self.times = jnp.array(times.tdb.jd)
        elif isinstance(times, list):
            self.times = jnp.array([t.tdb.jd for t in times])

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
            Time(self.times, format="jd", scale="tdb"), observatory_locations
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
        assert len(self.ra) > 1, "Must include at least two observations"

        self.best_fit = None
        self.fit = None

    def get_observer_positions(self, times, observatory_codes):
        # for code in observatory_codes:
        #     assert code == '500@399', "Only observations from the geocenter are currently supported"

        # positions = []
        # for i, t in enumerate(times):
        #     # Need to work out how to pass such a large requests to Horizons
        #     # temporary solution: use the downloaded ephemeris to get the geocenter
        #     # note is different than the [(0,3)] barycenter- using 'earth' instead
        #     positions.append(jnp.array(get_body_barycentric('earth', t).xyz.to(u.au).value))

        # positions = jnp.array(positions)
        # return positions
        assert len(times) == len(observatory_codes)

        horizons_query = Horizons(id="3", location="500@0", epochs=list(times))
        emb_from_ssb = horizons_query.vectors(refplane="earth")
        emb_from_ssb = jnp.array(
            [
                emb_from_ssb["x"].value.data,
                emb_from_ssb["y"].value.data,
                emb_from_ssb["z"].value.data,
            ]
        ).T

        if len(set(observatory_codes)) == 1:
            horizons_query = Horizons(
                id="3", location=observatory_codes[0], epochs=list(times)
            )
            emb_from_observer = horizons_query.vectors(refplane="earth")
            emb_from_observer = jnp.array(
                [
                    emb_from_observer["x"].value.data,
                    emb_from_observer["y"].value.data,
                    emb_from_observer["z"].value.data,
                ]
            ).T

        else:
            emb_from_observer = jnp.zeros((len(times), 3))
            for i, t in enumerate(times):
                horizons_query = Horizons(
                    id="3", location=observatory_codes[i], epochs=[t]
                )
                tmp = horizons_query.vectors(refplane="earth")
                emb_from_observer = emb_from_observer.at[i, :].set(
                    [
                        tmp["x"].value.data[0],
                        tmp["y"].value.data[0],
                        tmp["z"].value.data[0],
                    ]
                )

        postions = emb_from_ssb - emb_from_observer
        return postions

    def compute_best_fit(self):
        pass

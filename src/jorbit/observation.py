import jax

jax.config.update("jax_enable_x64", True)
import warnings

import jax.numpy as jnp

warnings.filterwarnings("ignore", module="erfa")
import astropy.units as u
from astropy.coordinates import ICRS, SkyCoord
from astropy.time import Time

from jorbit.data.observatory_codes import OBSERVATORY_CODES
from jorbit.utils.horizons import get_observer_positions
from jorbit.utils.mpc import read_mpc_file


class Observations:
    def __init__(
        self,
        observed_coordinates=None,
        times=None,
        observatories=None,
        astrometric_uncertainties=None,
        mpc_file=None,
    ):
        self._observed_coordinates = observed_coordinates
        self._times = times
        self._observatories = observatories
        self._astrometric_uncertainties = astrometric_uncertainties
        self._mpc_file = mpc_file

        self._input_checks()

        (
            self._ra,
            self._dec,
            self._times,
            self._observatories,
            self._astrometric_uncertainties,
            self._observer_positions,
            self._cov_matrices,
            self._inv_cov_matrices,
            self._cov_log_dets,
        ) = self._parse_astrometry()

        self._final_init_checks()

    def __repr__(self):
        return f"Observations with {len(self._ra)} set(s) of observations"

    def __len__(self):
        return len(self._ra)

    def __add__(self, newobs):
        t = jnp.concatenate([self._times, newobs.times])
        ra = jnp.concatenate([self._ra, newobs.ra])
        dec = jnp.concatenate([self._dec, newobs.dec])
        obs_precision = jnp.concatenate(
            [self._astrometric_uncertainties, newobs.astrometric_uncertainties]
        )
        observer_positions = jnp.concatenate(
            [self._observer_positions, newobs.observer_positions]
        )

        order = jnp.argsort(t)
        return Observations(
            observed_coordinates=SkyCoord(ra=ra[order], dec=dec[order], unit=u.rad),
            times=t[order],
            observatories=observer_positions[order],
            astrometric_uncertainties=obs_precision[order],
            mpc_file=None,
        )

    def __getitem__(self, index):
        return Observations(
            observed_coordinates=SkyCoord(
                ra=self._ra[index], dec=self._dec[index], unit=u.rad
            ),
            times=self._times[index],
            observatories=self._observatories[index],
            astrometric_uncertainties=self._astrometric_uncertainties[index] * u.arcsec,
            mpc_file=self._mpc_file,
        )

    @property
    def ra(self):
        return self._ra

    @property
    def dec(self):
        return self._dec

    @property
    def times(self):
        return self._times

    @property
    def observatories(self):
        return self._observatories

    @property
    def astrometric_uncertainties(self):
        return self._astrometric_uncertainties

    @property
    def observer_positions(self):
        return self._observer_positions

    @property
    def cov_matrices(self):
        return self._cov_matrices

    @property
    def inv_cov_matrices(self):
        return self._inv_cov_matrices

    @property
    def cov_log_dets(self):
        return self._cov_log_dets

    ####################################################################################
    # Initialization helpers
    def _input_checks(self):
        if self._mpc_file is None:
            assert (
                (self._observed_coordinates is not None)
                and (self._times is not None)
                and (self._observatories is not None)
                and (self._astrometric_uncertainties is not None)
            ), (
                "If no MPC file is provided, observed_coordinates, times,"
                " observatories, and astrometric_uncertainties must be given"
                " manually."
            )
            if not isinstance(
                self._times, (type(Time("2023-01-01")) | list | jnp.ndarray)
            ):
                raise ValueError(
                    "times must be either astropy.time.Time, list of astropy.time.Time,"
                    " or jax.numpy.ndarray (interpreted as JD in TDB)"
                )

            assert isinstance(self._observatories, (str | list | jnp.ndarray)), (
                "observatories must be either a string (interpreted as an MPC"
                " observatory code), a list of observatory codes, or a"
                " jax.numpy.ndarray"
            )
            if isinstance(self._observatories, list):
                assert len(self._observatories) == len(self._times), (
                    "If observatories is a list, it must be the same length as"
                    " the number of observations."
                )
            elif isinstance(self._observatories, jnp.ndarray):
                assert len(self._observatories) == len(self._times), (
                    "If observatories is a jax.numpy.ndarray, it must be the"
                    " same length as the number of observations."
                )
        else:
            assert (
                (self._observed_coordinates is None)
                and (self._times is None)
                and (self._observatories is None)
                and (self._astrometric_uncertainties is None)
            ), (
                "If an MPC file is provided, observed_coordinates, times,"
                " observatories, and astrometric_uncertainties must be None."
            )

    def _parse_astrometry(self):

        if self._mpc_file is None:
            (
                observed_coordinates,
                times,
                observatories,
                astrometric_uncertainties,
            ) = (
                self._observed_coordinates,
                self._times,
                self._observatories,
                self._astrometric_uncertainties,
            )

        else:
            (
                observed_coordinates,
                times,
                observatories,
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
        if isinstance(observatories, str):
            observatories = [observatories] * len(times)
        if isinstance(observatories, list):
            for i, loc in enumerate(observatories):
                loc = loc.lower()
                if loc in OBSERVATORY_CODES:
                    observatories[i] = OBSERVATORY_CODES[loc]
                elif "@" in loc:
                    pass
                else:
                    raise ValueError(
                        f"Observer location '{loc}' is not a recognized observatory. Please"
                        " refer to"
                        " https://minorplanetcenter.net/iau/lists/ObsCodesF.html"
                    )

            observer_positions = get_observer_positions(
                times=Time(times, format="jd", scale="tdb"),
                observatories=observatories,
            )
        else:
            observer_positions = observatories

        # UNCERTAINTIES
        if astrometric_uncertainties.shape == ():
            astrometric_uncertainties = (
                jnp.ones(len(times)) * astrometric_uncertainties.to(u.arcsec).value
            )
        if isinstance(astrometric_uncertainties, u.Quantity):
            astrometric_uncertainties = astrometric_uncertainties.to(u.arcsec).value
        # if our uncertainties are 1D, convert to diagonal covariance matrices
        if astrometric_uncertainties.ndim == 1:
            cov_matrices = jnp.array(
                [jnp.diag(jnp.array([a**2, a**2])) for a in astrometric_uncertainties]
            )
        else:
            cov_matrices = astrometric_uncertainties

        inv_cov_matrices = jnp.array([jnp.linalg.inv(c) for c in cov_matrices])

        cov_log_dets = jnp.log(jnp.array([jnp.linalg.det(c) for c in cov_matrices]))

        return (
            ra,
            dec,
            times,
            observatories,
            astrometric_uncertainties,
            observer_positions,
            cov_matrices,
            inv_cov_matrices,
            cov_log_dets,
        )

    def _final_init_checks(self):
        assert (
            len(self._ra)
            == len(self._dec)
            == len(self._times)
            == len(self.observer_positions)
            == len(self.astrometric_uncertainties)
        ), (
            f"Inputs must have the same length. Currently: ra={len(self._ra)}, dec={len(self._dec)}, times={len(self._times)},"
            f" observer_positions={len(self.observer_positions)}, astrometric_uncertainties={len(self.astrometric_uncertainties)}"
        )

        t = self._times[0]
        for i in range(1, len(self._times)):
            assert (
                self._times[i] > t
            ), "Observations must be in ascending chronological order."
            t = self._times[i]

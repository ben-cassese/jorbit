"""The user-facing ephemeris class and wrapper around EphemerisProcessor."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import warnings
from collections.abc import Callable

import astropy.units as u
import jax.numpy as jnp
from astropy.time import Time

warnings.filterwarnings("ignore", module="erfa")

from jorbit.data.constants import (
    ALL_PLANET_IDS,
    ALL_PLANET_LOG_GMS,
    ALL_PLANET_NAMES,
    DE430_ASTEROID_EPHEMERIS_URL,
    DE430_PLANET_EPHEMERIS_URL,
    DEFAULT_ASTEROID_EPHEMERIS_URL,
    DEFAULT_PLANET_EPHEMERIS_URL,
    LARGE_ASTEROID_IDS,
    LARGE_ASTEROID_LOG_GMS,
    LARGE_ASTEROID_NAMES,
)
from jorbit.ephemeris.ephemeris_processors import (
    EphemerisPostProcessor,
    EphemerisProcessor,
)
from jorbit.ephemeris.process_bsp import extract_data, merge_data


class Ephemeris:
    """A class for managing and processing a JPL DE solar system ephemeris.

    Attributes:
        ephs (tuple):
            Tuple of EphemerisProcessor objects for different solar system object
            groups
        processor (Union[EphemerisProcessor, EphemerisPostProcessor]):
            Main processor for ephemeris calculations
        _combined_names (List[str]):
            List of all object names
        _combined_log_gms (List[float]):
            List of logarithmic GM values for all objects
    """

    def __init__(
        self,
        ssos: str = "default planets",
        earliest_time: Time = Time("1980-01-01"),
        latest_time: Time = Time("2050-01-01"),
        postprocessing_func: Callable | None = None,
        de_ephemeris_version: str | None = "440",
    ) -> None:
        """Initialize the Ephemeris object.

        Args:
            ssos (str, optional):
                Specification of solar system objects to include. Options are
                "default planets" (for computing planet positions only) or
                "default solar system" (for computing positions of planets and large
                perturbing asteroids).
            earliest_time (Time, optional):
                The earliest time this ephemeris can compute. This must be > than the
                earliest time in the DE ephemeris file, but ideally shouldn't be much
                earlier than will be be actually used: a narrower time range allows for
                smaller in-memory subsets of the ephemeris to be loaded.
                Defaults to Time("1980-01-01").
            latest_time (Time, optional):
                Similar to earliest_time, but for the end time for ephemeris
                calculations.
                Defaults to Time("2050-01-01").
            postprocessing_func (Optional[Callable], optional):
                Function for post-processing state vectors.
                Defaults to None.
            de_ephemeris_version (Optional[str], optional):
                Version of the JPL DE ephemeris to use. Supported versions are '440'
                and '430'. Defaults to '440'.
        """
        if de_ephemeris_version == "440":
            planet_ephem_file = DEFAULT_PLANET_EPHEMERIS_URL
            asteroid_ephem_file = DEFAULT_ASTEROID_EPHEMERIS_URL
        elif de_ephemeris_version == "430":
            planet_ephem_file = DE430_PLANET_EPHEMERIS_URL
            asteroid_ephem_file = DE430_ASTEROID_EPHEMERIS_URL
        else:
            raise ValueError(
                f"Unsupported planet DE version: {de_ephemeris_version}. "
                "Supported versions are '440' and '430'."
            )
        earliest_time = earliest_time - 100 * u.day  # add a buffer to avoid edge issues
        latest_time = (
            latest_time + 100 * u.day
        )  # buffer should be longer than chunk lengths

        if ssos == "default planets":
            ssos = [
                {
                    "ephem_file": planet_ephem_file,
                    "names": ALL_PLANET_NAMES,
                    "targets": [ALL_PLANET_IDS[name] for name in ALL_PLANET_NAMES],
                    "centers": [0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0],
                    "log_gms": ALL_PLANET_LOG_GMS,
                }
            ]

            def postprocessing_func(x: jnp.ndarray, v: jnp.ndarray) -> tuple:
                # the earth and moon are relative to the earth barycenter, not the sun
                x = x.at[4:6].set(x[4:6] + x[3])
                v = v.at[4:6].set(v[4:6] + v[3])
                x = jnp.delete(x, 3, axis=0)
                v = jnp.delete(v, 3, axis=0)
                return x, v

            postprocessing_func = jax.tree_util.Partial(postprocessing_func)

        elif ssos == "default solar system":
            ssos = [
                {
                    "ephem_file": planet_ephem_file,
                    "names": ALL_PLANET_NAMES,
                    "targets": [ALL_PLANET_IDS[name] for name in ALL_PLANET_NAMES],
                    "centers": [0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0],
                    "log_gms": ALL_PLANET_LOG_GMS,
                },
                {
                    "ephem_file": asteroid_ephem_file,
                    "names": LARGE_ASTEROID_NAMES,
                    "targets": [
                        LARGE_ASTEROID_IDS[name] for name in LARGE_ASTEROID_NAMES
                    ],
                    "centers": [10] * len(LARGE_ASTEROID_IDS),
                    "log_gms": LARGE_ASTEROID_LOG_GMS,
                },
            ]

            def postprocessing_func(x: jnp.ndarray, v: jnp.ndarray) -> tuple:
                # the earth and moon are relative to the earth barycenter, not the sun
                x = x.at[4:6].set(x[4:6] + x[3])
                v = v.at[4:6].set(v[4:6] + v[3])
                x = jnp.delete(x, 3, axis=0)
                v = jnp.delete(v, 3, axis=0)

                # the asteroids are all relative to the sun, not the barycenter
                x = x.at[-16:].set(x[-16:] + x[0])
                v = v.at[-16:].set(v[-16:] + v[0])
                return x, v

            postprocessing_func = jax.tree_util.Partial(postprocessing_func)

        combined_names = []
        combined_log_gms = []
        for sso_group in ssos:
            combined_names += sso_group["names"]
            for n in sso_group["names"]:
                combined_log_gms.append(sso_group["log_gms"][n])
        if "earth_bary" in combined_names:
            ind = combined_names.index("earth_bary")
            _ = combined_names.pop(ind)
            _ = combined_log_gms.pop(ind)
        self._combined_names = combined_names
        self._combined_log_gms = combined_log_gms

        ephs = []
        for sso_group in ssos:
            inits, intlens, coeffs = [], [], []
            for target, center in zip(sso_group["targets"], sso_group["centers"]):
                init, intlen, coeff = extract_data(
                    center, target, sso_group["ephem_file"], earliest_time, latest_time
                )
                inits.append(init)
                intlens.append(intlen)
                coeffs.append(coeff)
            init, intlen, coeff = merge_data(
                inits, intlens, coeffs, earliest_time, latest_time
            )
            gms = []
            for n in sso_group["names"]:
                gms.append(sso_group["log_gms"][n])
            if "earth_bary" in sso_group["names"]:
                ind = sso_group["names"].index("earth_bary")
                _ = gms.pop(ind)
            gms = jnp.array(gms)
            ephs.append(EphemerisProcessor(init, intlen, coeff, gms))
        self.ephs = tuple(ephs)

        # if len(self.ephs) == 1:
        #     self.processor = self.ephs[0]
        # else:
        self.processor = EphemerisPostProcessor(self.ephs, postprocessing_func)

        self.earliest_time = earliest_time
        self.latest_time = latest_time

    def state(self, time: Time) -> dict:
        """Calculate the state vectors for solar system objects at the given time(s).

        This method computes position and velocity vectors for all tracked solar system
        objects at the specified time(s). It can handle arbitrary-length Time inputs.

        Args:
            time (Time): Times at which to evaluate the ephemeris.

        Returns:
            Dict[str, Dict[str, Union[u.Quantity, float]]]:
                Dictionary containing state information
                for each object. The outer dictionary is keyed by object name, and each
                inner dictionary contains
                - 'x': Position vector (astropy.units.Quantity in au)
                - 'v': Velocity vector (astropy.units.Quantity in au/day)
                - 'log_gm': Logarithmic GM value (float)
        """
        if time.shape == ():
            if time < self.earliest_time or time > self.latest_time:
                raise ValueError(
                    f"Requested time {time.iso} ({time.scale}) is outside of "
                    f"the previously requested ephemeris range "
                    f"({self.earliest_time.iso}, {self.earliest_time.scale} to "
                    f"{self.latest_time.iso}, ({self.latest_time.scale}))"
                )
            x, v = self.processor.state(time.tdb.jd)
        else:
            # if any time falls outside of the requested range, raise an error
            if jnp.any((time < self.earliest_time) | (time > self.latest_time)):
                raise ValueError(
                    "One or more requested times are outside of previously requested "
                    f"ephemeris range "
                    f"({self.earliest_time.iso} ({self.earliest_time.scale}) to "
                    f"{self.latest_time.iso} ({self.latest_time.scale}))"
                )
            x, v = jax.vmap(self.processor.state)(time.tdb.jd)
        s = {}
        for n in range(len(self._combined_names)):
            s[self._combined_names[n]] = {
                "x": x[n] * u.au if x.ndim == 2 else x[:, n] * u.au,
                "v": v[n] * u.au / u.day if v.ndim == 2 else v[:, n] * u.au / u.day,
                # "a": a[n] * u.au / u.day**2,
                "log_gm": self._combined_log_gms[n],
            }
        return s

import jax
from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
import numpy as np
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore", module="erfa")
from astropy.time import Time

from jorbit.data import STANDARD_SUN_PARAMS
from jorbit.data.constants import all_planets, large_asteroids
from jorbit.engine.utils import construct_perturbers
from jorbit.engine import (
    j_cart_to_elements,
    j_elements_to_cart,
)


class Particle:
    def __init__(
        self,
        x=None,
        v=None,
        elements=None,
        gm=0,
        time=None,
        observations=None,
        earliest_time=Time("1980-01-01"),
        latest_time=Time("2100-01-01"),
        name="",
        free_orbit=True,
        free_gm=False,
    ):
        assert time is not None, "Must provide an epoch for the particle"
        if isinstance(time, type(Time("2023-01-01"))):
            self._time = time.tdb.jd
        elif isinstance(time, float):
            self._time = time
        elif type(time) == type(jnp.array([1.0])) and len(time) == 1:
            self._time = time[0]
        else:
            raise ValueError(
                "time must be either astropy.time.Time or float (interpreted as JD in"
                " TDB)"
            )

        if (earliest_time != Time("1980-01-01")) | (latest_time != Time("2100-01-01")):
            Q = construct_perturbers(
                planets=[],
                asteroids=["juno"],
                earliest_time=Time("1980-01-01"),
                latest_time=Time("2100-01-01"),
            )
            self._sun_params = Q[0]
        else:
            self._sun_params = STANDARD_SUN_PARAMS

        assert not (
            (x is None) & (elements is None)
        ), "Must provide an initial state for the particle, even if a fit is desired"
        assert not ((x is not None) & (elements is not None)), (
            "Must provide either an initial cartesian state or orbital elements for the"
            " particle, but not both"
        )

        if elements is not None:
            assert (
                (type(elements) == dict)
                & (len(elements) == 6)
                & (
                    set(list(elements.keys()))
                    == {
                        "a",
                        "ecc",
                        "nu",
                        "inc",
                        "Omega",
                        "omega",
                    }
                )
            ), "elements must be a dictionary of standard orbital elements"
            if isinstance(elements["a"], float) or isinstance(elements["a"], int):
                elements["a"] = jnp.array([elements["a"]], dtype=jnp.float64)
                elements["ecc"] = jnp.array([elements["ecc"]], dtype=jnp.float64)
                elements["nu"] = jnp.array([elements["nu"]], dtype=jnp.float64)
                elements["inc"] = jnp.array([elements["inc"]], dtype=jnp.float64)
                elements["Omega"] = jnp.array([elements["Omega"]], dtype=jnp.float64)
                elements["omega"] = jnp.array([elements["omega"]], dtype=jnp.float64)
            x, v = j_elements_to_cart(
                **elements, time=self._time, sun_params=self._sun_params
            )
            x = x[0]
            v = v[0]

        self._x = x
        self._v = v
        self._gm = gm

        self.observations = observations
        if type(observations) != type(None):
            self._time = observations.times[0]
        self._earliest_time = earliest_time
        self._latest_time = latest_time

        self._name = name
        self._free_orbit = free_orbit
        self._free_gm = free_gm

    @property
    def elements(self):
        z = j_cart_to_elements(
            X=self._x[None, :],
            V=self._v[None, :],
            time=self._time,
            sun_params=self._sun_params,
        )
        return dict(zip(["a", "ecc", "nu", "inc", "Omega", "omega"], z))

    @property
    def x(self):
        return self._x

    @property
    def v(self):
        return self._v

    @property
    def gm(self):
        return self._gm

    @property
    def time(self):
        return self._time

    @property
    def earliest_time(self):
        return self._earliest_time

    @property
    def latest_time(self):
        return self._latest_time

    @property
    def name(self):
        return self._name

    @property
    def free_orbit(self):
        return self._free_orbit

    @property
    def free_gm(self):
        return self._free_gm

    def __repr__(self):
        a = f"Particle: {self.name}."
        b = (
            f"Current state:\n--------------\n {self._x} AU\n {self._v} AU/day\n"
            f" {self._time} JD\n"
        )
        c = f"Current Elements:\n-----------------\n" + str(self.elements).replace(
            "), ", "\n"
        )
        return a + "\n" + b + "\n" + c

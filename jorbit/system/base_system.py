import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from abc import ABC, abstractmethod, abstractproperty
import warnings

warnings.filterwarnings("ignore", module="erfa")
from astropy.time import Time
import astropy.units as u


from jorbit.utils import construct_perturbers
from jorbit.data import (
    STANDARD_PLANET_PARAMS,
    STANDARD_ASTEROID_PARAMS,
    STANDARD_PLANET_GMS,
    STANDARD_ASTEROID_GMS,
)
from jorbit.data.constants import ALL_PLANETS, LARGE_ASTEROIDS


class BaseSystem(ABC):
    def __init__(
        self,
        particles,
        planets=ALL_PLANETS,
        asteroids=LARGE_ASTEROIDS,
        fit_planet_gms=False,
        fit_asteroid_gms=False,
        infer_epoch=True,
        verbose=True,
    ):
        self._particles = particles
        self._planets = planets
        self._asteroids = asteroids
        self._fit_planet_gms = fit_planet_gms
        self._fit_asteroid_gms = fit_asteroid_gms
        self._infer_epoch = infer_epoch
        self._verbose = verbose

        if type(self._fit_planet_gms) == bool:
            self._fit_planet_gms = jnp.array([self._fit_planet_gms] * len(planets))
        if type(self._fit_asteroid_gms) == bool:
            self._fit_asteroid_gms = jnp.array(
                [self._fit_asteroid_gms] * len(asteroids)
            )

        for i, p in enumerate(self._particles):
            if p.name == "":
                p.name = f"Particle {i}"

        self._input_checks()

        (
            self._earliest_time,
            self._latest_time,
            self._planet_params,
            self._asteroid_params,
            self._planet_gms,
            self._asteroid_gms,
        ) = self._initialize_planets()

        self._num_devices = jax.local_device_count()

    def __repr__(self):
        return f"{self.__class__.__name__} with {len(self._particles)} particle(s)"

    def __len__(self):
        return len(self._particles)

    ################################################################################
    # Shared methods
    ################################################################################
    def _input_checks(self):
        if not self._infer_epoch:
            t = self._particles[0].time
            for p in self._particles:
                assert p.time == t, (
                    "If not inferring a common epoch, all particles must"
                    " be initialized to the same time"
                )

        names = []
        for p in self._particles:
            names.append(p.name)
        num_unique_names = len(set(names))
        assert num_unique_names == len(self._particles), (
            "2 or more identical names detected. If using custom names, make sure"
            " they are each unique and not 'Particle (int < # of particles in the"
            " system)'"
        )

        if self._fit_planet_gms is not False:
            assert len(self._fit_planet_gms) == len(self._planets), (
                "fit_planet_gms must be a boolean or an array of booleans with"
                " length equal to the number of planets"
            )

        if self._fit_asteroid_gms is not False:
            assert len(self._fit_asteroid_gms) == len(self._asteroids), (
                "fit_asteroid_gms must be a boolean or an array of booleans with"
                " length equal to the number of asteroids"
            )

    def _initialize_planets(self):
        earlys = []
        lates = []

        for p in self._particles:
            earlys.append(p.earliest_time.tdb.jd)
            lates.append(p.latest_time.tdb.jd)

        earliest_time = jnp.min(jnp.array(earlys))
        latest_time = jnp.max(jnp.array(lates))

        if (
            (self._planets != ALL_PLANETS)
            | (self._asteroids != LARGE_ASTEROIDS)
            | (earliest_time != Time("1980-01-01").tdb.jd)
            | (latest_time != Time("2100-01-01").tdb.jd)
        ):
            (
                planet_params,
                asteroid_params,
                planet_gms,
                asteroid_gms,
            ) = construct_perturbers(
                planets=self._planets,
                asteroids=self._asteroids,
                earliest_time=Time(earliest_time, format="jd") - 10 * u.day,
                latest_time=Time(latest_time, format="jd") + 10 * u.day,
            )
        else:
            planet_params = STANDARD_PLANET_PARAMS
            asteroid_params = STANDARD_ASTEROID_PARAMS
            planet_gms = STANDARD_PLANET_GMS
            asteroid_gms = STANDARD_ASTEROID_GMS

        assert len(planet_params[0]) == len(self._planets) + 1, (
            "Ephemeris could not be generated for at least one requested perturbing"
            " planet"
        )
        assert len(asteroid_params[0]) == len(self._asteroids), (
            "Ephemeris could not be generated for at least one requested perturbing"
            " asteroid"
        )

        return (
            earliest_time,
            latest_time,
            planet_params,
            asteroid_params,
            planet_gms,
            asteroid_gms,
        )

    ################################################################################
    # Shared properties
    ################################################################################

    @property
    def xs(self):
        if self._xs.shape[0] == 1:
            return self._xs[0]
        return self._xs

    @xs.setter
    def xs(self, value):
        raise AttributeError(
            "cannot change xs directly- use propagate(), which will update the entire"
            " state of the system"
        ) from None

    @property
    def vs(self):
        if self._vs.shape[0] == 1:
            return self._vs[0]
        return self._vs

    @vs.setter
    def vs(self, value):
        raise AttributeError(
            "cannot change vs directly- use propagate(), which will update the entire"
            " state of the system"
        ) from None

    @property
    def gms(self):
        if self._gms.shape[0] == 1:
            return self._gms[0]
        return self._gms

    @gms.setter
    def gms(self, value):
        raise AttributeError("cannot change gms ") from None

    @property
    def time(self):
        return self._time

    ################################################################################
    # Mandatory methods
    ################################################################################
    @abstractproperty
    def particles(self):
        pass

    @abstractproperty
    def residuals(self):
        pass

    @abstractmethod
    def _initialize_particles(self):
        pass  # modified_particles, epoch, obs_mask, original_observations

    @abstractmethod
    def _generate_likelihood_funcs(self):
        pass  # residual_func, likelihood_func

    @abstractmethod
    def propagate(self, t):
        pass

"""A collection of Chex dataclasses for representing the state of a system of particles."""

import jax

jax.config.update("jax_enable_x64", True)
import chex
import jax.numpy as jnp

from jorbit.astrometry.transformations import (
    cartesian_to_elements,
    elements_to_cartesian,
    horizons_ecliptic_to_icrs,
    icrs_to_horizons_ecliptic,
)
from jorbit.data.constants import SPEED_OF_LIGHT


@chex.dataclass
class SystemState:
    """Contains the state of a system of particles.

    Attributes:
        tracer_positions (jnp.ndarray): Positions of the tracers.
        tracer_velocities (jnp.ndarray): Velocities of the tracers.
        massive_positions (jnp.ndarray): Positions of the massive particles.
        massive_velocities (jnp.ndarray): Velocities of the massive particles.
        log_gms (jnp.ndarray): Logarithm of the gravitational masses.
        time (float): Time of the system state in JD TDB.
        acceleration_func_kwargs (dict): Additional arguments for acceleration functions.
    """

    tracer_positions: jnp.ndarray
    tracer_velocities: jnp.ndarray
    massive_positions: jnp.ndarray
    massive_velocities: jnp.ndarray
    log_gms: jnp.ndarray
    time: float
    acceleration_func_kwargs: dict(default_factory=lambda: {"c2": SPEED_OF_LIGHT**2})


@chex.dataclass
class KeplerianState:
    """Contains the state of a particle in Keplerian elements.

    Attributes:
        semi (float): Semi-major axis.
        ecc (float): Eccentricity.
        inc (float): Inclination.
        Omega (float): Longitude of the ascending node.
        omega (float): Argument of periapsis.
        nu (float): True anomaly.
        time (float): Time of the particle state.
    """

    semi: float
    ecc: float
    inc: float
    Omega: float
    omega: float
    nu: float
    # careful here- adding a default to allow users creating Particles to pass
    # astropy.time.Time objects, which wouldn't work in these dataclasses
    # but, in general, need to specify for the SystemState you get from .to_system()
    # to produce correct accelerations later
    time: float = 2458849.5

    def to_cartesian(self):
        """Converts the Keplerian state to Cartesian coordinates."""
        x, v = elements_to_cartesian(
            self.semi,
            self.ecc,
            self.nu,
            self.inc,
            self.Omega,
            self.omega,
        )
        x = horizons_ecliptic_to_icrs(x)
        v = horizons_ecliptic_to_icrs(v)
        return CartesianState(x=x, v=v, time=self.time)

    def to_keplerian(self):
        """Convert to a Keplerian state.

        Does nothing- this is already a Keplerian state. Included so that both
        KeplerianState and CartesianState have the same interface.
        """
        return self

    def to_system(self):
        """Converts the Keplerian state to a system state."""
        c = self.to_cartesian()
        return SystemState(
            tracer_positions=c.x,
            tracer_velocities=c.v,
            massive_positions=jnp.empty((0, 3)),
            massive_velocities=jnp.empty((0, 3)),
            log_gms=jnp.empty((0,)),
            time=self.time,
            acceleration_func_kwargs={},
        )


@chex.dataclass
class CartesianState:
    """Contains the state of a particle in Cartesian coordinates.

    Attributes:
        x (jnp.ndarray): Position of the particle.
        v (jnp.ndarray): Velocity of the particle.
        time (float): Time of the particle state in JD TDB.
    """

    x: jnp.ndarray
    v: jnp.ndarray
    # same warning as above
    time: float = 2458849.5

    def to_keplerian(self):
        """Converts the Cartesian state to Keplerian elements."""
        x = icrs_to_horizons_ecliptic(self.x)
        v = icrs_to_horizons_ecliptic(self.v)
        a, ecc, nu, inc, Omega, omega = cartesian_to_elements(x, v)
        return KeplerianState(
            semi=a, ecc=ecc, inc=inc, Omega=Omega, omega=omega, nu=nu, time=self.time
        )

    def to_cartesian(self):
        """Convert to a Cartesian state.

        Does nothing- this is already a Cartesian state. Included so that both
        KeplerianState and CartesianState have the same interface.
        """
        return self

    def to_system(self):
        """Converts the Cartesian state to a system state."""
        return SystemState(
            tracer_positions=self.x,
            tracer_velocities=self.v,
            massive_positions=jnp.empty((0, 3)),
            massive_velocities=jnp.empty((0, 3)),
            log_gms=jnp.empty((0,)),
            time=self.time,
            acceleration_func_kwargs={},
        )


@chex.dataclass
class IAS15IntegratorState:
    """Contains the state of the IAS15 integrator."""

    g: jnp.ndarray
    b: jnp.ndarray
    e: jnp.ndarray
    br: jnp.ndarray
    er: jnp.ndarray
    csx: jnp.ndarray
    csv: jnp.ndarray
    a0: jnp.ndarray
    dt: float
    dt_last_done: float

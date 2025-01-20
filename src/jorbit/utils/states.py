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
    tracer_positions: jnp.ndarray
    tracer_velocities: jnp.ndarray
    massive_positions: jnp.ndarray
    massive_velocities: jnp.ndarray
    log_gms: jnp.ndarray
    time: float
    acceleration_func_kwargs: dict(default_factory=lambda: {"c2": SPEED_OF_LIGHT**2})


@chex.dataclass
class KeplerianState:
    semi: float
    ecc: float
    inc: float
    Omega: float
    omega: float
    nu: float
    time: float

    def to_cartesian(self):
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
        return self

    def to_system(self):
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
    x: jnp.ndarray
    v: jnp.ndarray
    time: float

    def to_keplerian(self):
        x = icrs_to_horizons_ecliptic(self.x)
        v = icrs_to_horizons_ecliptic(self.v)
        a, ecc, nu, inc, Omega, omega = cartesian_to_elements(x, v)
        return KeplerianState(
            semi=a, ecc=ecc, inc=inc, Omega=Omega, omega=omega, nu=nu, time=self.time
        )

    def to_cartesian(self):
        return self

    def to_system(self):
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

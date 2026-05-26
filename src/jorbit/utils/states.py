"""A collection of Chex dataclasses for representing the state of a system of particles."""

import jax

jax.config.update("jax_enable_x64", True)
from dataclasses import field

import astropy.units as u
import chex
import jax.numpy as jnp
from astropy.time import Time

from jorbit import Ephemeris
from jorbit.astrometry.transformations import (
    cartesian_to_elements,
    elements_to_cartesian,
    horizons_ecliptic_to_icrs,
    icrs_to_horizons_ecliptic,
)
from jorbit.data.constants import (
    ALL_PLANET_LOG_GMS,
    SPEED_OF_LIGHT,
    TOTAL_SOLAR_SYSTEM_GM,
)

SUN_GM = jnp.exp(ALL_PLANET_LOG_GMS["sun"])


@chex.dataclass
class SystemState:
    """Contains the state of a system of particles."""

    tracer_positions: jnp.ndarray
    tracer_velocities: jnp.ndarray
    massive_positions: jnp.ndarray
    massive_velocities: jnp.ndarray
    log_gms: jnp.ndarray
    time: float
    fixed_perturber_positions: (
        jnp.ndarray
    )  # need a leading axis! (n_substeps, n_perturbers, 3)
    fixed_perturber_velocities: jnp.ndarray
    fixed_perturber_log_gms: jnp.ndarray
    acceleration_func_kwargs: dict  # at a minimum, {"c2": SPEED_OF_LIGHT**2}


@chex.dataclass
class KeplerianState:
    """Contains the *barycentric* state of a particle in Keplerian elements.

    Angles are in degrees. Elements will not agree with those presented in a
    heliocentric frame.

    Time is represented as a ``(relative_time, time_reference)`` pair so the
    state is self-describing about its absolute epoch:

    - ``relative_time`` is the value that flows through to ``SystemState.time``
      and the integrator / acceleration functions. Acceleration functions
      built with a non-zero ``t_ref_jd`` add it back to recover an absolute
      JD; standalone factories built with ``t_ref_jd=0`` use this value
      directly.
    - ``time_reference`` is the absolute JD (TDB) anchor that
      ``relative_time`` is measured against. ``Particle.__init__`` reads
      ``relative_time + time_reference`` to set the particle's reference
      epoch.

    Invariant: ``absolute_jd = relative_time + time_reference``.
    """

    semi: float
    ecc: float
    inc: float
    Omega: float
    omega: float
    nu: float
    acceleration_func_kwargs: dict
    # absolute JD (TDB) anchor that relative_time is measured against
    time_reference: float
    # offset in days from time_reference; this is what `to_system` propagates
    # to SystemState.time
    relative_time: float = field(default_factory=lambda: jnp.array(0.0))
    # 6x6 covariance matrix in Keplerian coordinates (semi, ecc, inc, Omega, omega, nu).
    # Shape (0, 0) means "not set". Not propagated through coordinate transforms.
    cov: jnp.ndarray = field(default_factory=lambda: jnp.empty((0, 0)))

    def to_cartesian(self) -> "CartesianState":
        """Converts the Keplerian state to Cartesian coordinates."""
        x, v = elements_to_cartesian(
            self.semi,
            self.ecc,
            self.nu,
            self.inc,
            self.Omega,
            self.omega,
            TOTAL_SOLAR_SYSTEM_GM,
        )
        x = horizons_ecliptic_to_icrs(x)
        v = horizons_ecliptic_to_icrs(v)
        return CartesianState(
            x=x,
            v=v,
            relative_time=self.relative_time,
            time_reference=self.time_reference,
            acceleration_func_kwargs=self.acceleration_func_kwargs,
        )

    def to_keplerian(self) -> "KeplerianState":
        """Convert to a Keplerian state.

        Does nothing- this is already a Keplerian state. Included so that both
        KeplerianState and CartesianState have the same interface.
        """
        return self

    def to_system(self) -> SystemState:
        """Converts the Keplerian state to a system state.

        ``SystemState.time`` is set from ``self.relative_time``; the
        ``time_reference`` anchor is dropped, since the acceleration function
        carries its own ``t_ref_jd`` for converting back to absolute JD.
        """
        c = self.to_cartesian()
        return SystemState(
            tracer_positions=c.x,
            tracer_velocities=c.v,
            massive_positions=jnp.empty((0, 3)),
            massive_velocities=jnp.empty((0, 3)),
            log_gms=jnp.empty((0,)),
            time=self.relative_time,
            fixed_perturber_positions=jnp.empty((0, 3)),
            fixed_perturber_velocities=jnp.empty((0, 3)),
            fixed_perturber_log_gms=jnp.empty((0,)),
            acceleration_func_kwargs=self.acceleration_func_kwargs,
        )


@chex.dataclass
class CartesianState:
    """Contains the *barycentric* state of a particle in Cartesian coordinates.

    Time is represented as a ``(relative_time, time_reference)`` pair so the
    state is self-describing about its absolute epoch. See
    :class:`KeplerianState` for the full convention.

    Invariant: ``absolute_jd = relative_time + time_reference``.
    """

    x: jnp.ndarray
    v: jnp.ndarray
    acceleration_func_kwargs: dict
    # absolute JD (TDB) anchor that relative_time is measured against
    time_reference: float
    # offset in days from time_reference; this is what `to_system` propagates
    # to SystemState.time
    relative_time: float = field(default_factory=lambda: jnp.array(0.0))
    # 6x6 covariance matrix in Cartesian coordinates (x0, x1, x2, v0, v1, v2).
    # Shape (0, 0) means "not set". Not propagated through coordinate transforms.
    cov: jnp.ndarray = field(default_factory=lambda: jnp.empty((0, 0)))

    def to_keplerian(self) -> KeplerianState:
        """Converts the Cartesian state to Keplerian elements."""
        x = icrs_to_horizons_ecliptic(self.x)
        v = icrs_to_horizons_ecliptic(self.v)
        a, ecc, nu, inc, Omega, omega = cartesian_to_elements(
            x, v, TOTAL_SOLAR_SYSTEM_GM
        )
        return KeplerianState(
            semi=a,
            ecc=ecc,
            inc=inc,
            Omega=Omega,
            omega=omega,
            nu=nu,
            relative_time=self.relative_time,
            time_reference=self.time_reference,
            acceleration_func_kwargs=self.acceleration_func_kwargs,
        )

    def to_cartesian(self) -> "CartesianState":
        """Convert to a Cartesian state.

        Does nothing- this is already a Cartesian state. Included so that both
        KeplerianState and CartesianState have the same interface.
        """
        return self

    def to_system(self) -> SystemState:
        """Converts the Cartesian state to a system state.

        ``SystemState.time`` is set from ``self.relative_time``; the
        ``time_reference`` anchor is dropped, since the acceleration function
        carries its own ``t_ref_jd`` for converting back to absolute JD.
        """
        return SystemState(
            tracer_positions=self.x,
            tracer_velocities=self.v,
            massive_positions=jnp.empty((0, 3)),
            massive_velocities=jnp.empty((0, 3)),
            log_gms=jnp.empty((0,)),
            time=self.relative_time,
            fixed_perturber_positions=jnp.empty((0, 3)),
            fixed_perturber_velocities=jnp.empty((0, 3)),
            fixed_perturber_log_gms=jnp.empty((0,)),
            acceleration_func_kwargs=self.acceleration_func_kwargs,
        )


@chex.dataclass
class IAS15IntegratorState:
    """Contains the state of the IAS15 integrator."""

    g: jnp.ndarray
    b: jnp.ndarray
    e: jnp.ndarray
    csx: jnp.ndarray
    csv: jnp.ndarray
    a0: jnp.ndarray
    dt: float
    dt_last_done: float


@chex.dataclass
class LeapfrogIntegratorState:
    """Contains the state of a leapfrog integrator."""

    dt: float
    C: jnp.ndarray
    D: jnp.ndarray


def _get_sun_state(
    time: Time, de_ephemeris_version: str = "440"
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Helper to get the state vector of the Sun at a given time.

    Uses the local copy of JPL DE440.

    Args:
        time: astropy.time.Time
            The time at which to get the Sun's state.
        de_ephemeris_version: str
            The version of the JPL DE ephemeris to use. Defaults to "440", accepts
            "430".

    Returns:
        tuple:
            A tuple containing the position and velocity of the Sun in AU and
            AU/day, respectively.
    """
    eph = Ephemeris(
        ssos="default planets",
        earliest_time=time - 30 * u.day,
        latest_time=time + 30 * u.day,
        de_ephemeris_version=de_ephemeris_version,
    )
    sun_state = eph.state(time)["sun"]
    return sun_state


def barycentric_to_heliocentric(
    state: CartesianState | KeplerianState,
    time: Time,
    de_ephemeris_version: str = "440",
) -> dict:
    """Helper to compute heliocentric quantities from barycentric states.

    Use the local copy of JPL DE440 to query the state vector of the Sun at the given
    time.

    Args:
        state: CartesianState or KeplerianState
            The barycentric state to convert.
        time: astropy.time.Time
            The time at which to compute the heliocentric elements.
        de_ephemeris_version: str
            The version of the JPL DE ephemeris to use. Defaults to "440", accepts
            "430".

    Returns:
        dict:
            A dictionary containing heliocentric quantities. If the input state is
            Cartesian, returns 'x_helio' and 'v_helio'. If Keplerian, returns
            'a_helio', 'ecc_helio', 'inc_helio', 'Omega_helio', 'omega_helio', and
            'nu_helio'.
    """
    sun_state = _get_sun_state(time=time, de_ephemeris_version=de_ephemeris_version)

    cart = state.to_cartesian()
    helio_x = cart.x - sun_state["x"].value
    helio_v = cart.v - sun_state["v"].value

    if isinstance(state, CartesianState):
        return {"x_helio": helio_x, "v_helio": helio_v}
    elif isinstance(state, KeplerianState):
        helio_x = icrs_to_horizons_ecliptic(helio_x)
        helio_v = icrs_to_horizons_ecliptic(helio_v)
        a_helio, ecc_helio, nu_helio, inc_helio, Omega_helio, omega_helio = (
            cartesian_to_elements(helio_x, helio_v, SUN_GM)
        )
        return {
            "a_helio": a_helio,
            "ecc_helio": ecc_helio,
            "inc_helio": inc_helio,
            "Omega_helio": Omega_helio,
            "omega_helio": omega_helio,
            "nu_helio": nu_helio,
        }
    else:
        raise ValueError(
            "state must be either a barycentric CartesianState or KeplerianState"
        )


def heliocentric_to_barycentric(
    heliocentric_dict: dict,
    time: Time,
    de_ephemeris_version: str = "440",
    acceleration_func_kwargs: dict = {"c2": SPEED_OF_LIGHT**2},
) -> CartesianState | KeplerianState:
    """Helper to compute barycentric quantities from heliocentric states.

    Use the local copy of JPL DE440 to query the state vector of the Sun at the given
    time.

    Args:
        heliocentric_dict: dict
            A dictionary containing heliocentric quantities. If the input state is
            Cartesian, must contain 'x_helio' and 'v_helio'. If Keplerian, must
            contain 'a_helio', 'ecc_helio', 'inc_helio', 'Omega_helio',
            'omega_helio', and 'nu_helio'.
        time: astropy.time.Time
            The time at which to compute the barycentric elements.
        de_ephemeris_version: str
            The version of the JPL DE ephemeris to use. Defaults to "440",
            accepts "430".
        acceleration_func_kwargs: dict
            Additional arguments to associate with the final CartesianState or
            KeplerianState. Defaults to {"c2": SPEED_OF_LIGHT**2}.

    Returns:
        CartesianState or KeplerianState:
            The barycentric state. If the input dict had Cartesian quantities, returns a
            CartesianState. If the inputs were Keplerian, returns a KeplerianState.
    """
    sun_state = _get_sun_state(time=time, de_ephemeris_version=de_ephemeris_version)

    if "x_helio" in heliocentric_dict:
        cart_x = heliocentric_dict["x_helio"] + sun_state["x"].value
        cart_v = heliocentric_dict["v_helio"] + sun_state["v"].value
        return CartesianState(
            x=cart_x,
            v=cart_v,
            time_reference=time.tdb.jd,
            acceleration_func_kwargs=acceleration_func_kwargs,
        )
    elif "a_helio" in heliocentric_dict:
        helio_x, helio_v = elements_to_cartesian(
            jnp.array([heliocentric_dict["a_helio"]]),
            jnp.array([heliocentric_dict["ecc_helio"]]),
            jnp.array([heliocentric_dict["nu_helio"]]),
            jnp.array([heliocentric_dict["inc_helio"]]),
            jnp.array([heliocentric_dict["Omega_helio"]]),
            jnp.array([heliocentric_dict["omega_helio"]]),
            SUN_GM,
        )
        cart_x = horizons_ecliptic_to_icrs(helio_x) + sun_state["x"].value
        cart_v = horizons_ecliptic_to_icrs(helio_v) + sun_state["v"].value
        state = CartesianState(
            x=cart_x,
            v=cart_v,
            time_reference=time.tdb.jd,
            acceleration_func_kwargs=acceleration_func_kwargs,
        )
        return state.to_keplerian()
    else:
        raise ValueError(
            "heliocentric_dict must contain either heliocentric Cartesian or Keplerian quantities"
        )

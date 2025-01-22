import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit import Observations
from jorbit.astrometry.transformations import icrs_to_horizons_ecliptic
from jorbit.data.constants import TOTAL_SOLAR_SYSTEM_GM
from jorbit.utils.states import CartesianState, KeplerianState


def gauss_method_orbit(obs: Observations) -> CartesianState:

    assert len(obs) == 3, "Gauss's method requires 3 (and only 3) observations"

    def radec_to_unit(ra, dec):
        cos_dec = jnp.cos(dec)
        return jnp.array([cos_dec * jnp.cos(ra), cos_dec * jnp.sin(ra), jnp.sin(dec)])

    rho0 = radec_to_unit(obs.ra[0], obs.dec[0])
    rho1 = radec_to_unit(obs.ra[1], obs.dec[1])
    rho2 = radec_to_unit(obs.ra[2], obs.dec[2])

    # Step 1: Calculate time intervals
    tau1 = obs.times[0] - obs.times[1]
    tau3 = obs.times[2] - obs.times[1]
    tau = obs.times[2] - obs.times[0]

    # Step 2: Calculate cross products
    p1 = jnp.cross(rho1, rho2)
    p2 = jnp.cross(rho0, rho2)
    p3 = jnp.cross(rho0, rho1)

    # Step 3: Calculate scalar triple product
    D0 = jnp.dot(rho0, p1)

    # Step 4: Calculate nine scalar quantities
    position_vectors = jnp.stack([p1, p2, p3], axis=1)
    D = obs.observer_positions @ position_vectors

    # Step 5: Calculate scalar position coefficients
    A = (1 / D0) * (-D[0, 1] * tau3 / tau + D[1, 1] + D[2, 1] * tau1 / tau)
    B = (1 / (6 * D0)) * (
        D[0, 1] * (tau3**2 - tau**2) * tau3 / tau
        + D[2, 1] * (tau**2 - tau1**2) * tau1 / tau
    )
    E = jnp.dot(obs.observer_positions[1], rho1)

    # Step 6: Calculate squared scalar distance of second observation
    R2_squared = jnp.dot(obs.observer_positions[1], obs.observer_positions[1])

    # Step 7: Calculate polynomial coefficients
    a = -(A**2 + 2 * A * E + R2_squared)
    b = -2 * TOTAL_SOLAR_SYSTEM_GM * B * (A + E)
    c = -((TOTAL_SOLAR_SYSTEM_GM * B) ** 2)

    # Step 8: Solve for r2 (scalar distance) using Newton-Raphson method
    def polynomial(r):
        return r**8 + a * r**6 + b * r**3 + c

    def polynomial_derivative(r):
        return 8 * r**7 + 6 * a * r**5 + 3 * b * r**2

    # Initial guess
    r2 = 3.0
    for _ in range(10):  # Usually converges in fewer iterations
        f = polynomial(r2)
        f_prime = polynomial_derivative(r2)
        delta = f / f_prime
        r2 = r2 - delta
        if abs(delta) < 1e-11:
            break

    # Step 9: Calculate slant ranges
    rho = jnp.zeros(3)

    # First observation slant range
    num1 = (
        6 * (D[2, 0] * tau1 / tau3 + D[1, 0] * tau / tau3) * r2**3
        + TOTAL_SOLAR_SYSTEM_GM * D[2, 0] * (tau**2 - tau1**2) * tau1 / tau3
    )
    den1 = 6 * r2**3 + TOTAL_SOLAR_SYSTEM_GM * (tau**2 - tau3**2)
    rho = rho.at[0].set((1 / D0) * (num1 / den1 - D[0, 0]))

    # Second observation slant range
    rho = rho.at[1].set(A + TOTAL_SOLAR_SYSTEM_GM * B / r2**3)

    # Third observation slant range
    num3 = (
        6 * (D[0, 2] * tau3 / tau1 - D[1, 2] * tau / tau1) * r2**3
        + TOTAL_SOLAR_SYSTEM_GM * D[0, 2] * (tau**2 - tau3**2) * tau3 / tau1
    )
    den3 = 6 * r2**3 + TOTAL_SOLAR_SYSTEM_GM * (tau**2 - tau1**2)
    rho = rho.at[2].set((1 / D0) * (num3 / den3 - D[2, 2]))

    # Step 10: Calculate position vectors
    r = jnp.zeros((3, 3))
    for i in range(3):
        r = r.at[i].set(obs.observer_positions[i] + rho[i] * [rho0, rho1, rho2][i])

    # Step 11: Calculate Lagrange coefficients and velocities
    # For second observation (as before)
    f1 = 1 - (TOTAL_SOLAR_SYSTEM_GM / (2 * r2**3)) * tau1**2
    f3 = 1 - (TOTAL_SOLAR_SYSTEM_GM / (2 * r2**3)) * tau3**2
    g1 = tau1 - (TOTAL_SOLAR_SYSTEM_GM / (6 * r2**3)) * tau1**3
    g3 = tau3 - (TOTAL_SOLAR_SYSTEM_GM / (6 * r2**3)) * tau3**3

    # Calculate velocity at second observation
    v2 = (-f3 * r[0] + f1 * r[2]) / (f1 * g3 - f3 * g1)

    # # Calculate additional Lagrange coefficients for first observation
    # f21 = 1 - (TOTAL_SOLAR_SYSTEM_GM/(2*r2**3))*(-tau1)**2  # f coefficient from time 2 to 1
    # g21 = -tau1 - (TOTAL_SOLAR_SYSTEM_GM/(6*r2**3))*(-tau1)**3  # g coefficient from time 2 to 1

    # Calculate derivatives of Lagrange coefficients
    fdot21 = (TOTAL_SOLAR_SYSTEM_GM / (r2**3)) * tau1
    gdot21 = 1 + (TOTAL_SOLAR_SYSTEM_GM / (2 * r2**3)) * tau1**2

    # Calculate velocity at first observation using the state transition matrix relationship:
    # r1 = f21*r2 + g21*v2 # but we already have r[0]
    # v1 = fdot21*r2 + gdot21*v2
    v1 = fdot21 * r[1] + gdot21 * v2

    return CartesianState(
        x=r[0],
        v=v1,
        time=obs.times[0],
    )


def simple_circular(ra, dec, semi, time):
    phi = ra
    theta = jnp.pi / 2 - dec

    x = jnp.sin(theta) * jnp.cos(phi)
    y = jnp.sin(theta) * jnp.sin(phi)
    z = jnp.cos(theta)

    x_icrs = jnp.hstack([x, y, z])
    x = icrs_to_horizons_ecliptic(x_icrs)

    # assume we're observing the thing at its highest excursion from the ecliptic:
    inc = jnp.array([jnp.abs(jnp.arcsin(x[2])) / jnp.linalg.norm(x) * 180 / jnp.pi])

    # its longitude of ascending node is the angle between the x-axis and the projection of the vector onto the xy-plane:
    varphi = (jnp.arctan2(x[1], x[0]) * 180 / jnp.pi) % 360
    Omega = jnp.array([varphi]) - 90 if x[2] > 0 else jnp.array([varphi]) + 90

    nu = jnp.array([90.0]) if x[2] > 0 else jnp.array([270.0])
    a = jnp.array([semi])
    ecc = jnp.array([0.0])
    omega = jnp.array([0.0])

    k = KeplerianState(
        semi=a,
        ecc=ecc,
        nu=nu,
        inc=inc,
        Omega=Omega,
        omega=omega,
        time=time,
    )
    c = k.to_cartesian()

    return c

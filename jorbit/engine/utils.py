import jax
from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, lax
import pickle

from ..construct_perturbers import (
    STANDARD_PLANET_PARAMS,
    STANDARD_ASTEROID_PARAMS,
    STANDARD_PLANET_GMS,
    STANDARD_ASTEROID_GMS,
    STANDARD_SUN_PARAMS,
)


from ..data.constants import (
    INV_SPEED_OF_LIGHT,
    ICRS_TO_BARY_ROT_MAT,
    BARY_TO_ICRS_ROT_MAT,
)

from .ephemeris import planet_state
from .slapshot_integrator import single_step


def on_sky(
    xs,
    vs,
    gms,
    times,
    observer_positions,
    planet_params=STANDARD_PLANET_PARAMS,
    asteroid_params=STANDARD_ASTEROID_PARAMS,
    planet_gms=STANDARD_PLANET_GMS,
    asteroid_gms=STANDARD_ASTEROID_GMS,
):
    """on_sky(xs,vs,gms,times,observer_positions,planet_params=STANDARD_PLANET_PARAMS,asteroid_params=STANDARD_ASTEROID_PARAMS,planet_gms=STANDARD_PLANET_GMS,asteroid_gms=STANDARD_ASTEROID_GMS)
    Calculate the RAs and Decs of a system of particles.

    This computes the on-sky position of several particles at several times. Each time
    is associated with one observer position. This function does not account for
    relativistic effects/bending, but does correct for light travel time delay. It
    uses single_step for that, which means this function should not be vmapped for the
    reasons described in single_step's docstring. It does not include the GR corrections
    when moving the particle backwards slightly to account for light travel time.

    This handles particles and times slightly differently than the integration
    functions. Now, everything is flattened so that there is one time for each particle.
    This is because it does not do any long integrations, only ~hours long steps for
    light travel time, so you have to do the integrations elsewhere then feed it the
    results. For example, to get the position of a single object at 5 different times,
    you should use integrate_multiple to get 5 different positions, velocities, and
    epochs, then feed those to on_sky as if they were 5 independent particles.

    Parameters:
        xs (jnp.ndarray(shape=(N, 3))):
            The initial 3D positions of N particles in AU
        vs (jnp.ndarray(shape=(N, 3))):
            The initial 3D velocities of N particles in AU/day
        gms (jnp.ndarray(shape=(N,))):
            The GM values of N particles in AU^3/day^2
        times (jnp.ndarray(shape=(N,))):
            The times to calculate the on-sky positions in TDB JD
        observer_positions (jnp.ndarray(shape=(N, 3))):
            The 3D positions of the observers at each observation time in AU
        planet_params (Tuple[jnp.ndarray(shape=(P,)), jnp.ndarray(shape=(P,)), jnp.ndarray(shape=(P,Q,3,R))], default=STANDARD_PLANET_PARAMS from jorbit.construct_perturbers):
            The ephemeris describing P massive objects in the solar system. The first
            element is the initial time of the ephemeris in seconds since J2000 TDB. The
            second element is the length of the interval covered by each piecewise chunk of
            the ephemeris in seconds (for DE44x planets, this is 16 days, and for
            asteroids, it's 32 days). The third element contains the Q coefficients of the
            R piecewise chunks of Chebyshev polynomials that make up the ephemeris, in 3
            x,y,z dimensions.
        asteroid_params (Tuple[jnp.ndarray(shape=(W,)), jnp.ndarray(shape=(W,)), jnp.ndarray(shape=(W,Z,3,K))], default=STANDARD_ASTEROID_PARAMS from jorbit.construct_perturbers):
            Same as planet_params but for W asteroids. They are separated only in case
            use_GR=True, in which case the planet perturbations are calculated using the
            PPN formalism while the asteroids are still just Newtonian.
        planet_gms (jnp.ndarray(shape=(G,)), default=STANDARD_PLANET_GMS from jorbit.construct_perturbers):
            The GM values of the included planets in AU^3/day^2. If sum(planet_gms) == 0,
            the planets are ignored. If sum(planet_gms) > 0 but G != P, there will be
            problems. To ignore planets, set planet_gms to jnp.array([0.]).
        asteroid_gms (jnp.ndarray(shape=(H,)), default=STANDARD_ASTEROID_GMS from jorbit.construct_perturbers):
            Same as planet_gms but for the asteroids. If sum(asteroid_gms) != 0, then
            H must equal W. To ignore asteroids, set asteroid_gms to jnp.array([0.]).


    Returns:
        Tuple(jnp.ndarray(shape=(N,)), jnp.ndarray(shape=(N,))):
            The RAs and Decs of each particle in radians

    Examples:
        >>> from jorbit.engine import on_sky
        >>> import jax.numpy as jnp
        >>> from astropy.time import Time
        >>> on_sky(
        ...     xs=jnp.array(
        ...         [[0.73291537, -1.85503972, -0.55163327], [0.73291537, -1.85503972, -0.55163327]]
        ...     ),
        ...     vs=jnp.array(
        ...         [[0.0115149, 0.00509674, 0.00161224], [0.0115149, 0.00509674, 0.00161224]]
        ...     ),
        ...     gms=jnp.array([0.0]),
        ...     times=jnp.array([Time("2023-01-01").tdb.jd, Time("2023-01-01").tdb.jd]),
        ...     observer_positions=jnp.array(
        ...         [[-0.17928936, 0.88858155, 0.38544855], [-0.17928936, 0.88858155, 0.38544855]]
        ...     ),
        ... )
    """

    def _on_sky(
        x,
        v,
        gm,
        time,
        observer_position,
        planet_params,
        asteroid_params,
        planet_gms,
        asteroid_gms,
    ):
        # x, v are both just (3,). Need to scan over particles 1 at a time
        # since the corrections for light travel time will be different for each
        # time is a float, jd in tdb scale
        # observer_position is (3,), ICRS just like the particles

        def scan_func(carry, scan_over):
            xz = carry
            earth_distance = jnp.linalg.norm(xz - observer_position)
            light_travel_time = earth_distance * INV_SPEED_OF_LIGHT
            emitted_time = time - light_travel_time
            nudge = emitted_time - time
            # Warning- really this should be integrate, not single_step
            # this assumes the light travel time delay is smaller than the acceptable timestep
            Q = single_step(
                x0=x[None, :],
                v0=v[None, :],
                gms=gm,
                dt=nudge,
                t=time,
                planet_params=planet_params,
                asteroid_params=asteroid_params,
                planet_gms=planet_gms,
                asteroid_gms=asteroid_gms,
                use_GR=True,
            )

            xz = Q[0][0]
            return xz, None

        xz = jax.lax.scan(scan_func, x, jnp.arange(3))[0]

        X = xz - observer_position
        calc_ra = jnp.mod(jnp.arctan2(X[1], X[0]) + 2 * jnp.pi, 2 * jnp.pi)
        calc_dec = jnp.pi / 2 - jnp.arccos(X[-1] / jnp.linalg.norm(X, axis=0))
        return calc_ra, calc_dec

    def scan_func(carry, scan_over):
        x, v, gm, time, observer_position = scan_over
        calc_ra, calc_dec = _on_sky(
            x=x,
            v=v,
            gm=gm,
            time=time,
            observer_position=observer_position,
            planet_params=planet_params,
            asteroid_params=asteroid_params,
            planet_gms=planet_gms,
            asteroid_gms=asteroid_gms,
        )
        return None, (calc_ra, calc_dec)

    return jax.lax.scan(scan_func, None, (xs, vs, gms, times, observer_positions))[1]


def sky_error(calc_ra, calc_dec, true_ra, true_dec):
    """
    Calculate the angular distance between two points on the sky.

    Parameters:
        calc_ra (jnp.ndarray(shape=(N,))):
            The calculated RAs of N points in radians
        calc_dec (jnp.ndarray(shape=(N,))):
            The calculated Decs of N points in radians
        true_ra (jnp.ndarray(shape=(N,))):
            The true RAs of N points in radians
        true_dec (jnp.ndarray(shape=(N,))):
            The true Decs of N points in radians

    Returns:
        jnp.ndarray(shape=(N,)):
            The angular distance between each pair of points in arcsec

    Examples:

        >>> from jorbit.engine import sky_error
        >>> import jax.numpy as jnp
        >>> sky_error(
        ...     calc_ra=jnp.array([5.03529705, 5.03429705]),
        ...     calc_dec=jnp.array([-0.31342869, -0.31342869]),
        ...     true_ra=jnp.array([5.033297051873651, 5.033297051873651]),
        ...     true_dec=jnp.array([-0.313428686137338, -0.313428686137338]),
        ... )
    """
    # all inputs are floats, ICRS positions in radians
    # output is in arcsec

    # Originally was using great circle distance formula, ran into numerical issues
    # for small differences. Following the astropy source on .separation, now using Vincenty
    delta = true_ra - calc_ra
    numerator = jnp.sqrt(
        (jnp.cos(true_dec) * jnp.sin(delta)) ** 2
        + (
            jnp.cos(calc_dec) * jnp.sin(true_dec)
            - jnp.sin(calc_dec) * jnp.cos(true_dec) * jnp.cos(delta)
        )
        ** 2
    )

    denominator = jnp.sin(calc_dec) * jnp.sin(true_dec) + jnp.cos(calc_dec) * jnp.cos(
        true_dec
    ) * jnp.cos(delta)

    return jnp.arctan2(numerator, denominator) * 206264.80624709636


################################################################################
# Element conversion functions
################################################################################


def _barycentricmeanecliptic_to_icrs(bary_vec):
    return jnp.matmul(BARY_TO_ICRS_ROT_MAT, bary_vec)


def barycentricmeanecliptic_to_icrs(xs):
    """
    Convert 3D positions or velocities from the Barycentric Mean Ecliptic frame to ICRS.

    Uses the rotation matrix from ERFA [1]_, just precomputed/hard coded as a jnp.array.

    Parameters:
        xs (jnp.ndarray(shape=(N, 3))):
            The N 3D positions or velocities in the Barycentric Mean Ecliptic frame.

    Returns:
        jnp.ndarray(shape=(N, 3)):
            The N 3D positions or velocities in ICRS.

    References:
        .. [1] liberfa/pyerfa: https://zenodo.org/record/7761918

    Examples:
        >>> from jorbit.engine import barycentricmeanecliptic_to_icrs
        >>> import jax.numpy as jnp
        >>> barycentricmeanecliptic_to_icrs(
        ...     jnp.array(
        ...         [[0.73291537, -1.85503972, -0.55163327], [0.73291537, -1.85503972, -0.55163327]]
        ...     )
        ... )

    """
    return jax.vmap(_barycentricmeanecliptic_to_icrs)(xs)


def _icrs_to_barycentricmeanecliptic(icrs_vec):
    return jnp.matmul(ICRS_TO_BARY_ROT_MAT, icrs_vec)


def icrs_to_barycentricmeanecliptic(xs):
    """
    Convert 3D positions or velocities from ICRS to the Barycentric Mean Ecliptic frame.

    Uses the rotation matrix from ERFA [1]_, just precomputed/hard coded as a jnp.array.

    Parameters:
        xs (jnp.ndarray(shape=(N, 3))):
            The N 3D positions or velocities in ICRS.

    Returns:
        jnp.ndarray(shape=(N, 3)):
            The N 3D positions or velocities in the Barycentric Mean Ecliptic frame.

    References:
        .. [1] liberfa/pyerfa: https://zenodo.org/record/7761918

    Examples:
        >>> from jorbit.engine import icrs_to_barycentricmeanecliptic
        >>> import jax.numpy as jnp
        >>> icrs_to_barycentricmeanecliptic(
        ...     jnp.array(
        ...         [[0.73291537, -1.48253882, -1.24400574], [0.73291537, -1.48253882, -1.24400574]]
        ...     )
        ... )
    """
    return jax.vmap(_icrs_to_barycentricmeanecliptic)(xs)


def cart_to_elements(X, V, time, sun_params=STANDARD_SUN_PARAMS):
    """ """
    # X is (n_particles, 3)
    # V is (n_particles, 3)
    # sun_params is TUPLE (3), the first entries of something like STANDARD_PLANET_PARAMS
    # time is float, the time in TDB

    sun_x, sun_v, _ = planet_state(
        sun_params, jnp.array([time]), velocity=True, acceleration=False
    )

    sun_pos = sun_x[0][0]
    sun_vel = sun_v[0][0]

    sun_pos, sun_vel = icrs_to_barycentricmeanecliptic(jnp.stack((sun_pos, sun_vel)))
    X = icrs_to_barycentricmeanecliptic(X)
    V = icrs_to_barycentricmeanecliptic(V)

    X = X - sun_pos
    V = V - sun_vel

    r = jnp.linalg.norm(X, axis=1)
    v = jnp.linalg.norm(V, axis=1)
    v_r = jnp.sum(X / r[:, None] * V, axis=1)
    v_p = jnp.sqrt(v**2 - v_r**2)

    Z = jnp.cross(X, V)
    z = jnp.linalg.norm(Z, axis=1)

    inc = jnp.arccos(Z[:, -1] / z)

    N = jnp.cross(jnp.array([0, 0, 1]), Z)
    n = jnp.linalg.norm(N, axis=1)
    Omega = jnp.arccos(N[:, 0] / n)
    Omega = jnp.where(N[:, 1] >= 0, Omega, 2 * jnp.pi - Omega)

    e = jnp.cross(V, Z) / STANDARD_PLANET_GMS[0] - X / r[:, None]
    ecc = jnp.linalg.norm(e, axis=1)

    a = z**2 / (STANDARD_PLANET_GMS[0] * (1 - ecc**2))

    omega = jnp.arccos(jnp.sum(e * N, axis=1) / (ecc * n))
    omega = jnp.where(e[:, -1] >= 0, omega, 2 * jnp.pi - omega)

    nu = jnp.arccos(jnp.sum(e * X, axis=1) / (ecc * r))
    nu = jnp.where(v_r >= 0, nu, 2 * jnp.pi - nu)

    return (
        a,
        ecc,
        nu * 180 / jnp.pi,
        inc * 180 / jnp.pi,
        Omega * 180 / jnp.pi,
        omega * 180 / jnp.pi,
    )


def elements_to_cart(
    a, ecc, nu, inc, Omega, omega, time, sun_params=STANDARD_SUN_PARAMS
):
    # Each of the elements are (n_particles, )
    # The angles are in *degrees*. Always assuming orbital element angles are in degrees

    sun_x, sun_v, _ = planet_state(
        sun_params, jnp.array([time]), velocity=True, acceleration=False
    )

    sun_pos = sun_x[0][0]
    sun_vel = sun_v[0][0]

    sun_pos, sun_vel = icrs_to_barycentricmeanecliptic(jnp.stack((sun_pos, sun_vel)))

    nu *= jnp.pi / 180
    inc *= jnp.pi / 180
    Omega *= jnp.pi / 180
    omega *= jnp.pi / 180

    t = (a * (1 - ecc**2))[:, None]
    r_w = (
        t
        / (1 + ecc[:, None] * jnp.cos(nu[:, None]))
        * jnp.column_stack((jnp.cos(nu), jnp.sin(nu), nu * 0.0))
    )
    v_w = (
        jnp.sqrt(STANDARD_PLANET_GMS[0])
        / jnp.sqrt(t)
        * jnp.column_stack((-jnp.sin(nu), ecc + jnp.cos(nu), nu * 0))
    )

    zeros = jnp.zeros_like(omega, dtype=jnp.float64)
    ones = jnp.ones_like(omega, dtype=jnp.float64)
    Rot1 = jnp.array(
        [
            [jnp.cos(-omega), -jnp.sin(-omega), zeros],
            [jnp.sin(-omega), jnp.cos(-omega), zeros],
            [zeros, zeros, ones],
        ]
    )

    Rot2 = jnp.array(
        [
            [ones, zeros, zeros],
            [zeros, jnp.cos(-inc), -jnp.sin(-inc)],
            [zeros, jnp.sin(-inc), jnp.cos(-inc)],
        ]
    )

    Rot3 = jnp.array(
        [
            [jnp.cos(-Omega), -jnp.sin(-Omega), zeros],
            [jnp.sin(-Omega), jnp.cos(-Omega), zeros],
            [zeros, zeros, ones],
        ]
    )

    rot = jax.vmap(lambda r1, r2, r3: r1 @ r2 @ r3, in_axes=(2, 2, 2))(Rot1, Rot2, Rot3)

    x = jax.vmap(lambda x, y: x @ y)(r_w, rot)
    v = jax.vmap(lambda x, y: x @ y)(v_w, rot)

    x = x + sun_pos
    v = v + sun_vel

    x = barycentricmeanecliptic_to_icrs(x)
    v = barycentricmeanecliptic_to_icrs(v)

    return x, v

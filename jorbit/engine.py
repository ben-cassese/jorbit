"""
The "engine" behind most of the functionality.

This module contains the functions that do the heavy lifting of when integrating/fitting
orbits. Everything in this module is written in JAX, and tries as closely as possible
to follow JAX best practices (and avoid `"🔪 the sharp bits 🔪" <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html>`_).
This means that all functions are "pure" (no side effects), all functions can be 
Just-In-Time compiled with jax.jit, and all functions can be differentiated.
Functions should not depend on anything besides their input parameters and pre-imported
constants, which must be jax.numpy arrays or other JAX-compatible types like pytrees.

By design, there are very few guardrails in this module. This is again to
appease JAX's JIT compiler, which does not play well with certain pure Python functions.
The functions here are brittle, and often depend on arrays with specific shapes to run
properly, but will still return something nonsensical if the input arrays are
the wrong shape. Procede with caution, and use the Observation, Particle, and System
classes for more user-friendly interfaces to these functions.

Several other quirks of JAX to keep in mind:

- for loops are possible, but lead to excessively long compilation times.
  jax.lax.scan is a better choice for most cases.
- jax.lax.cond will trace both branches of a conditional, but then only execute one at
  runtime. This is why there are so many throughout these module: often, it'd be
  great to only compute something until a condition is met, but JAX doesn't like
  breaking out of loops early. So, often we loop over a larger number of steps than
  needed, but use lax.cond to only perform an expensive computation if necessary, and
  otherwise run a dummy function which returns the unmodified values.
- BUT if you vmap a function that includes a lax.cond, it *will* execute both branches
  since it internally converts it to a lax.select. This can have serious performance
  implications, in general here so much so that it's worth avoiding vmap, as cool as it is.

"""

import jax
from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, lax
import pickle

from .construct_perturbers import (
    STANDARD_PLANET_PARAMS,
    STANDARD_ASTEROID_PARAMS,
    STANDARD_PLANET_GMS,
    STANDARD_ASTEROID_GMS,
    STANDARD_SUN_PARAMS,
)


from .data.constants import (
    SPEED_OF_LIGHT,
    INV_SPEED_OF_LIGHT,
    ICRS_TO_BARY_ROT_MAT,
    BARY_TO_ICRS_ROT_MAT,
    X_CONSTANT,
    V0_CONSTANT,
    XF_CONSTANT,
    V_CONSTANT,
    VF_CONSTANT,
    B6_CONSTANT,
    H,
)


################################################################################
# Integrator functions
################################################################################


@jit
def planet_state_helper(
    init,
    intlen,
    coefficients,
    tdb,
    velocity=False,
    acceleration=False,
):
    """
    Calculate the position, velocity, and acceleration of one object at a several times.

    This function borrows heavily from Brandon Rhode's
    `jplephem <https://github.com/brandon-rhodes/python-jplephem>`_ package, 
    specifically the jplephem.spk.SPK._compute method. However, this version is written
    for JAX, not numpy. lax.scan is used instead of a for loop, jnp.where is used
    instead of if statements, etc. Also, in addition to position and velocity, you can
    also compute acceleration by differentiating the Cheybshev polynomials again. The
    input parameters are usually derived from the construct_perturbers.py module.

    Parameters:
        init (float):
            The initial time of the ephemeris in seconds since J2000 TDB
        intlen (float):
            The length of the interval covered by each piecewise chunk of the ephemeris in
            seconds. For DE44x, planets, this is 16 days, and for asteroids, it's 32 days.
        coefficients (jnp.ndarray(shape=(N, 3, M))):
            The Chebyshev coefficients describing the ephemeris
            For each of the M equal length piecewise steps, there are N coefficients for
            each of the 3 {x,y,z} dimensions. For the planets in the DE44x series,
            N=14 (high order terms zero-padded in some cases),
            and for asteroids, N=8.
        tdb: jnp.ndarray(shape=(R,))
            The times at which to evaluate the ephemeris in TDB JD
        velocity (bool):
            Whether to calculate the velocity of the planet
        acceleration (bool):
            Whether to calculate the acceleration of the planet. N.B. these values
            will be nonsense if velocity=False.

    Returns:
        Tuple[jnp.ndarray(shape=(R,)), jnp.ndarray(shape=(R,)), jnp.ndarray(shape=(R,))]:
            The position, velocity, and acceleration of the planet at the requested times.
            The full tuple is returned, but the velocity and/or acceleration will be 0s if
            velocity=False and acceleration=False. The full tuple is always returned
            to help give jax.jit fewer reasons to trigger compilations.
            Units are AU, AU/day, AU/day^2

    Notes:
        The Chebyshev polynomials are evaluated using the `Clenshaw algorithm <https://en.wikipedia.org/wiki/Clenshaw_algorithm>`_.
        An implementation exists in scipy/numpy, but not in jax as of 0.4.8

    Examples:
        >>> from jorbit.engine import planet_state_helper
        >>> import jax.numpy as jnp
        >>> from jorbit.construct_perturbers): import STANDARD_PLANET_PARAMS
        >>> from astropy.time import Time
        >>> x, v, a = planet_state_helper(
        ...     init=STANDARD_PLANET_PARAMS[0][0],
        ...     intlen=STANDARD_PLANET_PARAMS[1][0],
        ...     coefficients=STANDARD_PLANET_PARAMS[2][0],
        ...     tdb=jnp.array([Time("2023-01-01").tdb.jd]),
        ...     velocity=True,
        ...     acceleration=True,
        ... )
    """



    tdb2 = 0.0  # leaving in case we ever decide to increase the time precision and use 2 floats

    _, _, n = coefficients.shape

    # 2451545.0 is the J2000 epoch in TDB
    index1, offset1 = jnp.divmod((tdb - 2451545.0) * 86400.0 - init, intlen)
    index2, offset2 = jnp.divmod(tdb2 * 86400.0, intlen)
    index3, offset = jnp.divmod(offset1 + offset2, intlen)
    index = (index1 + index2 + index3).astype(int)

    omegas = index == n
    index = jnp.where(omegas, index - 1, index)
    offset = jnp.where(omegas, offset + intlen, offset)

    coefficients = coefficients[:, :, index]

    s = 2.0 * offset / intlen - 1.0

    def eval_cheby(coefficients, x):
        b_ii = jnp.zeros((3, len(tdb)))
        b_i = jnp.zeros((3, len(tdb)))

        def scan_func(X, a):
            b_i, b_ii = X
            tmp = b_i
            b_i = a + 2 * s * b_i - b_ii
            b_ii = tmp
            return (b_i, b_ii), b_i

        Q = jax.lax.scan(scan_func, (b_i, b_ii), coefficients[:-1])
        b_i, b_ii = Q[0]
        return coefficients[-1] + s * b_i - b_ii, Q[1]

    x, As = eval_cheby(coefficients, s)  # in km here

    # Optionally calculate the velocity by differentiating the Chebyshev polynomials
    # Will return 0s if velocity=False
    def blank_vel(As):
        return jnp.zeros_like(x), (
            jnp.zeros_like(x),
            jnp.zeros((As.shape[0] - 1, 3, tdb.shape[0])),
        )

    def velocity_func(As):
        Q = eval_cheby(2 * As, s)
        v = Q[0] - As[-1]
        v /= intlen
        v *= 2.0  # in km/s here
        return v, Q

    v, Q = jax.lax.cond(velocity, velocity_func, blank_vel, As)

    # Optionally calculate the acceleration by differentiating the Chebyshev polynomials again
    # Note- will give back nonsense if velocity=False
    # Need to calculate that before the acceleration
    def blank_acc(Q):
        return jnp.zeros_like(x)

    def acceleration_func(Q):
        a = eval_cheby(4 * Q[1], s)[0] - 2 * Q[1][-1]
        a /= intlen**2
        a *= 4.0  # in km/s^2 here
        return a

    a = jax.lax.cond(acceleration, acceleration_func, blank_acc, Q)

    # Convert to AU, AU/day, AU/day^2
    return (
        x.T * 6.684587122268446e-09,
        v.T * 0.0005775483273639937,
        a.T * 49.900175484249054,
    )


@jit
def planet_state(
    planet_params,
    times,
    velocity=False,
    acceleration=False,
):
    """
    Calculate the position, velocity, and acceleration of several objects at a several times.

    This is a thin wrapper around planet_state_helper, which is the actual function that does
    the computation from the ephemeris. Uses jax.vmap to vectorize the computation over
    multiple objects at once; as a consequence, each object's ephemeris info needs to
    have the same shape. This is why jorbit.construct_perturbers.construct_perturbers
    zero pads some higher order terms in the Chebyshev polynomials and/or repeats
    piecewise chunks of certain objects.

    Parameters:
        planet_params (Tuple(jnp.ndarray(shape=(N,)), jnp.ndarray(shape=(N,)), jnp.ndarray(shape=(N, M, 3, R)))):
            The initial times, interval lengths, and Chebyshev coefficients for N objects.
            The Chebyshev coefficients are of shape (N, M, 3, R), where M is the number of
            piecewise chunks, 3 is the {x,y,z} dimensions, and R is the number of
            coefficients in each chunk.
        times (jnp.ndarray(shape=(P,))):
            The times at which to evaluate the ephemerides in TDB JD
        velocity (bool):, default=False
            Whether to calculate the velocity of the objects at each time. Must be True if
            acceleration=True
        acceleration (bool):, default=False
            Whether to calculate the acceleration of the objects at each time. N.B.
            these values will be nonsense if velocity=False.

    Returns:
        Tuple[jnp.ndarray(shape=(N, P, 3)), jnp.ndarray(shape=(N, P, 3)), jnp.ndarray(shape=(N, P, 3))]:
            The positions, velocities, and accelerations of the N objects at the P
            requested times. The full tuple is always returned, but the velocity and/or 
            acceleration will be 0s depending on the input flags.
            Units are AU, AU/day, AU/day^2.

    Examples:
        >>> from jorbit.engine import planet_state
        >>> import jax.numpy as jnp
        >>> from jorbit.construct_perturbers): import STANDARD_PLANET_PARAMS
        >>> from astropy.time import Time
        >>> x, v, a = planet_state(
        ...     planet_params=STANDARD_PLANET_PARAMS,
        ...     times=jnp.array([Time("2023-01-01").tdb.jd]),
        ...     velocity=True,
        ...     acceleration=True,
        ... )

    """

    inits, intlens, coeffs = planet_params
    return jax.vmap(planet_state_helper, in_axes=(0, 0, 0, None, None, None))(
        inits, intlens, coeffs, times, velocity, acceleration
    )


@jit
def gr_helper(x, v, planet_xs, planet_vs, planet_as, mu):
    """
    A helper function to calculate the acceleration of an object in the PPN framework.

    This function is called by gr, which is called by acceleration. It uses the
    Parameterized Post-Newtonian [1]_ framework to calculate the acceleration of a tracer
    particle caused by the massive objects in the solar system. Note: this approximation
    only considers the effects of the Sun/planets/asteroids which have pre-computed
    ephemerides. This means that if we inject other massive particles into the system,
    only their Newtonian influence will be considered. Also, even though we assume the
    particle is massless, this function will still be used even if the particle under
    consideration has mass.

    Parameters:
        x (jnp.ndarray(shape=(3,))):
            The 3D position of a particle in AU
        v (jnp.ndarray(shape=(3,))_:
            The 3D velocity of a particle in AU/day
        planet_xs (jnp.ndarray(shape=(N, 3))):
            The 3D positions of the N massive objects in the solar system in AU
        planet_vs (jnp.ndarray(shape=(N, 3))):
            The 3D velocities of the N massive objects in the solar system in AU/day
        planet_as (jnp.ndarray(shape=(N, 3))):
            The 3D accelerations of the N massive objects in the solar system in AU/day^2
        mu (jnp.ndarray(shape=(N,))):
            The GM values of the N massive objects in the solar system in AU^3/day^2

    Returns:
        jnp.ndarray(shape=(3,)):
            The 3D acceleration of the particle in AU/day^2

    References:
        .. [1] Moyer, T. D. 2003, Formulation for Observed and Computed Values of Deep Space Network Data Types for Navigation (Wiley-Interscience, Hoboken, NJ)

    Examples:
        >>> from jorbit.engine import gr_helper, planet_state
        >>> from jorbit.construct_perturbers): import STANDARD_PLANET_PARAMS, STANDARD_PLANET_GMS
        >>> import jax.numpy as jnp
        >>> from astropy.time import Time
        >>> planet_xs, planet_vs, planet_as = planet_state(
        ...     planet_params=STANDARD_PLANET_PARAMS,
        ...     times=jnp.array([Time("2023-01-01").tdb.jd]),
        ...     velocity=True, acceleration=True
        ...     )
        >>> gr_helper(
        ...     x=jnp.array([0.7329153, -1.85503972, -0.5516332]),
        ...     v=jnp.array([0.0115149, 0.00509674, 0.00161224]),
        ...     planet_xs=planet_xs[:,0,:],
        ...     planet_vs=planet_vs[:,0,:],
        ...     planet_as=planet_as[:,0,:],
        ...     mu=STANDARD_PLANET_GMS,
        ... )
    """

    beta = 1
    gamma = 1

    v_vec = v
    r_vec = planet_xs - x
    r = jnp.linalg.norm(r_vec, axis=1)
    v = jnp.linalg.norm(v_vec)
    planet_v = jnp.linalg.norm(planet_vs, axis=1)

    c1 = (mu / r**3)[:, None] * r_vec
    c2 = 1 - (2 * (beta + gamma) / SPEED_OF_LIGHT**2) * (mu / r).sum()
    c3 = -(
        (2 * beta - 1)
        / SPEED_OF_LIGHT**2
        * (
            jnp.repeat((mu / r)[None, :], len(mu), axis=0)
            * (jnp.eye(len(mu)) - 1)
            * (-1)
        ).sum(axis=1)
    )
    c4 = (
        gamma * (v / SPEED_OF_LIGHT) ** 2
        + (1 + gamma) * (planet_v / SPEED_OF_LIGHT) ** 2
    )
    c5 = -(2 * (1 + gamma) / SPEED_OF_LIGHT**2) * (v_vec[None, :] * planet_vs).sum(
        axis=1
    )
    c6 = (
        -3
        / (2 * SPEED_OF_LIGHT**2)
        * (((r_vec - planet_xs) * planet_vs).sum(axis=1) / r) ** 2
    )
    c7 = 1 / (2 * SPEED_OF_LIGHT**2) * ((planet_xs - r_vec) * planet_as).sum(axis=1)
    term1 = (c1 * (c2 + c3 + c4 + c5 + c6 + c7)[:, None]).sum(axis=0)

    c1 = (mu / r**3) * (
        (x[None, :] - planet_xs)
        * ((2 + 2 * gamma) * v_vec[None, :] - (1 + 2 * gamma) * planet_vs)
    ).sum(axis=1)
    c2 = (c1[:, None] * (v_vec - planet_vs)).sum(axis=0)
    term2 = 1 / SPEED_OF_LIGHT**2 * c2

    term3 = (
        (3 + 4 * gamma) / (2 * SPEED_OF_LIGHT**2) * (mu / r)[:, None] * planet_as
    ).sum(axis=0)

    g = term1 + term2 + term3
    # g = -((mu / r**3)[:,None] * r_vec).sum(axis=0) # leaving to quickly toggle newtonian
    return g


@jit
def gr(xs, vs, planet_xs, planet_vs, planet_as, planet_gms):
    """
    Calculate the acceleration of multiple particles caused by multiple planets at
    multiple times in the PPN framework.
    
    This function is a thin wrapper around gr_helper, which is the actual function that does
    the computation. This just uses jax.vmap to vectorize the computation over multiple
    particles and multiple times.

    Parameters:
        xs (jnp.ndarray(shape=(N, M, 3))):
            The 3D positions of N particles at M times in AU
        vs (jnp.ndarray(shape=(N, M, 3))):
            The 3D velocities of N particles at M times in AU/day
        planet_xs (jnp.ndarray(shape=(P, M, 3))):
            The 3D positions of P massive objects at M times in AU
        planet_vs (jnp.ndarray(shape=(P, M, 3))):
            The 3D velocities of P massive objects at M times in AU/day
        planet_as (jnp.ndarray(shape=(P, M, 3))):
            The 3D accelerations of P massive objects at M times in AU/day^2
        planet_gms (jnp.ndarray(shape=(P,))):
            The GM values of P massive objects in AU^3/day^2


    Returns:
        jnp.ndarray(shape=(N, M, 3)):
            The 3D accelerations of N particles at M times in AU/day^2

    Examples:
        >>> from jorbit.engine import gr, planet_state
        >>> from jorbit.construct_perturbers): import STANDARD_PLANET_PARAMS, STANDARD_PLANET_GMS
        >>> import jax.numpy as jnp
        >>> from astropy.time import Time
        >>> planet_xs, planet_vs, planet_as = planet_state(
        ...     planet_params=STANDARD_PLANET_PARAMS,
        ...     times=jnp.array(
        ...         [
        ...             Time("2023-01-01").tdb.jd,
        ...             Time("2023-01-02").tdb.jd,
        ...             Time("2023-01-03").tdb.jd,
        ...             Time("2023-01-04").tdb.jd,
        ...         ]
        ...     ),
        ...     velocity=True,
        ...     acceleration=True,
        ... )
        >>> particle1x = jnp.repeat(
        ...     jnp.array([0.7329153, -1.85503972, -0.5516332])[None, :], 4, axis=0
        ... )
        >>> particle2x = -particle1x
        >>> particle1v = jnp.repeat(
        ...     jnp.array([0.0115149, 0.00509674, 0.00161224])[None, :], 4, axis=0
        ... )
        >>> particle2v = -particle1v
        >>> gr(
        ...     xs=jnp.array([particle1x, particle2x]),
        ...     vs=jnp.array([particle1v, particle2v]),
        ...     planet_xs=planet_xs,
        ...     planet_vs=planet_vs,
        ...     planet_as=planet_as,
        ...     planet_gms=STANDARD_PLANET_GMS,
        ... )
    
    """
    return jax.vmap(
        lambda x, v: jax.vmap(gr_helper, in_axes=(0, 0, 1, 1, 1, None))(
            x, v, planet_xs, planet_vs, planet_as, planet_gms
        )
    )(xs, vs)


@jit
def newtonian_helper(x, planet_xs, mu):
    """
    A helper function to calculate the acceleration of a particle from Newtonian gravity
    
    Parameters:
        x (jnp.ndarray(shape=(3,))):
            The 3D position of a particle in AU
        planet_xs (jnp.ndarray(shape=(N, 3))):
            The 3D positions of the N massive objects in the solar system in AU
        mu (jnp.ndarray(shape=(N,))):
            The GM values of the N massive objects in the solar system in AU^3/day^2

    Returns:
        jnp.ndarray(shape=(3,)):
            The 3D acceleration of the particle caused by the massive objects in AU/day^2

    Examples:
        >>> from jorbit.engine import newtonian_helper, planet_state
        >>> from jorbit.construct_perturbers): import STANDARD_PLANET_PARAMS, STANDARD_PLANET_GMS
        >>> import jax.numpy as jnp
        >>> from astropy.time import Time
        >>> planet_xs, _, _ = planet_state(
        ...     planet_params=STANDARD_PLANET_PARAMS, times=jnp.array([Time("2023-01-01").tdb.jd])
        ... )
        >>> newtonian_helper(
        ...     x=jnp.array([0.7329153, -1.85503972, -0.5516332]),
        ...     planet_xs=planet_xs[:, 0, :],
        ...     mu=STANDARD_PLANET_GMS,
        ... )
    """

    r_vec = x - planet_xs
    r = jnp.linalg.norm(r_vec, axis=1)
    return -((mu / r**3)[:, None] * r_vec).sum(axis=0)


@jit
def newtonian(xs, planet_xs, planet_gms):
    """
    Calculate the gravitational acceleration felt by multiple particles from multiple
    massive objects at multiple times.

    This function is a thin wrapper around newtonian_helper, which is the actual function that
    does the computation. This just uses jax.vmap to vectorize the computation over
    multiple particles and multiple times.

    Parameters:
        xs (jnp.ndarray(shape=(N, M, 3))):
            The 3D positions of N particles at M times in AU
        planet_xs (jnp.ndarray(shape=(P, M, 3))):
            The 3D positions of P massive objects at M times in AU
        planet_gms (jnp.ndarray(shape=(P,))):
            The GM values of P massive objects in AU^3/day^2

    Returns:
        jnp.ndarray(shape=(N, M, 3)):
            The 3D accelerations of N particles at M times in AU/day^2

    Examples:
        >>> from jorbit.engine import newtonian, planet_state
        >>> from jorbit.construct_perturbers): import STANDARD_PLANET_PARAMS, STANDARD_PLANET_GMS
        >>> import jax.numpy as jnp
        >>> from astropy.time import Time
        >>> planet_xs, _, _ = planet_state(
        ...     planet_params=STANDARD_PLANET_PARAMS,
        ...     times=jnp.array(
        ...         [
        ...             Time("2023-01-01").tdb.jd,
        ...             Time("2023-02-01").tdb.jd,
        ...             Time("2023-03-01").tdb.jd,
        ...         ]
        ...     ),
        ... )
        >>> p1 = jnp.repeat(jnp.array([0.7329153, -1.85503972, -0.5516332])[None, :], 3, axis=0)
        >>> newtonian(
        ...     xs=jnp.stack((p1, p1), axis=0), planet_xs=planet_xs, planet_gms=STANDARD_PLANET_GMS
        ... )

    """
    return jax.vmap(
        lambda x: jax.vmap(newtonian_helper, in_axes=(0, 1, None))(x, planet_xs, planet_gms)
    )(xs)


@jit
def acceleration(
    xs,
    vs,
    gms,
    planet_xs,
    planet_vs,
    planet_as,
    asteroid_xs,
    planet_gms=jnp.array([0]),
    asteroid_gms=jnp.array([0]),
    use_GR=False,
):
    """
    Calculate the acceleration of multiple particles at multiple times

    "Planets" and "Asteroids" are called out separately only to mark which objects are
    calculated using the PPN GR formalism and which can be calculated using Newtonian
    gravity.

    Parameters:
        xs (jnp.ndarray(shape=(N, M, 3))):
            The 3D positions of N particles at M times.
            If in the solar system, in AU, otherwise scale free.
        vs (jnp.ndarray(shape=(N, M, 3))):
            The 3D velocities of N particles at M times.
            If in the solar system, in AU/day, otherwise scale free.
        gms (jnp.ndarray(shape=(N,))):
            The GM values of N particles.
            If in the solar system, in AU^3/day^2, otherwise scale free.
        planet_xs (jnp.ndarray(shape=(P, M, 3))):
            The 3D positions of P massive objects at M times in AU.
            Irrelevant if planet_gms is left to default value, and can be left as empty array.
        planet_vs (jnp.ndarray(shape=(P, M, 3))):
            The 3D velocities of P massive objects at M times in AU/day.
            Irrelevant if planet_gms is left to default value, and can be left as empty array.
        planet_as (jnp.ndarray(shape=(P, M, 3))):
            The 3D accelerations of P massive objects at M times in AU/day^2.
            Irrelevant if planet_gms is left to default value, and can be left as empty array.
        asteroid_xs (jnp.ndarray(shape=(Q, M, 3))):
            The 3D positions of Q massive objects at M times in AU.
            Irrelevant if asteroid_gms is left to default value, and can be left as empty array.
        planet_gms (jnp.ndarray(shape=(P,)), default=jnp.array([0])):
            The GM values of P massive objects in AU^3/day^2.
            If left to default value, the gravitational influence of the planets is ignored.
        asteroid_gms (jnp.ndarray(shape=(Q,)), default=jnp.array([0])):
            The GM values of Q massive objects in AU^3/day^2.
            If left to default value, the gravitational influence of the asteroids is ignored.

    Returns:
        jnp.ndarray(shape=(N, M, 3)):
            The 3D accelerations of N particles at M times in AU/day^2
    
    
    Examples:
        Circular acceleration of a particle around a point source:

        >>> from jorbit.engine import acceleration
        >>> import jax.numpy as jnp
        >>> acceleration(
        >>>     xs=jnp.array([[[0, 0, 0]], [[1, 0, 0]]]),
        >>>     vs=jnp.array([[[0, 0, 0]], [[0, 1, 0]]]),
        >>>     gms=jnp.array([1, 0]),
        >>>     planet_xs=jnp.empty((1,1,3)),
        >>>     planet_vs=jnp.empty((1,1,3)),
        >>>     planet_as=jnp.empty((1,1,3)),
        >>>     asteroid_xs=jnp.empty((1,1,3)),
        >>> )

        Newtonian gravity only, massive planets only, 2 particles at 8 times:

        >>> from jorbit.engine import acceleration, planet_state
        >>> from jorbit.construct_perturbers): import STANDARD_PLANET_PARAMS, STANDARD_PLANET_GMS
        >>> import jax.numpy as jnp
        >>> from astropy.time import Time
        >>> (
        ...     planet_xs,
        ...     _,
        ...     _,
        ... ) = planet_state(
        ...     planet_params=STANDARD_PLANET_PARAMS,
        ...     times=jnp.array(
        ...         [
        ...             Time("2023-01-01").tdb.jd,
        ...             Time("2023-02-01").tdb.jd,
        ...             Time("2023-03-01").tdb.jd,
        ...             Time("2023-04-01").tdb.jd,
        ...             Time("2023-05-01").tdb.jd,
        ...             Time("2023-06-01").tdb.jd,
        ...             Time("2023-07-01").tdb.jd,
        ...             Time("2023-08-01").tdb.jd,
        ...         ]
        ...     ),
        ...     velocity=False,
        ...     acceleration=False,
        ... )
        >>> particle1x = jnp.repeat(
        ...     jnp.array([0.7329153, -1.85503972, -0.5516332])[None, :], 8, axis=0
        ... )
        >>> particle2x = -particle1x
        >>> particle1v = jnp.repeat(
        ...     jnp.array([0.0115149, 0.00509674, 0.00161224])[None, :], 8, axis=0
        ... )
        >>> particle2v = -particle1v
        >>> acceleration(
        ...     xs=jnp.array([particle1x, particle2x]),
        ...     vs=jnp.array([particle1v, particle2v]),
        ...     gms=jnp.array([0.0, 0]),
        ...     planet_xs=planet_xs,
        ...     planet_vs=jnp.empty((1, 8, 3)),
        ...     planet_as=jnp.empty((1, 8, 3)),
        ...     asteroid_xs=jnp.empty((1, 8, 3)),
        ...     planet_gms=STANDARD_PLANET_GMS,
        ... )
        
        Two particles, 4 times, just the Sun, Jupiter, and the asteroid Psyche:

        >>> from jorbit.engine import acceleration, planet_state
        >>> from jorbit.construct_perturbers): import construct_perturbers
        >>> import jax.numpy as jnp
        >>> from astropy.time import Time
        >>> times = jnp.array(
        ...     [
        ...         Time("2023-01-01").tdb.jd,
        ...         Time("2023-02-01").tdb.jd,
        ...         Time("2023-03-01").tdb.jd,
        ...         Time("2023-04-01").tdb.jd,
        ...     ]
        ... )
        >>> planet_params, asteroid_params, planet_gms, asteroid_gms = construct_perturbers(
        ...     planets=["jupiter"], asteroids=["psyche"]
        ... )
        >>> (
        ...     planet_xs,
        ...     planet_vs,
        ...     planet_as,
        ... ) = planet_state(
        ...     planet_params=planet_params,
        ...     times=times,
        ...     velocity=True,
        ...     acceleration=True,
        ... )
        >>> asteroid_xs, _, _ = planet_state(
        ...     planet_params=asteroid_params,
        ...     times=times,
        ...     velocity=False,
        ...     acceleration=False,
        ... )
        >>> particle1x = jnp.repeat(
        ...     jnp.array([0.7329153, -1.85503972, -0.5516332])[None, :], 4, axis=0
        ... )
        >>> particle2x = -particle1x
        >>> particle1v = jnp.repeat(
        ...     jnp.array([0.0115149, 0.00509674, 0.00161224])[None, :], 4, axis=0
        ... )
        >>> particle2v = -particle1v
        >>> acceleration(
        ...     xs=jnp.array([particle1x, particle2x]),
        ...     vs=jnp.array([particle1v, particle2v]),
        ...     gms=jnp.array([0.0, 0]),
        ...     planet_xs=planet_xs,
        ...     planet_vs=planet_vs,
        ...     planet_as=planet_as,
        ...     asteroid_xs=asteroid_xs,
        ...     planet_gms=planet_gms,
        ...     asteroid_gms=asteroid_gms,
        ...     use_GR=True,
        ... )
    """
    # xs is (n_particles, n_times, 3)
    # gms is (n_particles). just 0 for tracer particles- kinda wasteful, but ah well for now

    # planet_xs is (n_planets, n_times, 3)
    # planet_gms is (n_planets)
    # asteroid_xs is (n_asteroids, n_times, 3)
    # asteroid_gms is (n_asteroids)

    A = jnp.zeros(xs.shape)


    # The fixed planets in the system- separated from asteroids in case you want GR corrections
    def true_func():
        def gr_func():
            return gr(
                xs=xs,
                vs=vs,
                planet_xs=planet_xs,
                planet_vs=planet_vs,
                planet_as=planet_as,
                planet_gms=planet_gms,
            )

        def newtonian_func():
            return newtonian(xs=xs, planet_xs=planet_xs, planet_gms=planet_gms)

        return jax.lax.cond(use_GR, gr_func, newtonian_func)

    def false_func():
        return jnp.zeros(xs.shape)

    A += jax.lax.cond(jnp.sum(planet_gms) > 0, true_func, false_func)

    # The asteroids/kbos in the system
    def true_func():
        return newtonian(xs=xs, planet_xs=asteroid_xs, planet_gms=asteroid_gms)

    def false_func():
        return jnp.zeros(xs.shape)

    A += jax.lax.cond(jnp.sum(asteroid_gms) > 0, true_func, false_func)

    # The self interaction of the massive particles
    def scan_func(carry, scan_over):
        d = scan_over[None, :] - scan_over[:, None]
        r_vec = d + jnp.tile(jnp.eye(len(xs))[:, :, None], 3)
        r = jnp.linalg.norm(r_vec, axis=2)
        s = (
            ((gms / r**3)[:, :, None] * r_vec)
            * (jnp.tile(jnp.eye(len(xs))[:, :, None], 3) - 1)
            * (-1)
        )
        a = jnp.sum(s, axis=1)
        return None, a

    ts = jnp.moveaxis(xs, 0, 1)  # (n_times, n_particles, 3)

    def true_func():
        tmp = lax.scan(scan_func, None, ts)[1]
        return jnp.moveaxis(tmp, 0, 1)

    def false_func():
        return jnp.zeros_like(A)

    A += jax.lax.cond(jnp.sum(gms) > 0, true_func, false_func)

    return A


@jit
def inferred_xs(As, v0, x0, dt):
    """
    Compute the 8 intermediate cartesian positions (substeps) during an integration step

    Parameters:
        As (jnp.ndarray(shape=(N, 8, 3))):
            The 3D accelerations of N particles felt at the 8 substep times in AU/day^2
        v0 (jnp.ndarray(shape=(N, 3))):
            The initial 3D velocities of N particles in AU/day
        x0 (jnp.ndarray(shape=(N, 3))):
            The initial 3D positions of N particles in AU
        dt (float):
            The timestep in days

    Returns:
        jnp.ndarray(shape=(N, 8, 3)):
            The 3D positions of N particles at the 8 substep times in AU

    Examples:
        >>> from jorbit.engine import inferred_xs
        >>> import jax.numpy as jnp
        >>> inferred_xs(As=jnp.zeros((2,8,3)), v0=jnp.ones((2,3)), x0=jnp.zeros((2,3)), dt=1)
    """
    # As is (n_particles, 8, 3)
    # v0 is (n_particles, 3)
    # x0 is (n_particles, 3)
    return (
        dt**2 * (X_CONSTANT[None, :, :, None] * As[:, None, :, :]).sum(axis=2)
        + dt * (v0[:, None, :] * V0_CONSTANT[None, :, None])
        + x0[:, None, :]
    )


@jit
def final_x_prediction(As, x0, v0, dt):
    """
    Compute the final cartesian position at the end of an integration step

    Parameters:
        As (jnp.ndarray(shape=(N, 8, 3))):
            The 3D accelerations of N particles felt at the 8 substep times in AU/day^2
        x0 (jnp.ndarray(shape=(N, 3))):
            The initial 3D positions of N particles in AU
        v0 (jnp.ndarray(shape=(N, 3))):
            The initial 3D velocities of N particles in AU/day
        dt (float):
            The timestep in days

    Returns:
        jnp.ndarray(shape=(N, 3)):
            The 3D positions of N particles at the end of the integration step in AU

    Examples:
        >>> from jorbit.engine import final_x_prediction
        >>> import jax.numpy as jnp
        >>> final_x_prediction(As=jnp.ones((2,8,3)), v0=jnp.ones((2,3)), x0=jnp.zeros((2,3)), dt=1)
    """
    return dt**2 * (XF_CONSTANT[None, :, None] * As).sum(axis=1) + dt * v0 + x0


@jit
def inferred_vs(As, v0, dt):
    """
    Compute the 8 intermediate cartesian velocities (substeps) during an integration step

    Parameters:
        As (jnp.ndarray(shape=(N, 8, 3))):
            The 3D accelerations of N particles felt at the 8 substep times in AU/day^2
        v0 (jnp.ndarray(shape=(N, 3))):
            The initial 3D velocities of N particles in AU/day
        dt (float):
            The timestep in days

    Returns:
        jnp.ndarray(shape=(N, 8, 3)):
            The 3D velocities of N particles at the 8 substep times in AU/day

    Examples:
        >>> from jorbit.engine import inferred_vs
        >>> import jax.numpy as jnp
        >>> inferred_vs(As=jnp.zeros((2,8,3)), v0=jnp.ones((2,3)), dt=1)
    """
    return (
        dt * (V_CONSTANT[None, :, :, None] * As[:, None, :, :]).sum(axis=2)
        + v0[:, None, :]
    )


@jit
def final_v_prediction(As, v0, dt):
    """
    Compute the final cartesian velocity at the end of an integration step

    Parameters:
        As (jnp.ndarray(shape=(N, 8, 3))):
            The 3D accelerations of N particles felt at the 8 substep times in AU/day^2
        v0 (jnp.ndarray(shape=(N, 3))):
            The initial 3D velocities of N particles in AU/day
        dt (float):

    Returns:
        jnp.ndarray(shape=(N, 3)):
            The 3D velocities of N particles at the end of the integration step in AU/day

    Examples:
        >>> from jorbit.engine import final_v_prediction
        >>> import jax.numpy as jnp
        >>> final_v_prediction(As=jnp.ones((2,8,3)), v0=jnp.ones((2,3)), dt=1)
    """
    return dt * (VF_CONSTANT[None, :, None] * As).sum(axis=1) + v0


@jit
def b6(As):
    """
    Compute the B6 constant from Rein and Spiegel 2015 [1]_ from the 8 substep accelerations

    Parameters:
        As (jnp.ndarray(shape=(N, 8, 3))):
            The 3D accelerations of N particles felt at the 8 substep times in AU/day^2
    
    Returns:
        jnp.ndarray(shape=(N,3)):
            The B6 constant for each particle in 3 dimensions in AU/day^2

    References:
        .. [1] Rein and Spiegel 2015: https://doi.org/10.1093/mnras/stu2164

    Examples:
        >>> from jorbit.engine import b6
        >>> import jax.numpy as jnp
        >>> b6(As=jnp.ones((5,8,3))*1000)
    """
    return (B6_CONSTANT[None, :, None] * As).sum(axis=1)


@jit
def single_step(
    x0,
    v0,
    gms,
    dt,
    t,
    planet_params=STANDARD_PLANET_PARAMS,
    asteroid_params=STANDARD_ASTEROID_PARAMS,
    planet_gms=STANDARD_PLANET_GMS,
    asteroid_gms=STANDARD_ASTEROID_GMS,
    use_GR=False,
):
    """single_step(x0,v0,gms,dt,t,planet_params=STANDARD_PLANET_PARAMS,asteroid_params=STANDARD_ASTEROID_PARAMS,planet_gms=STANDARD_PLANET_GMS,asteroid_gms=STANDARD_ASTEROID_GMS,use_GR=False,)

    Propagate a system of particles by one integration timestep

    This forms the core of the integrator. It attempts to take a step forwards, then
    evaluates whether a) the "predictor-corrector" loop converged and b) if the timestep
    was too large. It does not do anything with this information other than pass it
    along- the returned Xs and Vs are not guaranteed to be correct. The
    "predictor-corrector" is pretty crude compared to something like IAS15- [1]_ it
    takes in no information about the previous timestep, and instead starts by assuming
    particles move in straight lines. It then iteratively corrects this assumption by
    calculating the accelerations felt by the particles at the 8 substep times, then 
    uses thoseaccelerations to calculate the intermediate positions and velocities. This
    continuesuntil the difference in positions between successive iterations falls 
    below somethreshold. This usually takes 3-5 iterations compared to IAS15's ~2, but
    for now itworks well and fast enough.

    This is inspired by IAS15, but is not a true implementation of it. It is less
    accurate, slower, and diverges more often. Improvements are definitely possible,
    but for the purpose of this package/solar system orbit fitting, numerical errors
    account for << 0.1% of the discrepencies between integrations using this and JPL
    Horizons. Exact masses and GR prescriptions are much more important.

    Note: this function uses lax.cond within lax.scan to conditionally run the
    "expensive" predictor-corrector loop if it has not converged yet. As a result,
    vmapping this function will likely be very slow, since that would convert the
    lax.cond to a lax.select, which will run the expensive function even when it has
    already converged.

    Parameters:
        x0 (jnp.ndarray(shape=(N, 3))):
            The initial 3D positions of N particles in AU
        v0 (jnp.ndarray(shape=(N, 3))):
            The initial 3D velocities of N particles in AU/day
        gms (jnp.ndarray(shape=(N,))):
            The GM values of N particles in AU^3/day^2
        dt (float):
            The timestep in days
        t (float):
            The current time in TDB JD
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
        use_GR (bool, default=False):
            Whether to use the PPN formalism to calculate the gravitational influence of
            the planets. If False, the planets are treated as Newtonian point masses. The
            asteroids are always treated as Newtonian point masses regardless of this flag.

    Returns:
        Tuple[jnp.ndarray(shape=(N, 3)), jnp.ndarray(shape=(N, 3)), float, float, bool, bool, int]:
        x (jnp.ndarray(shape=(N, 3))):
            The 3D positions of N particles at the end of the timestep step in AU
        v (jnp.ndarray(shape=(N, 3))):
            The 3D velocities of N particles at the end of the timestep step in AU/day
        dt (float):
            The timestep in days. Matches the input dt.
        dt_acceptable (float):
            The largest timestep that would have been acceptable for this step in days.
            Calculated using Eq. 11 of Rein and Spiegel 2015.
        timestep_success (bool):
            Whether the attempted timestep was smaller than dt_acceptable. Note that the
            step can still fail even if timestep_success and converge_success are both True.
        converge_success (bool):
            Whether the "predictor-corrector" loop converged. Note that the step can still
            fail even if timestep_success and converge_success are both True.
        steps_to_converge (int):
            The number of iterations it took for the "predictor-corrector" loop to converge.


    References:
        .. [1] Rein and Spiegel 2015: https://doi.org/10.1093/mnras/stu2164
    
    Examples:
    
        A single step of a main belt asteroid:

        >>> from jorbit.engine import single_step
        >>> from jorbit.construct_perturbers): import (
        >>>     STANDARD_PLANET_PARAMS,
        >>>     STANDARD_ASTEROID_PARAMS,
        >>>     STANDARD_PLANET_GMS,
        >>>     STANDARD_ASTEROID_GMS,
        >>> )
        >>> import jax.numpy as jnp
        >>> from astropy.time import Time
        >>> single_step(
        ...     x0=jnp.array([[0.73291537, -1.85503972, -0.55163327]]),
        ...     v0=jnp.array([[0.0115149, 0.00509674, 0.00161224]]),
        ...     gms=jnp.array([0.0]),
        ...     dt=1.0,
        ...     t=Time("2023-01-01").tdb.jd,
        ...     planet_params=STANDARD_PLANET_PARAMS,
        ...     asteroid_params=STANDARD_ASTEROID_PARAMS,
        ...     planet_gms=STANDARD_PLANET_GMS,
        ...     asteroid_gms=STANDARD_ASTEROID_GMS,
        ...     use_GR=True,
        ... )

        Circular orbit of a tracer particle in a scale-free system:

        >>> from jorbit.engine import single_step
        >>> import jax.numpy as jnp
        >>> from astropy.time import Time
        >>> single_step(
        ...     x0=jnp.array([[1.0, 0, 0], [0, 0, 0]]),
        ...     v0=jnp.array([[0.0, 1, 0], [0, 0, 0]]),
        ...     gms=jnp.array([0.0, 1]),
        ...     dt=0.1,
        ...     t=0,
        ...     planet_gms=jnp.array([0]),
        ...     asteroid_gms=jnp.array([0]),
        ... )
    """
    # defaults to including planets/asteroids. if you leave them out, make sure there's more than 1 particle
    times = t + H * dt

    def true_func():
        planet_xs, planet_vs, planet_as = planet_state(
            planet_params, times, velocity=True, acceleration=True
        )
        return (planet_xs, planet_vs, planet_as)

    def false_func():
        return (
            jnp.zeros((len(planet_params[0]), len(times), 3)),
            jnp.zeros((len(planet_params[0]), len(times), 3)),
            jnp.zeros((len(planet_params[0]), len(times), 3)),
        )

    planet_xs, planet_vs, planet_as = lax.cond(
        jnp.sum(planet_gms) > 0, true_func, false_func
    )

    def true_func():
        asteroid_xs, _, _ = planet_state(
            asteroid_params, times, velocity=False, acceleration=False
        )
        return asteroid_xs

    def false_func():
        return jnp.zeros((len(asteroid_params[0]), len(times), 3))

    asteroid_xs = lax.cond(jnp.sum(asteroid_gms) > 0, true_func, false_func)

    predicted_as = jnp.zeros((x0.shape[0], 8, 3))
    # predicted_as = next_As(t=t, dt=dt, last_As=last_As, last_dt=last_dt)

    xs = inferred_xs(As=predicted_as, v0=v0, x0=x0, dt=dt)
    # jax.debug.print("{x}", x=xs.shape)
    # jax.debug.print("{v}", v=planet_xs.shape)
    vs = inferred_vs(As=predicted_as, v0=v0, dt=dt)

    def true_func(C):
        (predicted_as, guessed_xs, guessed_vs, diff) = C
        new_guessed_as = acceleration(
            xs=guessed_xs,
            vs=guessed_vs,
            gms=gms,
            planet_xs=planet_xs,
            planet_vs=planet_vs,
            planet_as=planet_as,
            planet_gms=planet_gms,
            asteroid_xs=asteroid_xs,
            asteroid_gms=asteroid_gms,
            use_GR=use_GR,
        )

        xs = inferred_xs(new_guessed_as, v0, x0, dt)
        vs = inferred_vs(new_guessed_as, v0, dt)
        diff = jnp.max(jnp.abs(guessed_xs - xs))
        return new_guessed_as, xs, vs, diff

    def false_func(C):
        predicted_as, guessed_xs, guessed_vs, diff = C
        return predicted_as, guessed_xs, guessed_vs, diff

    def scan_func(carry, scan_over):
        predicted_as, guessed_xs, guessed_vs, diff = carry
        predicted_as, guessed_xs, guessed_vs, diff = lax.cond(
            diff > 1e-24,
            true_func,
            false_func,
            (predicted_as, guessed_xs, guessed_vs, diff),
        )

        return (predicted_as, guessed_xs, guessed_vs, diff), diff < 1e-24

    Q = jax.lax.scan(scan_func, (predicted_as, xs, vs, 1.0), jnp.arange(25))
    As, _, _, diff = Q[0]
    converge_success = diff < 1e-24
    steps_to_converge = 25 - jnp.sum(Q[1]) + 1

    epsilon_b = 1e-9
    dt_acceptable = dt * (
        epsilon_b / (jnp.max(jnp.abs(b6(As))) / jnp.max(jnp.abs(As)))
    ) ** (1 / 7)
    timestep_success = jnp.abs(dt) <= jnp.abs(dt_acceptable)

    x = final_x_prediction(As, x0, v0, dt)
    v = final_v_prediction(As, v0, dt)

    return (
        x,
        v,
        dt,
        dt_acceptable,
        timestep_success,
        converge_success,
        steps_to_converge,
    )


@jit
def integrate(
    xs,
    vs,
    gms,
    initial_time,
    final_time,
    planet_params=STANDARD_PLANET_PARAMS,
    asteroid_params=STANDARD_ASTEROID_PARAMS,
    planet_gms=STANDARD_PLANET_GMS,
    asteroid_gms=STANDARD_ASTEROID_GMS,
    max_steps=jnp.arange(100),
    use_GR=False,
):
    """integrate(xs,vs,gms,initial_time,final_time,planet_params=STANDARD_PLANET_PARAMS,asteroid_params=STANDARD_ASTEROID_PARAMS,planet_gms=STANDARD_PLANET_GMS,asteroid_gms=STANDARD_ASTEROID_GMS,max_steps=jnp.arange(100),use_GR=False,)
    Propagate an N-body system from initial_time to final_time

    This is the fully assembled integrator. It repeatedly calls single_step to
    advance a system from initial_time to final_time, adapting the timestep as it goes.
    It similarly does not do any error checking though, and will return values even if
    a substep has not converged or if it did not reach final_time.

    Note: this function uses lax.cond within lax.scan to conditionally run the
    "expensive" single_step function if it has not yet reached final_time. As a result,
    vmapping this function will likely be very slow, since that would convert the
    lax.cond to a lax.select, which will run single_step even when the endpoint has
    already been reached. 

    Parameters:
        xs (jnp.ndarray(shape=(N, 3))):
            The initial 3D positions of N particles in AU
        vs (jnp.ndarray(shape=(N, 3))):
            The initial 3D velocities of N particles in AU/day
        gms (jnp.ndarray(shape=(N,))):
            The GM values of N particles in AU^3/day^2
        initial_time (float):
            The initial time of the system in TDB JD
        final_time (float):
            The final time of the system in TDB JD
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
        max_steps (jnp.ndarray(shape=(Z,)), default=jnp.arange(100)):
            Any array of length Z, the maximum number of calls to single_step. 
        use_GR (bool, default=False):
            Whether to use the PPN formalism to calculate the gravitational influence of
            the planets. If False, the planets are treated as Newtonian point masses. The
            asteroids are always treated as Newtonian point masses regardless of this flag.

    Returns:
        Tuple[jnp.ndarray(shape=(N, 3)), jnp.ndarray(shape=(N, 3)), float, float, float, bool]:
        xs (jnp.ndarray(shape=(N, 3))):
            The 3D positions of N particles at the end of the integration in AU
        vs (jnp.ndarray(shape=(N, 3))):
            The 3D velocities of N particles at the end of the integration in AU/day
        t (float):
            The final time of the integration in TDB JD. Should equal the final_time input
            if all went well.
        last_dt (float):
            The final timestep used in days. Will likely be small, since final step is
            chosen to exactly land on final_time. Not really used for anything else in the
            package anymore.
        last_big_dt (float):
            The second-to-last timestep used in days. Gives a sense of the steps being used
            throughout the integration. Not really used for anything else in the package
            anymore.
        success (bool):
            Whether the integration "succeeded". This is a pretty crude check- it just makes
            sure that all of the individual single_steps succeeded and that the final time
            was reached.

    Examples:

        Circular motion of a tracer particle in a scale-free system:

        >>> from jorbit.engine import integrate
        >>> import jax.numpy as jnp
        >>> integrate(
        ...     xs=jnp.array([[1.0, 0, 0], [0, 0, 0]]),
        ...     vs=jnp.array([[0.0, 1, 0], [0, 0, 0]]),
        ...     gms=jnp.array([0.0, 1]),
        ...     initial_time=0.0,
        ...     final_time=jnp.pi,
        ...     planet_gms=jnp.array([0]),
        ...     asteroid_gms=jnp.array([0]),
        ... )

        Propagate a pair of main belt asteroids forwards by a month:
        
        >>> from jorbit.engine import integrate
        >>> from jorbit.construct_perturbers): import (
        ...     STANDARD_PLANET_PARAMS,
        ...     STANDARD_ASTEROID_PARAMS,
        ...     STANDARD_PLANET_GMS,
        ...     STANDARD_ASTEROID_GMS,
        ... )
        >>> import jax.numpy as jnp
        >>> from astropy.time import Time
        >>> integrate(
        ...     xs=jnp.array(
        ...         [[0.73291537, -1.85503972, -0.55163327], [0.73291537, -1.85503972, -0.55163327]]
        ...     ),
        ...     vs=jnp.array(
        ...         [[0.0115149, 0.00509674, 0.00161224], [0.0115149, 0.00509674, 0.00161224]]
        ...     ),
        ...     gms=jnp.array([0.0, 0]),
        ...     initial_time=Time("2023-01-01").tdb.jd,
        ...     final_time=Time("2023-03-01").tdb.jd,
        ...     planet_params=STANDARD_PLANET_PARAMS,
        ...     asteroid_params=STANDARD_ASTEROID_PARAMS,
        ...     planet_gms=STANDARD_PLANET_GMS,
        ...     asteroid_gms=STANDARD_ASTEROID_GMS,
        ...     use_GR=True,
        ... )
    """
    # Thin ish wrapper for single_step. Integrates from initial_time to final_time
    # by technically taking 100 steps, but really it's using adaptive timesteps
    # and not calculating any changes if it's already reached its destination

    t = initial_time
    # As = jnp.zeros((xs.shape[0], 8, 3))
    # As = last_As
    dt = 1.0
    last_dt = 1.0
    converge_success = True
    timestep_success = True
    steps_to_converge = 1

    # Feels sketchy, but- to use lax.cond, both true and false funcs need to
    # accept the same inputs. But, the false_func needs to know all of the results from
    # the last step while single_step does not. So, two wrapper functions
    # which take all of the possible inputs and just ignores the ones they don't need
    def true_func(
        xs,
        vs,
        gms,
        dt,
        last_dt,
        t,
        planet_params,
        asteroid_params,
        planet_gms,
        asteroid_gms,
        converge_success,
        timestep_success,
        steps_to_converge,
    ):
        return single_step(
            x0=xs,
            v0=vs,
            gms=gms,
            dt=dt,
            t=t,
            planet_params=planet_params,
            asteroid_params=asteroid_params,
            planet_gms=planet_gms,
            asteroid_gms=asteroid_gms,
            use_GR=use_GR,
        )

    def false_func(
        xs,
        vs,
        gms,
        dt,
        last_dt,
        t,
        planet_params,
        asteroid_params,
        planet_gms,
        asteroid_gms,
        converge_success,
        timestep_success,
        steps_to_converge,
    ):
        return (
            xs,
            vs,
            last_dt,
            dt,
            timestep_success,
            converge_success,
            steps_to_converge,
        )

    def scan_func(carry, scan_over):
        (
            xs,
            vs,
            dt,
            last_dt,
            old_t,
            converge_success,
            timestep_success,
            steps_to_converge,
        ) = carry

        # make sure you're stepping in the right direction
        dt = jnp.where(jnp.sign(dt) == jnp.sign(final_time - initial_time), dt, -dt)
        # make sure you don't overshoot
        dt = jnp.where(
            final_time > initial_time,
            jnp.where(old_t + dt < final_time, dt, final_time - old_t),
            jnp.where(old_t + dt > final_time, dt, final_time - old_t),
        )

        dt = jnp.where(
            jnp.abs(dt) < 12.0, dt, 12.0 * jnp.sign(dt)
        )  # was getting some non-converging predictor/corrector loops with anything larger

        # DO NOT VMAP THIS. The whole point of this scheme is to use lax.cond,
        # which only evaluates one branch. So, if it's already at the final time,
        # it just runs the incredibly cheap false_func w/o doing any integrations.
        # But, if you vmap it, jax will turn it into lax.select and evaluate both branches
        # at every step
        (
            xsn,
            vsn,
            last_dtn,
            dtn,
            converge_successn,
            timestep_successn,
            steps_to_convergen,
        ) = jax.lax.cond(
            jnp.abs(dt) > 1e-18,
            lambda x: true_func(*x),
            lambda x: false_func(*x),
            (
                xs,
                vs,
                gms,
                dt,
                last_dt,
                old_t,
                planet_params,
                asteroid_params,
                planet_gms,
                asteroid_gms,
                converge_success,
                timestep_success,
                steps_to_converge,
            ),
        )

        s = converge_successn * timestep_successn
        xs = jnp.where(s, xsn, xs)
        vs = jnp.where(s, vsn, vs)
        last_dt = jnp.where(s, last_dtn, last_dt)
        converge_success = jnp.where(s, converge_successn, converge_success)
        timestep_success = jnp.where(s, timestep_successn, timestep_success)
        new_t = jnp.where(s, old_t + dt, old_t)
        dt = dtn * 0.75

        return (
            xs,
            vs,
            dt,
            last_dt,
            new_t,
            converge_successn,
            timestep_successn,
            steps_to_convergen,
        ), (converge_success * timestep_success, dt)

    Q = jax.lax.scan(
        scan_func,
        (
            xs,
            vs,
            dt,
            last_dt,
            t,
            converge_success,
            timestep_success,
            steps_to_converge,
        ),
        max_steps,
    )

    xs = Q[0][0]
    vs = Q[0][1]
    last_dt = Q[0][3]
    t = Q[0][4]
    success = (jnp.sum(Q[1][0]) == max_steps.shape[0]) * (t == final_time)
    last_big_dt = Q[1][1][jnp.argmin(Q[1][1]) - 2]

    return xs, vs, t, last_dt, last_big_dt, success


@jit
def integrate_multiple(
    xs,
    vs,
    gms,
    initial_time,
    final_times,
    planet_params=STANDARD_PLANET_PARAMS,
    asteroid_params=STANDARD_ASTEROID_PARAMS,
    planet_gms=STANDARD_PLANET_GMS,
    asteroid_gms=STANDARD_ASTEROID_GMS,
    max_steps=jnp.arange(100),
    use_GR=False,
):
    """integrate_multiple(xs,vs,gms,initial_time,final_times,planet_params=STANDARD_PLANET_PARAMS,asteroid_params=STANDARD_ASTEROID_PARAMS,planet_gms=STANDARD_PLANET_GMS,asteroid_gms=STANDARD_ASTEROID_GMS,max_steps=jnp.arange(100),use_GR=False,)
    Integrate an N-body system to several different times.

    This is a thin wrapper over integrate- it just scans over an array of final_times
    and saves the output of each integration. Useful for propagating a system to each
    of the times an observation was taken.

    Parameters:
        xs (jnp.ndarray(shape=(N, 3))):
            The initial 3D positions of N particles in AU
        vs (jnp.ndarray(shape=(N, 3))):
            The initial 3D velocities of N particles in AU/day
        gms (jnp.ndarray(shape=(N,))):
            The GM values of N particles in AU^3/day^2
        initial_time (float):
            The initial time of the system in TDB JD
        final_times (jnp.ndarray(shape=(M,))):
            The times to integrate to in TDB JD
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
        max_steps (jnp.ndarray(shape=(Z,)), default=jnp.arange(100)):
            Any array of length Z, the maximum number of calls to single_step. 
        use_GR (bool, default=False):
            Whether to use the PPN formalism to calculate the gravitational influence of
            the planets. If False, the planets are treated as Newtonian point masses. The
            asteroids are always treated as Newtonian point masses regardless of this flag.

    Returns:
        Tuple[jnp.ndarray(shape=(N, M, 3)), jnp.ndarray(shape=(N, M, 3)), jnp.ndarray(shape=(M,)), jnp.ndarray(shape=(M,), dtype=bool)]:
        xs: jnp.ndarray(shape=(N, M, 3))
            The 3D positions of N particles at each of the M times in AU
        vs: jnp.ndarray(shape=(N, M, 3))
            The 3D velocities of N particles at each of the M times in AU/day
        final_time: jnp.ndarray(shape=(M,))
            The final times of each integration in TDB JD. Should equal the final_times
            input if nothing failed.
        success: jnp.ndarray(shape=(M,), dytpe=bool)
            Flags for whether each integration "succeeded". See integrate and single_step
            for details.

    Examples:

        Circular motion in a scale-free system:

        >>> from jorbit.engine import integrate_multiple
        >>> import jax.numpy as jnp
        >>> integrate_multiple(
        ...     xs=jnp.array([[1.0, 0, 0], [0, 0, 0]]),
        ...     vs=jnp.array([[0.0, 1, 0], [0, 0, 0]]),
        ...     gms=jnp.array([0.0, 1]),
        ...     initial_time=0.0,
        ...     final_times=jnp.array([0, jnp.pi / 2, jnp.pi, 3 * jnp.pi / 2, 2 * jnp.pi]),
        ...     planet_gms=jnp.array([0]),
        ...     asteroid_gms=jnp.array([0]),
        ... )

        Propagate a pair of main belt asteroids to two times:

        >>> from jorbit.engine import integrate_multiple
        >>> from jorbit.construct_perturbers): import (
        ...     STANDARD_PLANET_PARAMS,
        ...     STANDARD_ASTEROID_PARAMS,
        ...     STANDARD_PLANET_GMS,
        ...     STANDARD_ASTEROID_GMS,
        ... )
        >>> import jax.numpy as jnp
        >>> from astropy.time import Time
        >>> integrate_multiple(
        ...     xs=jnp.array(
        ...         [[0.73291537, -1.85503972, -0.55163327], [0.73291537, -1.85503972, -0.55163327]]
        ...     ),
        ...     vs=jnp.array(
        ...         [[0.0115149, 0.00509674, 0.00161224], [0.0115149, 0.00509674, 0.00161224]]
        ...     ),
        ...     gms=jnp.array([0.0]),
        ...     initial_time=Time("2023-01-01").tdb.jd,
        ...     final_times=jnp.array([Time("2023-01-01").tdb.jd, Time("2023-03-01").tdb.jd]),
        ...     planet_params=STANDARD_PLANET_PARAMS,
        ...     asteroid_params=STANDARD_ASTEROID_PARAMS,
        ...     planet_gms=STANDARD_PLANET_GMS,
        ...     asteroid_gms=STANDARD_ASTEROID_GMS,
        ...     use_GR=True,
        ... )
    """
    def scan_func(carry, scan_over):
        xs, vs, t, last_dt, success = carry
        (
            xs,
            vs,
            t,
            last_dt,
            last_big_dt,
            success,
        ) = integrate(
            xs=xs,
            vs=vs,
            gms=gms,
            initial_time=t,
            final_time=scan_over,
            planet_params=planet_params,
            asteroid_params=asteroid_params,
            planet_gms=planet_gms,
            asteroid_gms=asteroid_gms,
            max_steps=max_steps,
            use_GR=use_GR,
        )
        return (xs, vs, t, last_dt, success), (xs, vs, t, last_dt, success)

    xs, vs, final_time, last_dt, success = jax.lax.scan(
        scan_func,
        (xs, vs, initial_time, 0.1, True),
        final_times,
    )[1]
    xs = jnp.swapaxes(xs, 0, 1)
    vs = jnp.swapaxes(vs, 0, 1)

    return xs, vs, final_time, success


################################################################################
# On-sky functions
################################################################################

@jit
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
                gms=gms,
                dt=nudge,
                t=time,
                planet_params=planet_params,
                asteroid_params=asteroid_params,
                planet_gms=planet_gms,
                asteroid_gms=asteroid_gms,
                use_GR=False,
            )

            xz = Q[0][0]
            return xz, None

        xz = jax.lax.scan(scan_func, x, jnp.arange(3))[0]

        X = xz - observer_position
        calc_ra = jnp.mod(jnp.arctan2(X[1], X[0]) + 2 * jnp.pi, 2 * jnp.pi)
        calc_dec = jnp.pi / 2 - jnp.arccos(X[-1] / jnp.linalg.norm(X, axis=0))
        return calc_ra, calc_dec

    def scan_func(carry, scan_over):
        x, v, time, observer_position = scan_over
        calc_ra, calc_dec = _on_sky(
            x=x,
            v=v,
            time=time,
            observer_position=observer_position,
            planet_params=planet_params,
            asteroid_params=asteroid_params,
            planet_gms=planet_gms,
            asteroid_gms=asteroid_gms,
        )
        return None, (calc_ra, calc_dec)

    return jax.lax.scan(scan_func, None, (xs, vs, times, observer_positions))[1]


@jit
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
# Loglikelihoods
################################################################################


@jit
def negative_loglike_single(
    X,
    times,
    planet_params,
    asteroid_params,
    planet_gms,
    asteroid_gms,
    observer_positions,
    ra,
    dec,
    position_uncertainties,
    max_steps=jnp.arange(100),
):
    x = X[:3]
    v = X[3:6]

    x = x[None, :]
    v = v[None, :]

    xs, vs, final_times, success = integrate_multiple(
        xs=x,
        vs=v,
        gms=jnp.array([0]),
        initial_time=times[0],
        final_times=times[1:],
        planet_params=planet_params,
        asteroid_params=asteroid_params,
        planet_gms=planet_gms,
        asteroid_gms=asteroid_gms,
        max_steps=max_steps,
        use_GR=True,
    )

    xs = jnp.concatenate((x[:, None, :], xs), axis=1)
    vs = jnp.concatenate((v[:, None, :], vs), axis=1)

    calc_RAs, calc_Decs = on_sky(
        xs=xs[0],
        vs=vs[0],
        gms=jnp.array([0]),
        times=times,
        observer_positions=observer_positions,
        planet_params=planet_params,
        asteroid_params=asteroid_params,
        planet_gms=planet_gms,
        asteroid_gms=asteroid_gms,
    )

    err = sky_error(calc_ra=calc_RAs, calc_dec=calc_Decs, true_ra=ra, true_dec=dec)

    sigma2 = position_uncertainties**2
    # p = jnp.log(2 * jnp.pi * sigma2)
    q = -0.5 * jnp.sum(((err**2 / sigma2)))  # + p))
    return -q


@jit
def negative_loglike_single_grad(
    X,
    times,
    planet_params,
    asteroid_params,
    planet_gms,
    asteroid_gms,
    observer_positions,
    ra,
    dec,
    position_uncertainties,
    max_steps=jnp.arange(100),
):
    return jax.grad(negative_loglike_single)(
        X,
        times,
        planet_params,
        asteroid_params,
        planet_gms,
        asteroid_gms,
        observer_positions,
        ra,
        dec,
        position_uncertainties,
        max_steps=max_steps,
    )


@jit
def weave_free_and_fixed(free_mask, free, fixed):
    """
    Combine free and fixed parameters into a single array.

    This is a helper for prepareloglike_helper_input. Combines the free and fixed parameters
    into a single array, making sure they stay in the correct order so that the
    states and GMs of each particle stay together. These "free" and "fixed" parameters
    can be either positions, velocities, or GMs.

    Parameters:
        free_mask (jnp.ndarray(shape=(N,), dtype=bool)):
            A mask marking parameters are free to vary in a fit/likelihood evaluation.
            True means free, False means fixed.
        free (jnp.ndarray(shape=(M,))):
            The current value of the free parameters.
        fixed (jnp.ndarray(shape=(N-M,))):
            The current value of the fixed parameters.

    Returns:
        jnp.ndarray(shape=(N,)):
            The combined free and fixed parameters, with the free parameters in the correct
            order.

    Examples:

        >>> from jorbit.engine import weave_free_and_fixed
        >>> import jax.numpy as jnp
        >>> weave_free_and_fixed(
        ...     free_mask=jnp.array([True, True, False, True, False]),
        ...     free=jnp.array([1.0, 2.0, 3.0]),
        ...     fixed=jnp.array([4.0, 5.0]),
        ... )
    """
    if len(free_mask) == 0:
        return jnp.empty(free.shape)
    if len(fixed) == 0:
        return free
    if len(free) == 0:
        return fixed

    def scan_func(carry, scan_over):
        free_placed = carry
        i = scan_over
        add_free = free_mask[i]
        x = jnp.where(add_free, free[free_placed], fixed[i - free_placed])
        return free_placed + add_free * 1, x

    return jax.lax.scan(scan_func, 0, jnp.arange(len(free_mask)))[1]


@jit
def prepare_loglike_input_helper(
    free_tracer_state_mask,
    free_tracer_xs,
    fixed_tracer_xs,
    free_tracer_vs,
    fixed_tracer_vs,
    free_massive_state_mask,
    free_massive_xs,
    fixed_massive_xs,
    free_massive_vs,
    fixed_massive_vs,
    free_massive_gm_mask,
    free_massive_gms,
    fixed_massive_gms,
    free_planet_gm_mask,
    free_planet_gms,
    fixed_planet_gms,
    free_asteroid_gm_mask,
    free_asteroid_gms,
    fixed_asteroid_gms,
    tracer_particle_times,
    tracer_particle_ras,
    tracer_particle_decs,
    tracer_particle_observer_positions,
    tracer_particle_astrometry_uncertainties,
    massive_particle_times,
    massive_particle_ras,
    massive_particle_decs,
    massive_particle_observer_positions,
    massive_particle_astrometry_uncertainties,
    planet_params,
    asteroid_params,
    max_steps,
    use_GR,
):
    """
    Convert the attributes of a System into a form that can be used by loglike.

    We ultimately want to perform inferences using loglike to evaluate the likelihood of
    our system given some data. However, loglike takes in the full state of the system,
    including bits that we want to remain fixed to certain values and therefore cannot
    influence the likelihood. For examples, we likely do not want to vary the masses of
    the planets included in the ephemeris, or we might want to fix the initial
    state vector to some value and fit for a mass. In these cases, we don't want to 
    differentiate w.r.t. these values, so we need a buffer layer to that combines things
    we want to vary with those we want to fix before passing them all off to loglike.
    That's the purpose of this function: to combine all of the parameters contained
    within a System object into a form that's useful for loglike.

    Parameters:
        free_tracer_state_mask (jnp.ndarray(shape=(N,), dtype=bool)):
            A mask marking which of the N tracer particles parameters have state vectors
            which are allowed to vary.
        free_tracer_xs (jnp.ndarray(shape=(M, 3))):
            The current values of the M free tracer particle positions in AU.
        fixed_tracer_xs (jnp.ndarray(shape=(N-M, 3))):
            The current values of the N-M fixed tracer particle positions in AU.
        free_tracer_vs (jnp.ndarray(shape=(M, 3))):
            The current values of the M free tracer particle velocities in AU/day.
        fixed_tracer_vs (jnp.ndarray(shape=(N-M, 3))):
            The current values of the N-M fixed tracer particle velocities in AU/day.
        free_massive_state_mask (jnp.ndarray(shape=(P,), dtype=bool)):
            A mask marking which of the P massive particles parameters have state vectors
            which are allowed to vary.
        free_massive_xs (jnp.ndarray(shape=(Q, 3))):
            The current values of the Q free massive particle positions in AU.
        fixed_massive_xs (jnp.ndarray(shape=(P-Q, 3))):
            The current values of the P-Q fixed massive particle positions in AU.
        free_massive_vs (jnp.ndarray(shape=(Q, 3))):
            The current values of the Q free massive particle velocities in AU/day.
        fixed_massive_vs (jnp.ndarray(shape=(P-Q, 3))):
            The current values of the P-Q fixed massive particle velocities in AU/day.
        free_massive_gm_mask (jnp.ndarray(shape=(P,), dtype=bool)):
            A mask marking which of the P massive particles parameters have GMs which are
            allowed to vary.
        free_massive_gms (jnp.ndarray(shape=(R,))):
            The current values of the R free massive particle GMs in AU^3/day^2.
        fixed_massive_gms (jnp.ndarray(shape=(P-R,))):
            The current values of the P-R fixed massive particle GMs in AU^3/day^2.
        free_planet_gm_mask (jnp.ndarray(shape=(Y,), dtype=bool)):
            A mask marking which of the Y planets from an ephemeris have GMs which are 
            allowed to vary
        free_planet_gms (jnp.ndarray(shape=(Z,))):
            The current values of the Z free planet GMs in AU^3/day^2.
        fixed_planet_gms (jnp.ndarray(shape=(Y-Z,))):
            The current values of the Y-Z fixed planet GMs in AU^3/day^2.
        free_asteroid_gm_mask (jnp.ndarray(shape=(W,), dtype=bool)):
            A mask marking which of the W asteroids from an ephemeris have GMs which are
            allowed to vary
        free_asteroid_gms (jnp.ndarray(shape=(X,))):
            The current values of the X free asteroid GMs in AU^3/day^2.
        fixed_asteroid_gms (jnp.ndarray(shape=(W-X,))):
            The current values of the W-X fixed asteroid GMs in AU^3/day^2.
        tracer_particle_times (jnp.ndarray(shape=(N,T))):
            The T times the N tracer particles were observed in TDB JD. Will be a sequence
            of T repetitions of the value "2458849" for particles that were not observed-
            see System for more info.
        tracer_particle_ras (jnp.ndarray(shape=(N,T))):
            The T RAs of the N tracer particles in radians. Will be a sequence of zeros for
            particles that were not observed- see System for more info.
        tracer_particle_decs (jnp.ndarray(shape=(N,T))):
            The T Decs of the N tracer particles in radians. Will be a sequence of zeros for
            particles that were not observed- see System for more info.
        tracer_particle_observer_positions (jnp.ndarray(shape=(N,T,3))):
            The T 3D positions of the observers at each observation time in AU. Will be a
            sequence of 999s for particles that were not observed- see System for more info.
        tracer_particle_astrometry_uncertainties (jnp.ndarray(shape=(N,T))):
            The T astrometric uncertainties of the N tracer particles in arcsec. Will be a
            sequence of infs for particles that were not observed- see System for more info.
        massive_particle_times: (jnp.ndarray(shape=(P,D))):
            Analog of free_particle times for the P massive particles each observed D times.
        massive_particle_ras (jnp.ndarray(shape=(P,D))):
            Analog of free_particle ras for the P massive particles each observed D times.
        massive_particle_decs (jnp.ndarray(shape=(P,D))):
            Analog of free_particle decs for the P massive particles each observed D times.
        massive_particle_observer_positions (jnp.ndarray(shape=(P,D,3))):
            Analog of free_particle observer_positions for the P massive particles each
            observed D times.
        massive_particle_astrometry_uncertainties (jnp.ndarray(shape=(P,D))):
            Analog of free_particle astrometry_uncertainties for the P massive particles
            each observed D times.
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
        max_steps (jnp.ndarray(shape=(Z,)), default=jnp.arange(100)):
            Any array of length Z, the maximum number of calls to single_step. 
        use_GR (bool, default=False):
            Whether to use the PPN formalism to calculate the gravitational influence of
            the planets. If False, the planets are treated as Newtonian point masses. The
            asteroids are always treated as Newtonian point masses regardless of this flag.


    Returns:
        dict:
            A dictionary containing all of the inputs needed for loglike.

    
    Examples:

        See the source code for System.loglike
    
    """
    tracer_xs = weave_free_and_fixed(
        free_tracer_state_mask, free_tracer_xs, fixed_tracer_xs
    )
    tracer_vs = weave_free_and_fixed(
        free_tracer_state_mask, free_tracer_vs, fixed_tracer_vs
    )
    massive_xs = weave_free_and_fixed(
        free_massive_state_mask, free_massive_xs, fixed_massive_xs
    )
    massive_vs = weave_free_and_fixed(
        free_massive_state_mask, free_massive_vs, fixed_massive_vs
    )
    massive_gms = weave_free_and_fixed(
        free_massive_gm_mask, free_massive_gms, fixed_massive_gms
    )
    planet_gms = weave_free_and_fixed(
        free_planet_gm_mask, free_planet_gms, fixed_planet_gms
    )
    asteroid_gms = weave_free_and_fixed(
        free_asteroid_gm_mask, free_asteroid_gms, fixed_asteroid_gms
    )

    return {
        "tracer_particle_xs": tracer_xs,
        "tracer_particle_vs": tracer_vs,
        "tracer_particle_times": tracer_particle_times,
        "tracer_particle_ras": tracer_particle_ras,
        "tracer_particle_decs": tracer_particle_decs,
        "tracer_particle_observer_positions": tracer_particle_observer_positions,
        "tracer_particle_astrometry_uncertainties": tracer_particle_astrometry_uncertainties,
        "massive_particle_xs": massive_xs,
        "massive_particle_vs": massive_vs,
        "massive_particle_gms": massive_gms,
        "massive_particle_times": massive_particle_times,
        "massive_particle_ras": massive_particle_ras,
        "massive_particle_decs": massive_particle_decs,
        "massive_particle_observer_positions": massive_particle_observer_positions,
        "massive_particle_astrometry_uncertainties": massive_particle_astrometry_uncertainties,
        "planet_params": planet_params,
        "asteroid_params": asteroid_params,
        "planet_gms": planet_gms,
        "asteroid_gms": asteroid_gms,
        "max_steps": max_steps,
        "use_GR": use_GR,
    }


@jit
def prepare_loglike_input(free_params, fixed_params, use_GR, max_steps):
    """
    A wrapper for prepareloglike_helper_input_helper that deals with dictionaries instead of arrays.

    See prepareloglike_helper_input_helper for more info.

    Parameters:
        free_params (dict):
            A dictionary containing the free parameters of a System.
        fixed_params (dict):
            A dictionary containing the fixed parameters of a System.
        use_GR (bool):
            Whether to use the PPN formalism to calculate the gravitational influence of
            the planets. If False, the planets are treated as Newtonian point masses. The
            asteroids are always treated as Newtonian point masses regardless of this flag.
        max_steps (jnp.ndarray(shape=(S,))):
            Any array of length S, the maximum number of calls to single_step in a single
            integration.

    Returns:
        dict:
            A dictionary containing all of the inputs needed for loglike.

    Examples:
        If system is a System object, then:

        >>> from jorbit.engine import prepareloglike_helper_input
        >>> prepareloglike_helper_input(
        ...    free_params=system._free_params,
        ...    fixed_params=system._fixed_params,
        ...    use_GR=True,
        ...    max_steps=jnp.arange(100)
        ... )    
    """
    return prepare_loglike_input_helper(
        **fixed_params, **free_params, use_GR=use_GR, max_steps=max_steps
    )

@jit
def loglike_helper(
    tracer_particle_xs=jnp.empty((0, 3)),
    tracer_particle_vs=jnp.empty((0, 3)),
    tracer_particle_times=jnp.empty((0, 1)),
    tracer_particle_ras=jnp.empty((0, 1)),
    tracer_particle_decs=jnp.empty((0, 1)),
    tracer_particle_observer_positions=jnp.empty((0, 1, 3)),
    tracer_particle_astrometry_uncertainties=jnp.empty((0, 1)),
    massive_particle_xs=jnp.empty((0, 3)),
    massive_particle_vs=jnp.empty((0, 3)),
    massive_particle_gms=jnp.empty((0)),
    massive_particle_times=jnp.empty((0, 1)),
    massive_particle_ras=jnp.empty((0, 1)),
    massive_particle_decs=jnp.empty((0, 1)),
    massive_particle_observer_positions=jnp.empty((0, 1, 3)),
    massive_particle_astrometry_uncertainties=jnp.empty((0, 1)),
    planet_params=STANDARD_PLANET_PARAMS,
    asteroid_params=STANDARD_ASTEROID_PARAMS,
    planet_gms=STANDARD_PLANET_GMS,
    asteroid_gms=STANDARD_ASTEROID_GMS,
    max_steps=jnp.arange(100),
    use_GR=False,
):
    """loglike_helper(tracer_particle_xs=jnp.empty((0, 3)),tracer_particle_vs=jnp.empty((0, 3)), tracer_particle_times=jnp.empty((0, 1)), tracer_particle_ras=jnp.empty((0, 1)), tracer_particle_decs=jnp.empty((0, 1)), tracer_particle_observer_positions=jnp.empty((0, 1, 3)), tracer_particle_astrometry_uncertainties=jnp.empty((0, 1)), massive_particle_xs=jnp.empty((0, 3)), massive_particle_vs=jnp.empty((0, 3)), massive_particle_gms=jnp.empty((0)), massive_particle_times=jnp.empty((0, 1)), massive_particle_ras=jnp.empty((0, 1)), massive_particle_decs=jnp.empty((0, 1)), massive_particle_observer_positions=jnp.empty((0, 1, 3)), massive_particle_astrometry_uncertainties=jnp.empty((0, 1)), planet_params=STANDARD_PLANET_PARAMS, asteroid_params=STANDARD_ASTEROID_PARAMS, planet_gms=STANDARD_PLANET_GMS, asteroid_gms=STANDARD_ASTEROID_GMS, max_steps=jnp.arange(100), use_GR=False)
    Calculate the log likelihood of a System given some data.

    Parameters:
        tracer_particle_xs (jnp.ndarray(shape=(N, 3), default=jnp.empty((0, 3))):
            The current values of the N tracer particle positions in AU.
        tracer_particle_vs (jnp.ndarray(shape=(N, 3), default=jnp.empty((0, 3))):
            The current values of the N tracer particle velocities in AU/day.
        tracer_particle_times (jnp.ndarray(shape=(N,T), default=jnp.empty((0, 1))):
            The T times the N tracer particles were observed in TDB JD. Every particle must
            have the same number of observations, but padding entries with inf is fine and
            will not affect the log likelihood.
        tracer_particle_ras (jnp.ndarray(shape=(N,T), default=jnp.empty((0, 1))):
            The T RAs of the N tracer particles in radians.
        tracer_particle_decs (jnp.ndarray(shape=(N,T), default=jnp.empty((0, 1))):
            The T Decs of the N tracer particles in radians.
        tracer_particle_observer_positions (jnp.ndarray(shape=(N,T,3), default=jnp.empty((0, 1, 3))):
            The T 3D positions of the observers at each observation time in AU.
        tracer_particle_astrometry_uncertainties (jnp.ndarray(shape=(N,T), default=jnp.empty((0, 1))):
            The T astrometric uncertainties of the N tracer particles in arcsec. To make
            sure padded observations are ignored, dummy values should be set to "inf". This
            is so that division by sigma^2 gives zero, which will not contribute to the sum.
        massive_particle_xs (jnp.ndarray(shape=(P, 3), default=jnp.empty((0, 3))):
            The current values of the P massive particle positions in AU.
        massive_particle_vs (jnp.ndarray(shape=(P, 3), default=jnp.empty((0, 3))):
            The current values of the P massive particle velocities in AU/day.
        massive_particle_gms (jnp.ndarray(shape=(P,), default=jnp.empty((0,))):
            The current values of the P massive particle GMs in AU^3/day^2.
        massive_particle_times (jnp.ndarray(shape=(P,D), default=jnp.empty((0, 1))):
            The D times the P massive particles were observed in TDB JD. Similar to the
            tracer particles, every massive particle must have the same number of
            observations, but padding entries with inf is fine and will not affect the log
            likelihood.
        massive_particle_ras (jnp.ndarray(shape=(P,D), default=jnp.empty((0, 1))):
            The D RAs of the P massive particles in radians.
        massive_particle_decs (jnp.ndarray(shape=(P,D), default=jnp.empty((0, 1))):
            The D Decs of the P massive particles in radians.
        massive_particle_observer_positions (jnp.ndarray(shape=(P,D,3), default=jnp.empty((0, 1, 3))):
            The D 3D positions of the observers at each observation time in AU.
        massive_particle_astrometry_uncertainties (jnp.ndarray(shape=(P,D)jnp.empty((0, 1))
            The D astrometric uncertainties of the P massive particles in arcsec. Similar
            again to the tracers, use inf to pad out observations that don't exist.
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
        max_steps (jnp.ndarray(shape=(Z,)), default=jnp.arange(100)):
            Any array of length Z, the maximum number of calls to single_step. 
        use_GR (bool, default=False):
            Whether to use the PPN formalism to calculate the gravitational influence of
            the planets. If False, the planets are treated as Newtonian point masses. The
            asteroids are always treated as Newtonian point masses regardless of this flag.

    Returns:
        jnp.ndarray(shape=(1,)):
            The log likelihood of the system given the data, assuming a Gaussian likelihood
            with the given (possibly heteroscedastic) astrometric uncertainties.

    Examples:

    >>> from jorbit.engine import loglike_helper
    >>> from jorbit import Observations, Particle
    >>> from jorbit.construct_perturbers import (
    ...      STANDARD_PLANET_PARAMS,
    ...      STANDARD_ASTEROID_PARAMS,
    ...      STANDARD_PLANET_GMS,
    ...      STANDARD_ASTEROID_GMS,
    ...  )
    >>> import jax
    >>> jax.config.update("jax_enable_x64", True)
    >>> import jax.numpy as jnp
    >>> from astropy.time import Time
    >>> import astropy.units as u
    >>> from astropy.coordinates import SkyCoord
    >>> from astroquery.jplhorizons import Horizons
    >>> target = 274301  # MBA (274301) Wikipedia
    >>> time = Time("2023-01-01")
    >>> times = Time(jnp.linspace(time.jd, time.jd + 40, 6), format="jd")
    >>> horizons_query = Horizons(
    ...     id=target,
    ...     location="500@0",  # set the vector origin to the solar system barycenter
    ...     epochs=[time.tdb.jd],
    ... )
    >>> horizons_vectors = horizons_query.vectors(
    ...     refplane="earth"
    ... )  # the refplane argument gives us ICRS-alinged vectors
    >>> true_x0 = jnp.array(
    ...     [horizons_vectors[0]["x"], horizons_vectors[0]["y"], horizons_vectors[0]["z"]]
    ... )
    >>> true_v0 = jnp.array(
    ...     [horizons_vectors[0]["vx"], horizons_vectors[0]["vy"], horizons_vectors[0]["vz"]]
    ... )
    >>> horizons_query = Horizons(
    ...     id=target,
    ...     location="695@399",  # set the observer location to Kitt Peak
    ...     epochs=[t.jd for t in times],
    ... )
    >>> horizons_astrometry = horizons_query.ephemerides(extra_precision=True)
    >>> horizons_positions = jnp.column_stack(
    ...     (horizons_astrometry["RA"].data.filled(), horizons_astrometry["DEC"].data.filled())
    ... )
    >>> obs = Observations(
    ...     positions=SkyCoord(
    ...         horizons_astrometry["RA"].data.filled(),
    ...         horizons_astrometry["DEC"].data.filled(),
    ...         unit=u.deg,
    ...     ),
    ...     times=times,
    ...     observatory_locations="kitt peak",
    ...     astrometry_uncertainties=100 * u.mas,
    ... )
    >>> asteroid = Particle(
    ...     x=true_x0,
    ...     v=true_v0,
    ...     time=time,
    ...     name="wiki",
    ...     observations=obs,
    ...     fit_state=True,
    ...     fit_gm=False,
    ... )
    >>> loglike_helper(
    ...     tracer_particle_xs=asteroid.x[None, :],
    ...     tracer_particle_vs=asteroid.v[None, :],
    ...     tracer_particle_times=obs.times[None, :],
    ...     tracer_particle_ras=obs.ra[None, :],
    ...     tracer_particle_decs=obs.dec[None, :],
    ...     tracer_particle_observer_positions=obs.observer_positions[None, :],
    ...     tracer_particle_astrometry_uncertainties=obs.astrometry_uncertainties[None, :],
    ...     massive_particle_xs=jnp.empty((0, 3)),
    ...     massive_particle_vs=jnp.empty((0, 3)),
    ...     massive_particle_gms=jnp.empty((0)),
    ...     massive_particle_times=jnp.empty((0, 1)),
    ...     massive_particle_ras=jnp.empty((0, 1)),
    ...     massive_particle_decs=jnp.empty((0, 1)),
    ...     massive_particle_observer_positions=jnp.empty((0, 1, 3)),
    ...     massive_particle_astrometry_uncertainties=jnp.empty((0, 1)),
    ...     planet_params=STANDARD_PLANET_PARAMS,
    ...     asteroid_params=STANDARD_ASTEROID_PARAMS,
    ...     planet_gms=STANDARD_PLANET_GMS,
    ...     asteroid_gms=STANDARD_ASTEROID_GMS,
    ...     max_steps=jnp.arange(100),
    ...     use_GR=True,
    ... )
    """
    Q = 0.0

    def false_func(C):
        return 0.0

    def tracer_true_func(C):
        x_, v_, times, observer_pos, ra, dec, pos_uncertainty = C
        x = jnp.concatenate((massive_particle_xs, x_[None, :]))
        v = jnp.concatenate((massive_particle_vs, v_[None, :]))
        xs, vs, final_times, success = integrate_multiple(
            xs=x,
            vs=v,
            gms=jnp.concatenate((massive_particle_gms, jnp.array([0]))),
            initial_time=times[0],
            final_times=times[1:],
            planet_params=planet_params,
            asteroid_params=asteroid_params,
            planet_gms=planet_gms,
            asteroid_gms=asteroid_gms,
            max_steps=max_steps,
            use_GR=use_GR,
        )

        xs = jnp.concatenate((x_[None, None, :], xs[-1][None, :]), axis=1)
        vs = jnp.concatenate((v_[None, None, :], vs[-1][None, :]), axis=1)

        calc_RAs, calc_Decs = on_sky(
            xs=xs[0],
            vs=vs[0],
            gms=jnp.array([0]),
            times=times,
            observer_positions=observer_pos,
            planet_params=planet_params,
            asteroid_params=asteroid_params,
            planet_gms=planet_gms,
            asteroid_gms=asteroid_gms,
        )

        err = sky_error(calc_ra=calc_RAs, calc_dec=calc_Decs, true_ra=ra, true_dec=dec)

        sigma2 = pos_uncertainty**2
        # p = jnp.log(2 * jnp.pi * sigma2)
        q = -0.5 * jnp.sum(((err**2 / sigma2)))  # + p))
        return q

    def tracer_scan_func(carry, scan_over):
        # x_, v_, times, observer_positions, ras, decs, astrometry_uncertainties = scan_over

        # This is a little gross right now- if the very first entry in the list is inf,
        # then there were no observations of this particle and it's not worth integrating/checking error.
        # But, just because the first entry is not zero doesn't mean all of them are-
        # to avoid a ragged array, these are all padded out to the length of whichever particle had the most
        # observations. So, you'll be integrating out to a bunch of dummy times- you only get saved
        # in the final loglike calc since 1/inf = 0, so the bogus times don't contribute to the total loglike.
        # Possibly lots of wasted computation, but hopefully not terrible since a) all the bogus times are the same,
        # so integrating should be instant and b) hopefully all particles have similar number of observations.
        # Worst case scenario would be i.e. one particle has 1,000 observations, 50 particles have 1.
        q = jax.lax.cond(
            scan_over[-1][0] != jnp.inf, tracer_true_func, false_func, scan_over
        )
        return None, q

    def massive_true_func(C):
        ind = C
        xs, vs, final_times, success = integrate_multiple(
            xs=massive_particle_xs,
            vs=massive_particle_vs,
            gms=massive_particle_gms,
            initial_time=massive_particle_times[ind][0],
            final_times=massive_particle_times[ind][1:],
            planet_params=planet_params,
            asteroid_params=asteroid_params,
            planet_gms=planet_gms,
            asteroid_gms=asteroid_gms,
            max_steps=max_steps,
        )

        xs = jnp.concatenate((massive_particle_xs[:, None, :], xs), axis=1)
        vs = jnp.concatenate((massive_particle_vs[:, None, :], vs), axis=1)

        calc_RAs, calc_Decs = on_sky(
            xs=xs[ind],
            vs=vs[ind],
            gms=jnp.array([0]),
            times=massive_particle_times[ind],
            observer_positions=massive_particle_observer_positions[ind],
            planet_params=planet_params,
            asteroid_params=asteroid_params,
            planet_gms=planet_gms,
            asteroid_gms=asteroid_gms,
        )

        err = sky_error(
            calc_ra=calc_RAs,
            calc_dec=calc_Decs,
            true_ra=massive_particle_ras[ind],
            true_dec=massive_particle_decs[ind],
        )

        sigma2 = massive_particle_astrometry_uncertainties[ind] ** 2
        # p = jnp.log(2 * jnp.pi * sigma2)
        q = -0.5 * jnp.sum(((err**2 / sigma2)))  # + p))
        return q

    def massive_scan_func(carry, scan_over):
        q = jax.lax.cond(
            massive_particle_astrometry_uncertainties[scan_over][0] != 999.0,
            massive_true_func,
            false_func,
            scan_over,
        )
        return None, q

    if len(tracer_particle_astrometry_uncertainties) > 0:
        tracer_contribution = jax.lax.scan(
            tracer_scan_func,
            None,
            (
                tracer_particle_xs,
                tracer_particle_vs,
                tracer_particle_times,
                tracer_particle_observer_positions,
                tracer_particle_ras,
                tracer_particle_decs,
                tracer_particle_astrometry_uncertainties,
            ),
        )[1].sum()
    else:
        tracer_contribution = 0.0

    if len(massive_particle_astrometry_uncertainties) > 0:
        massive_contribution = jax.lax.scan(
            massive_scan_func, None, jnp.arange(len(massive_particle_xs))
        )[1].sum()
    else:
        massive_contribution = 0.0

    Q += tracer_contribution
    Q += massive_contribution

    return Q


@jit
def loglike(params):
    """
    A wrapper for loglike_helper that deals with dictionaries instead of arrays.

    See loglike_helper for more info.
    """
    return loglike_helper(**params)


################################################################################
# Element conversion functions
################################################################################


@jit
def _barycentricmeanecliptic_to_icrs(bary_vec):
    return jnp.matmul(BARY_TO_ICRS_ROT_MAT, bary_vec)


@jit
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


@jit
def _icrs_to_barycentricmeanecliptic(icrs_vec):
    return jnp.matmul(ICRS_TO_BARY_ROT_MAT, icrs_vec)


@jit
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


@jit
def cart_to_elements(X, V, time, sun_params=STANDARD_SUN_PARAMS):
    """
    
    """
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


@jit
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

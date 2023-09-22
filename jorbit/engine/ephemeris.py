import jax
from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp


# def perturber_position(
#   init,
#   intlen,
#   coefficients,
#   tdb
# ):
#     pass


# def planet_state_helper(
#     init,
#     intlen,
#     coefficients,
#     tdb,
#     velocity=False,
#     acceleration=False,
# ):
#     """
#     Calculate the position, velocity, and acceleration of one object at a several times.

#     This function borrows heavily from Brandon Rhode's
#     `jplephem <https://github.com/brandon-rhodes/python-jplephem>`_ package,
#     specifically the jplephem.spk.SPK._compute method. However, this version is written
#     for JAX, not numpy. lax.scan is used instead of a for loop, jnp.where is used
#     instead of if statements, etc. Also, in addition to position and velocity, you can
#     also compute acceleration by differentiating the Cheybshev polynomials again. The
#     input parameters are usually derived from the construct_perturbers.py module.

#     Parameters:
#         init (float):
#             The initial time of the ephemeris in seconds since J2000 TDB
#         intlen (float):
#             The length of the interval covered by each piecewise chunk of the ephemeris in
#             seconds. For DE44x, planets, this is 16 days, and for asteroids, it's 32 days.
#         coefficients (jnp.ndarray(shape=(N, 3, M))):
#             The Chebyshev coefficients describing the ephemeris
#             For each of the M equal length piecewise steps, there are N coefficients for
#             each of the 3 {x,y,z} dimensions. For the planets in the DE44x series,
#             N=14 (high order terms zero-padded in some cases),
#             and for asteroids, N=8.
#         tdb: jnp.ndarray(shape=(R,))
#             The times at which to evaluate the ephemeris in TDB JD
#         velocity (bool):
#             Whether to calculate the velocity of the planet
#         acceleration (bool):
#             Whether to calculate the acceleration of the planet. N.B. these values
#             will be nonsense if velocity=False.

#     Returns:
#         Tuple[jnp.ndarray(shape=(R,)), jnp.ndarray(shape=(R,)), jnp.ndarray(shape=(R,))]:
#             The position, velocity, and acceleration of the planet at the requested times.
#             The full tuple is returned, but the velocity and/or acceleration will be 0s if
#             velocity=False and acceleration=False. The full tuple is always returned
#             to help give jax.jit fewer reasons to trigger compilations.
#             Units are AU, AU/day, AU/day^2

#     Notes:
#         The Chebyshev polynomials are evaluated using the `Clenshaw algorithm <https://en.wikipedia.org/wiki/Clenshaw_algorithm>`_.
#         An implementation exists in scipy/numpy, but not in jax as of 0.4.8

#     Examples:
#         >>> from jorbit.engine import planet_state_helper
#         >>> import jax.numpy as jnp
#         >>> from jorbit.construct_perturbers): import STANDARD_PLANET_PARAMS
#         >>> from astropy.time import Time
#         >>> x, v, a = planet_state_helper(
#         ...     init=STANDARD_PLANET_PARAMS[0][0],
#         ...     intlen=STANDARD_PLANET_PARAMS[1][0],
#         ...     coefficients=STANDARD_PLANET_PARAMS[2][0],
#         ...     tdb=jnp.array([Time("2023-01-01").tdb.jd]),
#         ...     velocity=True,
#         ...     acceleration=True,
#         ... )
#     """

#     tdb2 = 0.0  # leaving in case we ever decide to increase the time precision and use 2 floats

#     _, _, n = coefficients.shape

#     # 2451545.0 is the J2000 epoch in TDB
#     index1, offset1 = jnp.divmod((tdb - 2451545.0) * 86400.0 - init, intlen)
#     index2, offset2 = jnp.divmod(tdb2 * 86400.0, intlen)
#     index3, offset = jnp.divmod(offset1 + offset2, intlen)
#     index = (index1 + index2 + index3).astype(int)

#     omegas = index == n
#     index = jnp.where(omegas, index - 1, index)
#     offset = jnp.where(omegas, offset + intlen, offset)

#     coefficients = coefficients[:, :, index]

#     s = 2.0 * offset / intlen - 1.0

#     def eval_cheby(coefficients, x):
#         b_ii = jnp.zeros((3, len(tdb)))
#         b_i = jnp.zeros((3, len(tdb)))

#         def scan_func(X, a):
#             b_i, b_ii = X
#             tmp = b_i
#             b_i = a + 2 * x * b_i - b_ii
#             b_ii = tmp
#             return (b_i, b_ii), b_i

#         Q = jax.lax.scan(scan_func, (b_i, b_ii), coefficients[:-1])
#         b_i, b_ii = Q[0]
#         return coefficients[-1] + s * b_i - b_ii, Q[1]

#     x, As = eval_cheby(coefficients, s)  # in km here

#     # Optionally calculate the velocity by differentiating the Chebyshev polynomials
#     # Will return 0s if velocity=False
#     def blank_vel(As):
#         return jnp.zeros_like(x), (
#             jnp.zeros_like(x),
#             jnp.zeros((As.shape[0] - 1, 3, tdb.shape[0])),
#         )

#     def velocity_func(As):
#         Q = eval_cheby(2 * As, s)
#         v = Q[0] - As[-1]
#         v /= intlen
#         v *= 2.0  # in km/s here
#         return v, Q

#     v, Q = jax.lax.cond(velocity, velocity_func, blank_vel, As)

#     # Optionally calculate the acceleration by differentiating the Chebyshev polynomials again
#     # Note- will give back nonsense if velocity=False
#     # Need to calculate that before the acceleration
#     def blank_acc(Q):
#         return jnp.zeros_like(x)

#     def acceleration_func(Q):
#         a = eval_cheby(4 * Q[1], s)[0] - 2 * Q[1][-1]
#         a /= intlen**2
#         a *= 4.0  # in km/s^2 here
#         return a

#     a = jax.lax.cond(acceleration, acceleration_func, blank_acc, Q)

#     # Convert to AU, AU/day, AU/day^2
#     return (
#         x.T * 6.684587122268446e-09,
#         v.T * 0.0005775483273639937,
#         a.T * 49.900175484249054,
#     )


# def planet_state(
#     planet_params,
#     times,
#     velocity=False,
#     acceleration=False,
# ):
#     """
#     Calculate the position, velocity, and acceleration of several objects at a several times.

#     This is a thin wrapper around planet_state_helper, which is the actual function that does
#     the computation from the ephemeris. Uses jax.vmap to vectorize the computation over
#     multiple objects at once; as a consequence, each object's ephemeris info needs to
#     have the same shape. This is why jorbit.construct_perturbers.construct_perturbers
#     zero pads some higher order terms in the Chebyshev polynomials and/or repeats
#     piecewise chunks of certain objects.

#     Parameters:
#         planet_params (Tuple(jnp.ndarray(shape=(N,)), jnp.ndarray(shape=(N,)), jnp.ndarray(shape=(N, M, 3, R)))):
#             The initial times, interval lengths, and Chebyshev coefficients for N objects.
#             The Chebyshev coefficients are of shape (N, M, 3, R), where M is the number of
#             piecewise chunks, 3 is the {x,y,z} dimensions, and R is the number of
#             coefficients in each chunk.
#         times (jnp.ndarray(shape=(P,))):
#             The times at which to evaluate the ephemerides in TDB JD
#         velocity (bool):, default=False
#             Whether to calculate the velocity of the objects at each time. Must be True if
#             acceleration=True
#         acceleration (bool):, default=False
#             Whether to calculate the acceleration of the objects at each time. N.B.
#             these values will be nonsense if velocity=False.

#     Returns:
#         Tuple[jnp.ndarray(shape=(N, P, 3)), jnp.ndarray(shape=(N, P, 3)), jnp.ndarray(shape=(N, P, 3))]:
#             The positions, velocities, and accelerations of the N objects at the P
#             requested times. The full tuple is always returned, but the velocity and/or
#             acceleration will be 0s depending on the input flags.
#             Units are AU, AU/day, AU/day^2.

#     Examples:
#         >>> from jorbit.engine import planet_state
#         >>> import jax.numpy as jnp
#         >>> from jorbit.construct_perturbers): import STANDARD_PLANET_PARAMS
#         >>> from astropy.time import Time
#         >>> x, v, a = planet_state(
#         ...     planet_params=STANDARD_PLANET_PARAMS,
#         ...     times=jnp.array([Time("2023-01-01").tdb.jd]),
#         ...     velocity=True,
#         ...     acceleration=True,
#         ... )

#     """

#     inits, intlens, coeffs = planet_params
#     return jax.vmap(planet_state_helper, in_axes=(0, 0, 0, None, None, None))(
#         inits, intlens, coeffs, times, velocity, acceleration
#     )


def eval_cheby(coefficients, x):
    b_ii = jnp.zeros((3, x.shape[0]))
    b_i = jnp.zeros((3, x.shape[0]))

    def scan_func(X, a):
        b_i, b_ii = X
        tmp = b_i
        b_i = a + 2 * x * b_i - b_ii
        b_ii = tmp
        return (b_i, b_ii), b_i

    Q = jax.lax.scan(scan_func, (b_i, b_ii), coefficients[:-1])
    b_i, b_ii = Q[0]
    return coefficients[-1] + x * b_i - b_ii, Q[1]


def prep_ephemeris(
    init,
    intlen,
    coefficients,
    tdb,
):
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
    return s, coefficients


def perturber_states(planet_params, times):
    # Compute the x/v/a at multiple times, but only for one perturber at a time
    def single_perturber_states(
        init,
        intlen,
        coefficients,
        tdb,
    ):
        s, coefficients = prep_ephemeris(init, intlen, coefficients, tdb)

        # Position
        x, As = eval_cheby(coefficients, s)  # in km here

        # Velocity
        Q = eval_cheby(2 * As, s)
        v = Q[0] - As[-1]
        v /= intlen
        v *= 2.0  # in km/s here

        # Acceleration
        a = eval_cheby(4 * Q[1], s)[0] - 2 * Q[1][-1]
        a /= intlen**2
        a *= 4.0  # in km/s^2 here

        # Convert to AU, AU/day, AU/day^2
        return (
            x.T * 6.684587122268446e-09,
            v.T * 0.0005775483273639937,
            a.T * 49.900175484249054,
        )

    inits, intlens, coeffs = planet_params
    return jax.vmap(single_perturber_states, in_axes=(0, 0, 0, None))(
        inits, intlens, coeffs, times
    )


def perturber_positions(planet_params, times):
    # Compute the x at multiple times, but only for one perturber at a time
    def single_perturber_positions(
        init,
        intlen,
        coefficients,
        tdb,
    ):
        s, coefficients = prep_ephemeris(init, intlen, coefficients, tdb)

        # Position
        x, As = eval_cheby(coefficients, s)  # in km here

        # Convert to AU
        return x.T * 6.684587122268446e-09

    inits, intlens, coeffs = planet_params
    return jax.vmap(single_perturber_positions, in_axes=(0, 0, 0, None))(
        inits, intlens, coeffs, times
    )

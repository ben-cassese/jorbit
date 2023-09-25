import jax
from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.data.constants import SPEED_OF_LIGHT
from jorbit.engine.ephemeris import perturber_positions, perturber_states


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
        lambda x: jax.vmap(newtonian_helper, in_axes=(0, 1, None))(
            x, planet_xs, planet_gms
        )
    )(xs)


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
    use_GR=True,
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
        # return newtonian(xs=xs, planet_xs=planet_xs, planet_gms=planet_gms)

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

    def true_func():
        ts = jnp.moveaxis(xs, 0, 1)  # (n_times, n_particles, 3)
        tmp = jax.lax.scan(scan_func, None, ts)[1]
        return jnp.moveaxis(tmp, 0, 1)

    def false_func():
        return jnp.zeros_like(A)

    A += jax.lax.cond(jnp.sum(gms) > 0, true_func, false_func)

    return A


def acceleration_at_time(
    x, v, gm, time, planet_params, asteroid_params, planet_gms, asteroid_gms
):
    planet_xs, planet_vs, planet_as = perturber_states(
        planet_params=planet_params, times=jnp.array([time])
    )
    asteroid_xs = perturber_positions(
        planet_params=asteroid_params, times=jnp.array([time])
    )
    return acceleration(
        xs=x[:, None, :],
        vs=v[:, None, :],
        gms=jnp.array([gm]),
        planet_xs=planet_xs,
        planet_vs=planet_vs,
        planet_as=planet_as,
        asteroid_xs=asteroid_xs,
        planet_gms=planet_gms,
        asteroid_gms=asteroid_gms,
        use_GR=True,
    )[:, 0, :]


#     acceleration(
#     xs,
#     vs,
#     gms,
#     planet_xs,
#     planet_vs,
#     planet_as,
#     asteroid_xs,
#     planet_gms=jnp.array([0]),
#     asteroid_gms=jnp.array([0]),
#     use_GR=True,
# )

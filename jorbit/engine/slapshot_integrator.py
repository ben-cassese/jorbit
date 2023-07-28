import jax
from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, lax
import pickle

from jorbit.data import (
    STANDARD_PLANET_PARAMS,
    STANDARD_ASTEROID_PARAMS,
    STANDARD_PLANET_GMS,
    STANDARD_ASTEROID_GMS,
)

from jorbit.data.constants import (
    SLAPSHOT_X_CONSTANT,
    SLAPSHOT_V0_CONSTANT,
    SLAPSHOT_XF_CONSTANT,
    SLAPSHOT_V_CONSTANT,
    SLAPSHOT_VF_CONSTANT,
    SLAPSHOT_B6_CONSTANT,
    SLAPSHOT_H,
)

from jorbit.engine.ephemeris import planet_state
from jorbit.engine.accelerations import acceleration


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
        dt**2
        * (SLAPSHOT_X_CONSTANT[None, :, :, None] * As[:, None, :, :]).sum(axis=2)
        + dt * (v0[:, None, :] * SLAPSHOT_V0_CONSTANT[None, :, None])
        + x0[:, None, :]
    )


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
    return (
        dt**2 * (SLAPSHOT_XF_CONSTANT[None, :, None] * As).sum(axis=1) + dt * v0 + x0
    )


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
        dt * (SLAPSHOT_V_CONSTANT[None, :, :, None] * As[:, None, :, :]).sum(axis=2)
        + v0[:, None, :]
    )


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
    return dt * (SLAPSHOT_VF_CONSTANT[None, :, None] * As).sum(axis=1) + v0


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
    return (SLAPSHOT_B6_CONSTANT[None, :, None] * As).sum(axis=1)


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
    use_GR=True,
):
    """single_step(x0,v0,gms,dt,t,planet_params=STANDARD_PLANET_PARAMS,asteroid_params=STANDARD_ASTEROID_PARAMS,planet_gms=STANDARD_PLANET_GMS,asteroid_gms=STANDARD_ASTEROID_GMS,use_GR=True,)

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
    times = t + SLAPSHOT_H * dt

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
    use_GR=True,
):
    """integrate(xs,vs,gms,initial_time,final_time,planet_params=STANDARD_PLANET_PARAMS,asteroid_params=STANDARD_ASTEROID_PARAMS,planet_gms=STANDARD_PLANET_GMS,asteroid_gms=STANDARD_ASTEROID_GMS,max_steps=jnp.arange(100),use_GR=True,)
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
        Tuple[jnp.ndarray(shape=(N, 3)), jnp.ndarray(shape=(N, 3)), float, float, bool]:
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

    return xs, vs, t, last_dt, success


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
    use_GR=True,
):
    """integrate_multiple(xs,vs,gms,initial_time,final_times,planet_params=STANDARD_PLANET_PARAMS,asteroid_params=STANDARD_ASTEROID_PARAMS,planet_gms=STANDARD_PLANET_GMS,asteroid_gms=STANDARD_ASTEROID_GMS,max_steps=jnp.arange(100),use_GR=True,)
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
        (xs, vs, initial_time, 0.01, True),
        final_times,
    )[1]
    xs = jnp.swapaxes(xs, 0, 1)
    vs = jnp.swapaxes(vs, 0, 1)

    return xs, vs, final_time, success

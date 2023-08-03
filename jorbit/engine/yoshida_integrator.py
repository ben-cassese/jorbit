"""
A leapfrog integrator that does not include corrections for GR.
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.engine.accelerations import acceleration


def _single_step(
    x0, v0, gms, planet_xs, asteroid_xs, planet_gms, asteroid_gms, dt, C, D
):
    """
    Single step of the Yoshida integrator.

    Parameters:
        x0 (jnp.ndarray(shape=(N, 3))):
            The initial 3D positions of N particles in AU
        v0 (jnp.ndarray(shape=(N, 3))):
            The initial 3D velocities of N particles in AU/day
        gms (jnp.ndarray(shape=(N,))):
            The GM values of N particles in AU^3/day^2
        planet_xs (jnp.ndarray(shape=(M, 3))):
            The 3D positions of M planets in AU. This assumes the planets do not move
            significantly over the time step-- we use only one position for each of the
            perturbers, even as the particles leapfrog in time within the one step.
        asteroid_xs (jnp.ndarray(shape=(P, 3))):
            Same as the planet_xs, bu for P asteroids. They are treated the same under
            the hood, since we do not correct for GR effects with this integrator.
        planet_gms (jnp.ndarray(shape=(M,))):
            The GM values of M planets in AU^3/day^2
        asteroid_gms (jnp.ndarray(shape=(P,))):
            Same as the planet_gms, but for P asteroids.
        dt (float):
            The total length of the time step in days.
        C (jnp.ndarry):
            The coefficients for the mid-step position updates. These are pre-computed
            for 4th, 6th, and 8th order integrators and saved in jorbit.data.constants.
            Left as a function input to reuse the same function for different orders.
        D (jnp.ndarray):
            The coefficients for the mid-step velocity updates. Similar to the C
            coefficients.

    Returns:
        Tuple[jnp.ndarray(shape=(N, 3)), jnp.ndarray(shape=(N, 3))]:
        x (jnp.ndarray(shape=(N, 3))):
            The positions of the N particles at the end of the time step. In AU
        v (jnp.ndarray(shape=(N, 3))):
            The velocities of the N particles at the end of the time step. In AU/day
    """

    def leapfrog_scan(X, mid_step_coeffs):
        x, v = X
        c, d = mid_step_coeffs
        x = x + c * v * dt
        acc = acceleration(
            xs=x[:, None, :],
            vs=v[:, None, :],
            gms=gms,
            planet_xs=planet_xs[:, None, :],
            planet_vs=jnp.zeros_like(planet_xs[:, None, :]),
            planet_as=jnp.zeros_like(planet_xs[:, None, :]),
            asteroid_xs=asteroid_xs[:, None, :],
            planet_gms=planet_gms,
            asteroid_gms=asteroid_gms,
            use_GR=False,
        )[:, 0, :]
        v = v + d * acc * dt
        return (x, v), None

    q = jax.lax.scan(leapfrog_scan, (x0, v0), (C[:-1], D))[0]
    x, v = q
    x = x + C[-1] * v * dt
    return x, v


def yoshida_integrate(
    x0, v0, dt, gms, planet_xs, asteroid_xs, planet_gms, asteroid_gms, C, D
):
    """
    Integrate multiple particles in time using a Yoshida leapfrog integrator.

    This assumes the planets do not move significantly within each time step-- we
    use only one position for each of the perturbers per step, even as the particles
    leapfrog within that step. The input planet_xs and asteroid_xs should be offset by
    dt/2 to better represent their positions during the step.

    We include the perturber positions as input rather than calculating it within the
    function, to save computation time when repeatedly integrating over the same time
    span to calculate a likelihood.

    Parameters:
        x0 (jnp.ndarray(shape=(N, 3))):
            The initial 3D positions of N particles in AU
        v0 (jnp.ndarray(shape=(N, 3))):
            The initial 3D velocities of N particles in AU/day
        dt (float):
            The total length of each time step in days
        gms (jnp.ndarray(shape=(N,))):
            The GM values of N particles in AU^3/day^2
        planet_xs (jnp.ndarray(shape=(M, P, 3))):
            The 3D positions of M planets at P times in AU. The P times must be evenly
            spaced by dt.
        asteroid_xs (jnp.ndarray(shape=(Q, P, 3))):
            Same as the planet_xs, bu for Q asteroids. They are treated the same under
            the hood, since we do not correct for GR effects with this integrator.
        planet_gms (jnp.ndarray(shape=(M,))):
            The GM values of M planets in AU^3/day^2
        asteroid_gms (jnp.ndarray(shape=(Q,))):
            Same as the planet_gms, but for Q asteroids.
        C (jnp.ndarry):
            The coefficients for the mid-step position updates. These are pre-computed
            for 4th, 6th, and 8th order integrators and saved in jorbit.data.constants.
            Left as a function input to reuse the same function for different orders.
        D (jnp.ndarray):
            The coefficients for the mid-step velocity updates. Similar to the C
            coefficients.

    Returns:
        Tuple[jnp.ndarray(shape=(N, 3)), jnp.ndarray(shape=(N, 3))]:
        x (jnp.ndarray(shape=(N, 3))):
            The positions of the N particles at the end of the time step. In AU
        v (jnp.ndarray(shape=(N, 3))):
            The velocities of the N particles at the end of the time step. In AU/day

    Examples:

        >>> from astropy.time import Time
        >>> import astropy.units as u
        >>> import jax.numpy as jnp
        >>> from jorbit.data import (
        ...     STANDARD_PLANET_PARAMS,
        ...     STANDARD_ASTEROID_PARAMS,
        ...     STANDARD_PLANET_GMS,
        ...     STANDARD_ASTEROID_GMS,
        ... )
        >>> from jorbit.data.constants import EXAMPLE_X0, EXAMPLE_V0, EXAMPLE_XF, Y8_C, Y8_D
        >>> from jorbit.engine.yoshida_integrator import prep_leapfrog_ntegrator_single, yoshida_integrate
        >>> planet_xs, asteroid_xs, dt = prep_leapfrog_ntegrator_single(
        ...     t0=Time("2023-01-01").tdb.jd,
        ...     tf=Time("2023-06-01").tdb.jd,
        ...     steps=10_000,
        ...     planet_params=STANDARD_PLANET_PARAMS,
        ...     asteroid_params=STANDARD_ASTEROID_PARAMS,
        ... )
        >>> x, v = yoshida_integrate(
        ...     x0=jnp.array([EXAMPLE_X0]),
        ...     v0=jnp.array([EXAMPLE_V0]),
        ...     dt=dt,
        ...     gms=jnp.array([0.0]),
        ...     planet_xs=planet_xs,
        ...     asteroid_xs=asteroid_xs,
        ...     planet_gms=STANDARD_PLANET_GMS,
        ...     asteroid_gms=STANDARD_ASTEROID_GMS,
        ...     C=Y8_C,
        ...     D=Y8_D,
        ... )
        >>> print(jnp.linalg.norm(x - EXAMPLE_XF) * u.au.to(u.m))

    """

    def scan_func(carry, scan_over):
        return (
            _single_step(
                x0=carry[0],
                v0=carry[1],
                gms=gms,
                planet_xs=scan_over[0],
                asteroid_xs=scan_over[1],
                planet_gms=planet_gms,
                asteroid_gms=asteroid_gms,
                dt=dt,
                C=C,
                D=D,
            ),
            None,
        )

    planet_xs = jnp.swapaxes(planet_xs, 0, 1)
    asteroid_xs = jnp.swapaxes(asteroid_xs, 0, 1)
    return jax.lax.scan(scan_func, (x0, v0), (planet_xs, asteroid_xs))[0]


def yoshida_integrate_multiple(
    x0, v0, gms, dts, planet_xs, asteroid_xs, planet_gms, asteroid_gms, C, D
):
    """
    Integrate multiple particle to multiple times using a Yoshida leapfrog integrator.

    This loops over yoshida_integrate to move a system along to a series of time. Note:
    it takes the *same number of steps* to get between each time *regardless of size of
    the gap between times*. e.g., you if want the state of the system at t+1 and t+101,
    and you take 50 substeps between each, your actual step size will be 1/50 days for
    the first integration, then 2 days for the second.

    Parameters:
        x0 (jnp.ndarray(shape=(N, 3))):
            The initial 3D positions of N particles in AU
        v0 (jnp.ndarray(shape=(N, 3))):
            The initial 3D velocities of N particles in AU/day
        gms (jnp.ndarray(shape=(N,))):
            The GM values of N particles in AU^3/day^2
        dts (jnp.ndarray(shape=(T,))):
            The time spacing between each substep for each of the T integrations
        planet_xs (jnp.ndarray(shape=(T, M, S, 3))):
            The 3D positions of M planets, at S evenly spaced substeps, for each of the
            T times. Units are in AU. The time spacing between each substep is even
            within each T but can vary between them.
        asteroid_xs (jnp.ndarray(shape=(T, P, S, 3))):
            Same as the planet_xs, but for P asteroids. Must have the same number of S
            substeps and T times as the planet_xs.
        planet_gms (jnp.ndarray(shape=(M,))):
            The GM values of M planets in AU^3/day^2
        asteroid_gms (jnp.ndarray(shape=(P,))):
            Same as the planet_gms, but for P asteroids.
        C (jnp.ndarry):
            The coefficients for the mid-step position updates. These are pre-computed
            for 4th, 6th, and 8th order integrators and saved in jorbit.data.constants
        D (jnp.ndarray):
            The coefficients for the mid-step velocity updates. Similar to the C
            coefficients.

    Returns:
        Tuple[jnp.ndarray(shape=(T, N, 3)), jnp.ndarray(shape=(T, N, 3))]:
        xs: The positions of the N particles at T times. Units are in AU.
        vs: The velocities of the N particles at T times. Units are in AU/day.

    Examples:

        Two particles, central massive:

        >>> x = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        >>> v = jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        >>> yoshida_integrate_multiple(
        ...     x0=x,
        ...     v0=v,
        ...     t0=0.,
        ...     times=jnp.array([jnp.pi / 2, 3 * jnp.pi / 2, 2 * jnp.pi, 5 * jnp.pi / 2]),
        ...     gms=jnp.array([0., 1]),
        ...     planet_xs=jnp.zeros((4, 1, 100, 3)),
        ...     asteroid_xs=jnp.zeros((4, 1, 100, 3)),
        ...     planet_gms=jnp.array([0]),
        ...     asteroid_gms=jnp.array([0]),
        ...     C=Y8_C,
        ...     D=Y8_D,
        ... )

        One particle, fixed "sun":

        >>> x = jnp.array([[1.0, 0.0, 0.0]])
        >>> v = jnp.array([[0.0, 1.0, 0.0]])
        >>> yoshida_integrate_multiple(
        ...     x0=x,
        ...     v0=v,
        ...     t0=0.,
        ...     times=jnp.array([jnp.pi / 2, 3 * jnp.pi / 2, 2 * jnp.pi, 5 * jnp.pi / 2]),
        ...     gms=jnp.array([0, 1]),
        ...     planet_xs=jnp.zeros((4, 1, 100, 3)),
        ...     asteroid_xs=jnp.zeros((4, 1, 100, 3)),
        ...     planet_gms=jnp.array([1]),
        ...     asteroid_gms=jnp.array([0]),
        ...     C=Y8_C,
        ...     D=Y8_D,
        ... )

        A realistic main belt asteroid:

        >>> import jax.numpy as jnp
        >>> from astropy.time import Time
        >>> import astropy.units as u
        >>> from astroquery.jplhorizons import Horizons
        >>> from jorbit.data import (
        >>>     STANDARD_PLANET_PARAMS,
        >>>     STANDARD_ASTEROID_PARAMS,
        >>>     STANDARD_PLANET_GMS,
        >>>     STANDARD_ASTEROID_GMS,
        >>> )
        >>> from jorbit.data.constants import Y8_C, Y8_D
        >>> from jorbit.engine.yoshida_integrator import (
        ...     prep_leapfrog_integrator_multiple,
        ...     yoshida_integrate_multiple,
        ... )
        >>> times = Time(["2023-04-08", "2023-06-08", "2023-09-01"])
        >>> target = 274301  # MBA (274301) Wikipedia
        >>> horizons_query = Horizons(
        ...     id=target,
        ...     location="500@0",
        ...     epochs=[t.tdb.jd for t in times],
        ... )
        >>> horizons_vectors = horizons_query.vectors(refplane="earth")
        >>> tx0 = jnp.array(
        ...     [horizons_vectors[0]["x"], horizons_vectors[0]["y"], horizons_vectors[0]["z"]]
        ... )
        >>> tv0 = jnp.array(
        ...     [horizons_vectors[0]["vx"], horizons_vectors[0]["vy"], horizons_vectors[0]["vz"]]
        ... )
        >>> txq = jnp.array(
        ...     [horizons_vectors[1]["x"], horizons_vectors[1]["y"], horizons_vectors[1]["z"]]
        ... )
        >>> tvq = jnp.array(
        ...     [horizons_vectors[1]["vx"], horizons_vectors[1]["vy"], horizons_vectors[1]["vz"]]
        ... )
        >>> txf = jnp.array(
        ...     [horizons_vectors[-1]["x"], horizons_vectors[-1]["y"], horizons_vectors[-1]["z"]]
        ... )
        >>> tvf = jnp.array(
        ...     [horizons_vectors[-1]["vx"], horizons_vectors[-1]["vy"], horizons_vectors[-1]["vz"]]
        ... )
        >>> planet_xs, asteroid_xs, dts = prep_leapfrog_integrator_multiple(
        ...     t0=times[0].tdb.jd,
        ...     times=jnp.array([t.tdb.jd for t in times][1:]),
        ...     steps=1_000,
        ...     planet_params=STANDARD_PLANET_PARAMS,
        ...     asteroid_params=STANDARD_ASTEROID_PARAMS,
        ...     offset=0,
        ... )
        >>> a = yoshida_integrate_multiple(
        ...     x0=jnp.array([tx0]),
        ...     v0=jnp.array([tv0]),
        ...     dts=dts,
        ...     gms=jnp.array([0.0]),
        ...     planet_xs=planet_xs,
        ...     asteroid_xs=asteroid_xs,
        ...     planet_gms=STANDARD_PLANET_GMS,
        ...     asteroid_gms=STANDARD_ASTEROID_GMS,
        ...     C=Y8_C,
        ...     D=Y8_D,
        ... )
        >>> print(jnp.linalg.norm(a[0][0, 0] - txq) * u.au.to(u.m))
        >>> print(jnp.linalg.norm(a[0][0, 1] - txf) * u.au.to(u.m))

    """

    def scan_func(carry, scan_over):
        x, v = carry
        dt, planet_x, asteroid_x = scan_over
        x, v = yoshida_integrate(
            x0=x,
            v0=v,
            dt=dt,
            gms=gms,
            planet_xs=planet_x,
            asteroid_xs=asteroid_x,
            planet_gms=planet_gms,
            asteroid_gms=asteroid_gms,
            C=C,
            D=D,
        )
        return (x, v), (x, v)

    xs, vs = jax.lax.scan(scan_func, (x0, v0), (dts, planet_xs, asteroid_xs))[1]
    xs = jnp.swapaxes(xs, 0, 1)
    vs = jnp.swapaxes(vs, 0, 1)
    return xs, vs

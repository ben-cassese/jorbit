import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jorbit.engine.ephemeris import planet_state
from jorbit.engine.accelerations import acceleration
from jorbit.data.constants import EPSILON

from jorbit.data import (
    STANDARD_PLANET_PARAMS,
    STANDARD_ASTEROID_PARAMS,
    STANDARD_PLANET_GMS,
    STANDARD_ASTEROID_GMS,
)


def _create_yoshida_coeffs(Ws):
    """
    Convert the Ws from Tables 1 and 2 of Yoshida (1990) into C and D coefficients

    Saving this for later reference, but it isn't called anymore- values were
    precomputed and saved in jorbit.data.constants.
    """
    w0 = 1 - 2 * (jnp.sum(Ws))
    w = jnp.concatenate((jnp.array([w0]), Ws))

    Ds = jnp.zeros(2 * len(Ws) + 1)
    Ds = Ds.at[: len(Ws)].set(Ws[::-1])
    Ds = Ds.at[len(Ws)].set(w0)
    Ds = Ds.at[len(Ws) + 1 :].set(Ws)

    Cs = jnp.zeros(2 * len(Ws) + 2)
    for i in range(len(w) - 1):
        Cs = Cs.at[i + 1].set(0.5 * (w[len(w) - 1 - i] + w[len(w) - 2 - i]))

    Cs = Cs.at[int(len(Cs) / 2) :].set(Cs[: int(len(Cs) / 2)][::-1])
    Cs = Cs.at[0].set(0.5 * w[-1])
    Cs = Cs.at[-1].set(0.5 * w[-1])

    # to do it at extended precision, use Decimal:
    # tmp = 0
    # for i in Ws:
    #     tmp += i
    # w0 = 1 - 2 * tmp
    # w = [w0] + Ws

    # Ds = [0]*(2 * len(Ws) + 1)
    # Ds[:len(Ws)] = Ws[::-1]
    # Ds[len(Ws)] = w0
    # Ds[len(Ws) + 1:] = Ws

    # Cs = [0]*(2 * len(Ws) + 2)
    # for i in range(len(w) - 1):
    #     Cs[i + 1] = Decimal(0.5) * (w[len(w) - 1 - i] + w[len(w) - 2 - i])
    # Cs[int(len(Cs) / 2):] = Cs[: int(len(Cs) / 2)][::-1]
    # Cs[0] = Decimal(0.5) * w[-1]
    # Cs[-1] = Decimal(0.5) * w[-1]

    return jnp.array(Cs), jnp.array(Ds)


def single_step(
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


def prep_single_integrator(
    t0,
    tf,
    steps,
    planet_params=STANDARD_PLANET_PARAMS,
    asteroid_params=STANDARD_ASTEROID_PARAMS,
):
    times = jnp.linspace(t0, tf, steps)
    dt = jnp.diff(times)[0]

    times += dt / 2  # offset the planet positions so they are at the mid-step point.
    # should be a tiny correction for small steps/large distances

    planet_xs, _, _ = planet_state(
        planet_params=planet_params, times=times, velocity=False, acceleration=False
    )
    asteroid_xs, _, _ = planet_state(
        planet_params=asteroid_params, times=times, velocity=False, acceleration=False
    )

    return planet_xs, asteroid_xs, dt


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


    """

    def scan_func(carry, scan_over):
        return (
            single_step(
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
    x0, v0, gms, t0, times, planet_xs, asteroid_xs, planet_gms, asteroid_gms, C, D
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
        t0 (float):
            The initial time in TDB JD
        times (jnp.ndarray(shape=(T,))):
            The T times to propagate the system to in TDB JD. Does not include t0
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
    """
    times = jnp.concatenate((jnp.array([t0]), times))
    steps = planet_xs.shape[2]
    dts = jnp.diff(times) / (steps)

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

    return jax.lax.scan(scan_func, (x0, v0), (dts, planet_xs, asteroid_xs))[1][0]

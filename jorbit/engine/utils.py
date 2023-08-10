"""
The one module within jorbit.engine where the functions do not have to be jax/jit/grad
compatible. Functions here should only have to be run once per system, and outputs
should be able to be reused for every subsequent likelihood call or integration to the
same epochs.
"""

import jax
from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jplephem.spk import SPK
import astropy.units as u
from astropy.time import Time
from astropy.utils.data import download_file
import pandas as pd
from tqdm import tqdm

# from tqdm.notebook import tqdm

# jorbit imports:
from jorbit.engine.ephemeris import planet_state

from jorbit.data import (
    STANDARD_PLANET_PARAMS,
    STANDARD_ASTEROID_PARAMS,
    STANDARD_PLANET_GMS,
    STANDARD_ASTEROID_GMS,
)

from jorbit.data.constants import (
    planet_ephemeris,
    asteroid_ephemeris,
    all_planets,
    all_planet_nums,
    all_planet_gms,
    large_asteroids,
    large_asteroid_nums,
    large_asteroid_gms,
)

########################################################################################
# Ephemeris helpers
########################################################################################


def construct_perturbers(
    planets=all_planets,
    asteroids=large_asteroids,
    earliest_time=Time("1980-01-01"),
    latest_time=Time("2100-01-01"),
):
    assert earliest_time.tdb.jd < latest_time.tdb.jd

    assert earliest_time.tdb.jd > 2287184.5, (
        "The DE440 ephemeris only covers between 1549-12-31 and 2650-01-25. Please"
        " adjust earliest and latest times."
    )
    assert latest_time.tdb.jd < 2688976.5, (
        "The DE440 ephemeris only covers between 1549-12-31 and 2650-01-25. Please"
        " adjust earliest and latest times."
    )

    assert isinstance(planets, list)
    assert "sun" not in planets
    assert len(list(set(planets))) == len(planets)
    planets = ["sun"] + planets  # Should always be first

    assert isinstance(asteroids, list)
    assert len(list(set(planets))) == len(planets)

    planet_ids = []
    planet_gms = []
    for p in planets:
        try:
            planet_ids.append(all_planet_nums[p.lower()])
            planet_gms.append(all_planet_gms[p.lower()])
        except:
            print(f"{p} not a valid planet name, not included in integration")

    kernel = SPK.open(download_file(planet_ephemeris, cache=True))

    planet_init = []
    planet_intlen = []
    planet_coeffs = []
    for id in planet_ids:
        for seg in kernel.segments:
            if seg.target != id:
                continue
            init, intlen, coeff = seg._data
            planet_init.append(jnp.array(init))
            planet_intlen.append(jnp.array(intlen))
            planet_coeffs.append(jnp.array(coeff))

    # ***
    # Was previously using the large asteroids file to allow for outer solar system,
    # but swapping for now to the n16 file and GMs from the same source as the planets
    # to compare more directly to ASSIST
    # ***
    # ############################################################################
    # # table created from:
    # # https://ssd.jpl.nasa.gov/ftp/eph/small_bodies/asteroids_de441/SB441_IOM392R-21-005_perturbers.pdf

    # asteroid_info = pd.read_csv('data/Asteroid_Perturbers.csv')
    # ids = []
    # for body in asteroids:
    #     try:
    #         if isinstance(body, str):
    #             body = body.lower()
    #             ids.append(asteroid_info[asteroid_info['Name/Provisional designation'] == body]['Spice ID'].values[0])
    #             gms.append(asteroid_info[asteroid_info['Name/Provisional designation'] == body]['GM - au3 /d2'].values[0])
    #         elif isinstance(body, int):
    #             ids.append(asteroid_info[asteroid_info['Number'] == body]['Spice ID'].values[0])
    #             gms.append(asteroid_info[asteroid_info['Number'] == body]['GM - au3 /d2'].values[0])
    #     except:
    #         print(f'{body} not a valid asteroid name or number, not included in integration')

    # asteroid_ephem = 'https://ssd.jpl.nasa.gov/ftp/eph/small_bodies/asteroids_de441/sb441-n373.bsp'
    # kernel = SPK.open(download_file(asteroid_ephem, cache=True))
    if len(asteroids) == 0:
        asteroids = ["ceres"]
        no_asteroids = True
    else:
        no_asteroids = False

    asteroid_ids = []
    asteroid_gms = []
    for p in asteroids:
        try:
            asteroid_ids.append(large_asteroid_nums[p.lower()])
            asteroid_gms.append(large_asteroid_gms[p.lower()])
        except:
            print(f"{p} not a valid asteroid name, not included in integration")
    # asteroid_ephem = 'https://ssd.jpl.nasa.gov/ftp/eph/small_bodies/asteroids_de441/sb441-n373.bsp'
    kernel = SPK.open(download_file(asteroid_ephemeris, cache=True))

    asteroid_init = []
    asteroid_intlen = []
    asteroid_coeffs = []
    for id in asteroid_ids:
        for seg in kernel.segments:
            if seg.target != id:
                continue
            if seg.start_jd != 1999474.5:
                continue
            init, intlen, coeff = seg._data
            asteroid_init.append(jnp.array(init))
            asteroid_intlen.append(jnp.array(intlen))
            asteroid_coeffs.append(jnp.array(coeff))

    ############################################################################
    # the vectorizing rewrite

    planet_init, planet_intlen, planet_coeffs = _construct_perturbers_helper(
        Init=jnp.array(planet_init),
        Intlen=jnp.array(planet_intlen),
        Coeff=planet_coeffs,
        earliest_time=earliest_time.tdb.jd - 60,
        latest_time=latest_time.tdb.jd + 60,
    )

    asteroid_init, asteroid_intlen, asteroid_coeffs = _construct_perturbers_helper(
        Init=jnp.array(asteroid_init),
        Intlen=jnp.array(asteroid_intlen),
        Coeff=asteroid_coeffs,
        earliest_time=earliest_time.tdb.jd - 60,
        latest_time=latest_time.tdb.jd + 60,
    )
    if no_asteroids:
        asteroid_init *= 0
        asteroid_intlen *= 0
        asteroid_coeffs *= 0
        asteroid_gms = [0]
    ############################################################################
    # gms = planet_gms + asteroid_gms
    assert len(planet_gms) + len(asteroid_gms) == len(planet_init) + len(asteroid_init)
    return (
        (planet_init, planet_intlen, planet_coeffs),
        (asteroid_init, asteroid_intlen, asteroid_coeffs),
        jnp.array(planet_gms),
        jnp.array(asteroid_gms),
    )


def _construct_perturbers_helper(Init, Intlen, Coeff, earliest_time, latest_time):
    init0 = Init[0]
    for i in Init:
        assert i == init0

    # Trim the timespans down to the earliest and latest times
    longest_intlen = jnp.max(Intlen)
    ratios = longest_intlen / Intlen
    early_indecies = []
    late_indecies = []
    for i in range(len(Init)):
        component_count, coefficient_count, n = Coeff[i].shape
        index, offset = jnp.divmod(
            (earliest_time - 2451545.0) * 86400.0 - Init[i], Intlen[i]
        )
        omegas = index == n
        index = jnp.where(omegas, index - 1, index)
        early_indecies.append(index)

        index, offset = jnp.divmod(
            (latest_time - 2451545.0) * 86400.0 - Init[i], Intlen[i]
        )
        omegas = index == n
        index = jnp.where(omegas, index - 1, index)
        late_indecies.append(index)

    early_indecies = (
        jnp.ones(len(early_indecies)) * jnp.min(jnp.array(early_indecies)) * ratios
    ).astype(int)
    new_inits = Init + early_indecies * Intlen
    late_indecies = (
        jnp.ones(len(late_indecies)) * jnp.min(jnp.array(late_indecies)) * ratios
    ).astype(int)
    trimmed_coeffs = []
    for i in range(len(Init)):
        trimmed_coeffs.append(Coeff[i][:, :, early_indecies[i] : late_indecies[i]])

    # Add extra Chebyshev coefficients (zeros) to make the number of
    # coefficients at each time slice the same across all planets
    coeff_shapes = []
    for i in trimmed_coeffs:
        coeff_shapes.append(i.shape)
    coeff_shapes = jnp.array(coeff_shapes)
    most_coefficients, _, most_time_slices = jnp.max(coeff_shapes, axis=0)

    padded_coefficients = []
    for c in trimmed_coeffs:
        c = jnp.pad(c, ((most_coefficients - c.shape[0], 0), (0, 0), (0, 0)))
        padded_coefficients.append(c)

    # This is a little sketchy- tile each planet so that they all have the same
    # number of time slices. This means that for planets with longer original intlens,
    # we could technically feed in times outside the original timespan and get a false result
    # But, by keeping their original intlens intact, if we feed in a time within
    # the timespan, we should just always stay in the first half, quarter, whatever
    shortest_intlen = jnp.min(Intlen)
    padded_intlens = jnp.ones(len(Intlen)) * shortest_intlen
    extra_padded = []
    for i in range(len(padded_coefficients)):
        extra_padded.append(
            jnp.tile(padded_coefficients[i], int(Intlen[i] / shortest_intlen))
        )

    new_coeff = jnp.array(extra_padded)

    return new_inits, Intlen, new_coeff


########################################################################################
# Leapfrog integrator helpers
########################################################################################


def _create_yoshida_coeffs(Ws):
    """
    Convert the Ws from Tables 1 and 2 of Yoshida (1990) into C and D coefficients

    Saving this for later reference, but it isn't called anymore- values were
    precomputed and saved in jorbit.data.constants.

    Parameters:
        WS (jnp.ndarray):
            An array of "W" values from Tables 1 and 2 of Yoshida (1990)

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]:
        C (jnp.ndarray):
            The coefficients for the mid-step position updates
        D (jnp.ndarray):
            The coefficients for the mid-step velocity updates
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


def prep_leapfrog_integrator_single(
    t0,
    tf,
    steps,
    planet_params=STANDARD_PLANET_PARAMS,
    asteroid_params=STANDARD_ASTEROID_PARAMS,
    offset=0,
):
    """
    Create the inputs to run the leapfrog integrator between a single time span.

    Parameters:
        t0 (float):
            The initial time in TDB JD
        tf (float):
            The final time in TDB JD
        steps (int):
            The number of steps to take between t0 and tf
        planet_params (Tuple[jnp.ndarray(shape=(P,)), jnp.ndarray(shape=(P,)), jnp.ndarray(shape=(P,Q,3,R))], default=STANDARD_PLANET_PARAMS from jorbit.data):
            The ephemeris describing P massive objects in the solar system. The first
            element is the initial time of the ephemeris in seconds since J2000 TDB. The
            second element is the length of the interval covered by each piecewise chunk of
            the ephemeris in seconds (for DE44x planets, this is 16 days, and for
            asteroids, it's 32 days). The third element contains the Q coefficients of the
            R piecewise chunks of Chebyshev polynomials that make up the ephemeris, in 3
            x,y,z dimensions.
        asteroid_params (Tuple[jnp.ndarray(shape=(Q,)), jnp.ndarray(shape=(Q,)), jnp.ndarray(shape=(Q,Q,3,R))], default=STANDARD_ASTEROID_PARAMS from jorbit.data):
            Same as the planet_params, but for Q asteroids.
        offset (float, default=0):
            Experimental. Offset the time to evaluate the planet positions so that
            they're more representative over a time step. Not currently used.

    Returns:
        Tuple[jnp.ndarray(shape=(P, steps, 3)), jnp.ndarray(shape=(Q, steps 3)), float]:
        planet_xs (jnp.ndarray(shape=(P, steps, 3))):
            The 3D positions of P planets at steps times. Units are in AU.
        asteroid_xs (jnp.ndarray(shape=(Q, steps, 3))):
            The 3D positions of Q asteroids at steps times. Units are in AU.
        dt (float):
            The length of each time step in days.

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

    """
    times = jnp.linspace(t0, tf, steps)
    dt = (tf - t0) / steps

    times += offset
    # offset the planet positions so they are at the mid-step point, something like
    # dt / 2. Played around with this and something smaller, on the order of 0.003 days,
    # almost always improves the accurary compared to Horizons. can't see why though,
    # leaving it out for now. worth coming back to though, I'm guesssing it's something
    # with time scale conversions

    planet_xs, _, _ = planet_state(
        planet_params=planet_params, times=times, velocity=False, acceleration=False
    )
    asteroid_xs, _, _ = planet_state(
        planet_params=asteroid_params, times=times, velocity=False, acceleration=False
    )

    return planet_xs, asteroid_xs, dt


def prep_leapfrog_integrator_multiple(
    t0,
    times,
    steps,
    planet_params=STANDARD_PLANET_PARAMS,
    asteroid_params=STANDARD_ASTEROID_PARAMS,
    offset=0,
):
    """
    Prepare the integrator to integrate through multiple epochs.

    Parameters:
        t0 (float):
            The initial time in TDB JD
        times (jnp.ndarray(shape=(T,))):
            The times to integrate to, in TDB JD
        steps (int):
            The number of steps to take between each time. The number of steps taken to
            jump between epochs will remain constant- the actual size of those steps,
            in units like days, will vary depending on the size of the gap between
            epochs. Take care that the resulting dts are not too large, insert
            additional intermediate epochs if needed
        planet_params (Tuple[jnp.ndarray(shape=(P,)), jnp.ndarray(shape=(P,)), jnp.ndarray(shape=(P,Q,3,R))], default=STANDARD_PLANET_PARAMS from jorbit.data):
            The ephemeris describing P massive objects in the solar system. The first
            element is the initial time of the ephemeris in seconds since J2000 TDB. The
            second element is the length of the interval covered by each piecewise chunk of
            the ephemeris in seconds (for DE44x planets, this is 16 days, and for
            asteroids, it's 32 days). The third element contains the Q coefficients of the
            R piecewise chunks of Chebyshev polynomials that make up the ephemeris, in 3
            x,y,z dimensions.
        asteroid_params (Tuple[jnp.ndarray(shape=(Q,)), jnp.ndarray(shape=(Q,)), jnp.ndarray(shape=(Q,Q,3,R))], default=STANDARD_ASTEROID_PARAMS from jorbit.data):
            Same as the planet_params, but for Q asteroids.
        offset (float, default=0):
            Experimental. See prep_leapfrog_ntegrator_single for more

    Returns:
        Tuple[jnp.ndarray(shape=(T, P, steps, 3)), jnp.ndarray(shape=(T, Q, steps, 3)), float]:
        planet_xs (jnp.ndarray(shape=(T, P, steps, 3))):
            The 3D positions of P planets at [steps] evenly spaced steps between the
            T times. Units are in AU
        asteroid_xs (jnp.ndarray(shape=(T, Q, steps, 3))):
            Same as planet_xs but for Q asteroids
        dts (jnp.ndarray(shape=(T,))):
            The length of the times steps taken to reach each of the T times. Units are
            in days

    Examples:

        >>> import jax.numpy as jnp
        >>> from astropy.time import Time
        >>> from jorbit.data import STANDARD_PLANET_PARAMS, STANDARD_ASTEROID_PARAMS
        >>> from jorbit.engine.yoshida_integrator import prep_leapfrog_integrator_multiple
        >>> times = Time(["2021-01-01T00:00:00", "2021-01-02T00:00:00", "2021-01-03T00:00:00"])
        >>> planet_xs, asteroid_xs, dts = prep_leapfrog_integrator_multiple(
        ...     t0=times[0].tdb.jd,
        ...     times=jnp.array([t.tdb.jd for t in times][1:]),
        ...     steps=100,
        ...     planet_params=STANDARD_PLANET_PARAMS,
        ...     asteroid_params=STANDARD_ASTEROID_PARAMS,
        ...     offset=0,
        ... )
    """

    def scan_func(carry, scan_over):
        p_xs, a_xs, dt = prep_leapfrog_integrator_single(
            t0=carry,
            tf=scan_over,
            steps=steps,
            planet_params=planet_params,
            asteroid_params=asteroid_params,
            offset=offset,
        )
        return scan_over, (p_xs, a_xs, dt)

    return jax.lax.scan(scan_func, t0, times)[1]


########################################################################################
# Gauss-Jackson integrator helpers
########################################################################################


def prep_gj_integrator_single(
    t0,
    tf,
    jumps,
    a_jk,
    planet_params=STANDARD_PLANET_PARAMS,
    asteroid_params=STANDARD_ASTEROID_PARAMS,
):
    """
    Create some of the inputs needed to run the Gauss-Jackson integrator for one epoch.

    Parameters:
        t0 (float):
            The initial time in TDB JD
        tf (float):
            The final time in TDB JD
        jumps (int):
            The number of jumps to take between t0 and tf. This implicitly sets the size
            of the timesteps taken: dt=(tf-t0)/jumps, with dt in days. Must be > K/2,
            where K is the order of the integrator.
        a_jk (jnp.ndarray(shape=(K+2, K+1))):
            The "a_jk" coefficients for the Gauss-Jackson integrator, as defined in
            Berry and Healy 2004 [1]_. K is the order of the integrator. Values are
            precomputed/stored in jorbit.data.constants for orders 6, 8, 10, 12, 14,
            and 16.
        planet_params (Tuple[jnp.ndarray(shape=(P,)), jnp.ndarray(shape=(P,)), jnp.ndarray(shape=(P,Q,3,R))], default=STANDARD_PLANET_PARAMS from jorbit.data):
            The ephemeris describing P massive objects in the solar system. The first
            element is the initial time of the ephemeris in seconds since J2000 TDB. The
            second element is the length of the interval covered by each piecewise chunk of
            the ephemeris in seconds (for DE44x planets, this is 16 days, and for
            asteroids, it's 32 days). The third element contains the Q coefficients of the
            R piecewise chunks of Chebyshev polynomials that make up the ephemeris, in 3
            x,y,z dimensions.
        asteroid_params (Tuple[jnp.ndarray(shape=(Q,)), jnp.ndarray(shape=(Q,)), jnp.ndarray(shape=(Q,Q,3,R))], default=STANDARD_ASTEROID_PARAMS from jorbit.data):
            Same as the planet_params, but for Q asteroids.

    Returns:
        Tuple[(jnp.ndarray(shape=(M, S + K/2, 3)), jnp.ndarray(shape=(M, S + K/2, 3)), jnp.ndarray(shape=(M, S + K/2, 3)), jnp.ndarray(shape=(P, S + K/2, 3)), jnp.ndarray(shape=(2, K/2, M, Q, 3)), jnp.ndarray(shape=(2, K/2, P, Q, 3)), jnp.ndarray(shape=(2, K/2)))]
        planet_xs (jnp.ndarray(shape=(M, S + K/2, 3))):
            The 3D positions of M planets. Each planet has S + K/2 positions, where S
            is the number of subpositions between t0 and tf. The first K/2 positions are
            *before* the initial time and are needed to warm up the integrator. Between
            t0 and tf, the integrator will take S-1 jumps: dt=(tf-t0)/(S - (K/2) - 1)
        planet_vs (jnp.ndarray(shape=(M, S + K/2, 3))):
            The 3D velocities of M planets. Same as planet_xs, but for velocities.
        planet_as (jnp.ndarray(shape=(M, S + K/2, 3))):
            The 3D accelerations of M planets. Same as planet_xs, but for accelerations.
        asteroid_xs (jnp.ndarray(shape=(P, S + K/2, 3))):
            The 3D positions of P asteroids. Same as planet_xs, but for asteroids.
        planet_xs_warmup (jnp.ndarray(shape=(2, K/2, M, Q, 3))):
            The 3D positions of M planets. Axis 0 is for the forward/backward warmup
            steps. Axis 1, K is the order of the GJ integrator, since that's how many
            steps you need to take forwards and backwards during warmup. Axis 2 is for
            the M planets. Axis 3 is for the Q substeps taken for each of the K
            integration steps. Axis 4 is for the {x,y,z} dimensions, in AU
        asteroid_xs_warmup (jnp.ndarray(shape=(2, K/2, P, Q, 3))):
            The 3D positions of P asteroids. Same as planet_xs_warmup, but for
            asteroids.
        dts_warmup (jnp.ndarray(shape=(2, K/2))):
            The time steps used during warmup. Axis 0 is for forward/backward warmup
            steps. Axis 1, K is the order of the GJ integrator, since that's how many
            steps you need to take forwards and backwards during warmup. See
            jorbit.engine.yoshida_integrator.yoshida_integrate_multiple for more
    """
    dt = (tf - t0) / (jumps)
    MID_IND = int((a_jk.shape[1] - 1) / 2)
    backwards_times = t0 - jnp.arange(1, MID_IND + 1) * dt
    forwards_times = t0 + jnp.arange(1, MID_IND + 1) * dt

    # arbitrarily saying the lower order leapfrog should take 5x as many steps as GJ
    b_planet_xs, b_asteroid_xs, b_dts = prep_leapfrog_integrator_multiple(
        t0=t0,
        times=backwards_times,
        steps=5,
        planet_params=planet_params,
        asteroid_params=asteroid_params,
    )

    f_planet_xs, f_asteroid_xs, f_dts = prep_leapfrog_integrator_multiple(
        t0=t0,
        times=forwards_times,
        steps=5,
        planet_params=planet_params,
        asteroid_params=asteroid_params,
    )

    planet_xs_warmup = jnp.stack((b_planet_xs, f_planet_xs))
    asteroid_xs_warmup = jnp.stack((b_asteroid_xs, f_asteroid_xs))
    dts_warmup = jnp.stack((b_dts, f_dts))

    times = jnp.concatenate((backwards_times[::-1], jnp.linspace(t0, tf, jumps + 1)))
    planet_xs, planet_vs, planet_as = planet_state(
        planet_params=planet_params, times=times, velocity=True, acceleration=True
    )

    asteroid_xs, _, _ = planet_state(
        planet_params=asteroid_params, times=times, velocity=False, acceleration=False
    )

    return (
        planet_xs,
        planet_vs,
        planet_as,
        asteroid_xs,
        planet_xs_warmup,
        asteroid_xs_warmup,
        dts_warmup,
    )


def prep_gj_integrator_multiple(t0, times, jumps, a_jk, planet_params, asteroid_params):
    """
    Create some of the inputs needed to run the Gauss-Jackson integrator for multiple epochs.

    Just a wrapper for prep_gj_integrator_single.

    Parameters:
        t0 (float):
            The initial time in TDB JD
        times (jnp.ndarray(shape=(S,))):
            The epochs to integrate to in TDB JD
        jumps (int):
            The number of jumps to take between each epoch. This implicitly sets the
            size of the timesteps taken: dt=(t_{i+1}-t_{i})/jumps, with dt in days, for
            each ith epoch. The number of jumps must be the same for each epoch, so
            unless the epochs themselves are evenly spaced, the size of the timesteps
            will differ between them. Must be > K/2, where K is the order of the
            integrator
        a_jk (jnp.ndarray(shape=(K+2, K+1))):
            The "a_jk" coefficients for the Gauss-Jackson integrator, as defined in
            Berry and Healy 2004 [1]_. K is the order of the integrator. Values are
            precomputed/stored in jorbit.data.constants for orders 8, 10, 12, and 14.
        planet_params (Tuple[jnp.ndarray(shape=(P,)), jnp.ndarray(shape=(P,)), jnp.ndarray(shape=(P,Q,3,R))], default=STANDARD_PLANET_PARAMS from jorbit.data):
            The ephemeris describing P massive objects in the solar system. The first
            element is the initial time of the ephemeris in seconds since J2000 TDB. The
            second element is the length of the interval covered by each piecewise chunk of
            the ephemeris in seconds (for DE44x planets, this is 16 days, and for
            asteroids, it's 32 days). The third element contains the Q coefficients of the
            R piecewise chunks of Chebyshev polynomials that make up the ephemeris, in 3
            x,y,z dimensions.
        asteroid_params (Tuple[jnp.ndarray(shape=(Q,)), jnp.ndarray(shape=(Q,)), jnp.ndarray(shape=(Q,Q,3,R))], default=STANDARD_ASTEROID_PARAMS from jorbit.data):
            Same as the planet_params, but for Q asteroids.

    Returns:
        planet_xs (jnp.ndarray(shape=(S, M, J + K/2, 3))):
            The 3D positions of M planets. Each planet has J + K/2 positions, per
            integration epoch, and there are S epochs. The first K/2 positions of each
            epoch are *before* the epoch and are needed to warm up the integrator- see
            jorbit.engine.gauss_jackson_integrator.gj_integrate for more
        planet_vs (jnp.ndarray(shape=(S, M, J + K/2, 3))):
            The 3D velocities of M planets. Same as planet_xs, but for velocities. Units
            are AU/day
        planet_as (jnp.ndarray(shape=(S, M, J + K/2, 3))):
            The 3D accelerations of M planets. Same as planet_xs, but for accelerations.
            Units are AU/day^2
        asteroid_xs (jnp.ndarray(shape=(S, P, J + K/2, 3))):
            The 3D positions of P asteroids. Same as planet_xs, but for asteroids
        planet_xs_warmup (jnp.ndarray(shape=(S, 2, K/2, M, Q, 3))):
            The 3D positions of M planets during the integrator warmup time. Axis 0
            marks the epochs. Axis 1 is for the backwards and forwards positions. Axis
            2, K is the order of the GJ integrator, since that's how many warmup steps
            you need. Axis 3 is for the M planets. Axis 4 is for the Q substeps taken
            to reach each of the warmup positions. Axis 5 is for the {x,y,z} dimensions.
            Units are AU
        asteroid_xs_warmup (jnp.ndarray(shape=(S, 2, K/2, P, Q, 3))):
            The 3D positions of P asteroids during the integrator warmup time. Same as
            planet_xs_warmup, but for asteroids
        dts_warmup (jnp.ndarray(shape=(S, 2, K/2))):
            The length of the time steps used to reach each of the warmup positions.
            Axis 0 marks the epochs. Axis 1 is for the backwards and forwards positions.
            Axis 2, K is the order of the GJ integrator, since that's how many warmup
            steps are taken. Units are days
    """

    def scan_func(carry, scan_over):
        (
            planet_xs,
            planet_vs,
            planet_as,
            asteroid_xs,
            planet_xs_warmup,
            asteroid_xs_warmup,
            dts_warmup,
        ) = prep_gj_integrator_single(
            t0=carry,
            tf=scan_over,
            jumps=jumps,
            a_jk=a_jk,
            planet_params=planet_params,
            asteroid_params=asteroid_params,
        )
        return scan_over, (
            planet_xs,
            planet_vs,
            planet_as,
            asteroid_xs,
            planet_xs_warmup,
            asteroid_xs_warmup,
            dts_warmup,
        )

    return jax.lax.scan(scan_func, t0, times)[1]


def prep_uneven_GJ_integrator(
    times,
    low_order,
    high_order,
    low_order_cutoff,
    targeted_low_order_timestep,
    targeted_high_order_timestep,
    coeffs_dict,
    ragged=False,
):
    orders = []
    Jumps = []
    for i in range(len(times) - 1):
        # print(f'Current time: {times[i]}')
        # print(f'Next time: {times[i+1]}')
        integrator_jump = times[i + 1] - times[i]
        # print(f'Jump: {integrator_jump}')

        if integrator_jump < low_order_cutoff:
            # print('desired timestep is below the minimum, using low order')
            proposed_jumps = integrator_jump / targeted_low_order_timestep
            actual_jumps = jnp.where(
                proposed_jumps < (low_order / 2 + 1),
                low_order / 2 + 1,
                jnp.ceil(proposed_jumps),
            )
            # print(f'actual jumps: {actual_jumps} (should be {low_order/2})')
            t = integrator_jump / actual_jumps
            # print(f'low order timestep: {t}')
            orders.append(low_order)
            Jumps.append(int(jnp.ceil(actual_jumps)))
        else:
            # print('can use more time steps than minimum, using high order')
            proposed_jumps = integrator_jump / targeted_high_order_timestep
            # print(f'proposed jumps: {proposed_jumps}')
            actual_jumps = jnp.where(
                proposed_jumps < high_order / 2 + 1,
                high_order / 2 + 1,
                jnp.ceil(proposed_jumps),
            )
            # print(f'actual jumps: {actual_jumps}')
            t = integrator_jump / actual_jumps
            # print(f'high order timestep: {t}')
            # print(f'steps: {jnp.ceil(actual_jumps/t)} (cannot be smaller than {high_order/2})')
            orders.append(high_order)
            assert jnp.ceil(integrator_jump / t) >= high_order / 2 + 1
            Jumps.append(int(jnp.ceil(integrator_jump / t)))

        # print()
    chunks = []
    ts = []
    for i in range(len(times) - 1):
        if i == 0:
            current_size = Jumps[i]
            current_order = orders[i]
            ts = [times[i + 1]]
            continue

        if (Jumps[i] == current_size) & (orders[i] == current_order):
            ts.append(times[i + 1])
            if i == len(times) - 2:
                chunks.append(
                    {
                        "integrator order": str(int(current_order)),
                        "jumps per integration": current_size,
                        "times": jnp.array(ts),
                    }
                )
            continue
        chunks.append(
            {
                "integrator order": str(int(current_order)),
                "jumps per integration": current_size,
                "times": jnp.array(ts),
            }
        )
        current_size = Jumps[i]
        current_order = orders[i]
        ts = [times[i + 1]]
    if chunks[-1]["times"][-1] != times[-1]:
        chunks.append(
            {
                "integrator order": str(int(current_order)),
                "jumps per integration": current_size,
                "times": jnp.array(ts),
            }
        )

    Valid_Steps = []
    Planet_Xs = []
    Planet_Vs = []
    Planet_As = []
    Asteroid_Xs = []
    Planet_Xs_Warmup = []
    Asteroid_Xs_Warmup = []
    Dts_Warmup = []

    t = times[0]
    for c in tqdm(chunks, position=1, leave=False):
        z = c["jumps per integration"] + int(c["integrator order"]) / 2 + 1
        Valid_Steps.append(jnp.array([z] * len(c["times"])))
        c["valid steps"] = jnp.array([z] * len(c["times"]))

        x = prep_gj_integrator_multiple(
            t0=t,
            times=c["times"],
            jumps=c["jumps per integration"],
            a_jk=coeffs_dict["a_jk"][c["integrator order"]],
            planet_params=STANDARD_PLANET_PARAMS,
            asteroid_params=STANDARD_ASTEROID_PARAMS,
        )
        Planet_Xs.append(x[0])
        Planet_Vs.append(x[1])
        Planet_As.append(x[2])
        Asteroid_Xs.append(x[3])
        Planet_Xs_Warmup.append(x[4])
        Asteroid_Xs_Warmup.append(x[5])
        Dts_Warmup.append(x[6])

        t = c["times"][-1]

    # return Valid_Steps
    if not ragged:
        most_jumps = 0
        for p in Planet_Xs:
            if p.shape[2] > most_jumps:
                most_jumps = p.shape[2]

        padded_planet_xs = []
        padded_planet_vs = []
        padded_planet_as = []
        padded_asteroid_xs = []

        for i in range(len(Planet_Xs)):
            padded = (
                jnp.ones((Planet_Xs[i].shape[0], Planet_Xs[i].shape[1], most_jumps, 3))
                * 999.0
            )
            padded_planet_xs.append(
                padded.at[:, :, : Planet_Xs[i].shape[2], :].set(Planet_Xs[i])
            )
            padded_planet_vs.append(
                padded.at[:, :, : Planet_Vs[i].shape[2], :].set(Planet_Vs[i])
            )
            padded_planet_as.append(
                padded.at[:, :, : Planet_As[i].shape[2], :].set(Planet_As[i])
            )

            padded = (
                jnp.ones(
                    (Planet_Xs[i].shape[0], Asteroid_Xs[i].shape[1], most_jumps, 3)
                )
                * 999.0
            )
            padded_asteroid_xs.append(
                padded.at[:, :, : Asteroid_Xs[i].shape[2], :].set(Asteroid_Xs[i])
            )

        Valid_Steps = jnp.concatenate(Valid_Steps)
        Planet_Xs = jnp.row_stack(padded_planet_xs)
        Planet_Vs = jnp.row_stack(padded_planet_vs)
        Planet_As = jnp.row_stack(padded_planet_as)
        Asteroid_Xs = jnp.row_stack(padded_asteroid_xs)
        Planet_Xs_Warmup = jnp.row_stack(Planet_Xs_Warmup)
        Asteroid_Xs_Warmup = jnp.row_stack(Asteroid_Xs_Warmup)
        Dts_Warmup = jnp.row_stack(Dts_Warmup)
        return (
            Valid_Steps,
            Planet_Xs,
            Planet_Vs,
            Planet_As,
            Asteroid_Xs,
            Planet_Xs_Warmup,
            Asteroid_Xs_Warmup,
            Dts_Warmup,
        )

    return (
        Valid_Steps,
        Planet_Xs,
        Planet_Vs,
        Planet_As,
        Asteroid_Xs,
        Planet_Xs_Warmup,
        Asteroid_Xs_Warmup,
        Dts_Warmup,
        chunks,
    )


# particles = [p1, p2, p3, p4]
# system_time = p1.time
# fit_planet_gms = False
# fit_asteroid_gms = False
# _planet_params = STANDARD_PLANET_PARAMS
# _asteroid_params = STANDARD_ASTEROID_PARAMS
# _planet_gms = STANDARD_PLANET_GMS
# _asteroid_gms = STANDARD_ASTEROID_GMS

# coeffs_dict = {
#     "a_jk": {
#         "6": GJ6_A,
#         "8": GJ8_A,
#         "10": GJ10_A,
#         "12": GJ12_A,
#         "14": GJ14_A,
#         "16": GJ16_A,
#     },
#     "b_jk": {
#         "6": GJ6_B,
#         "8": GJ8_B,
#         "10": GJ10_B,
#         "12": GJ12_B,
#         "14": GJ14_B,
#         "16": GJ16_B,
#     },
# }


# def create_free_fixed_params():
#     # f for Free, r for Rigid
#     tracer_fx_rm = {"x": [], "v": [], "gm": [], "obs": [], "particles": []}
#     tracer_rx_rm = {"x": [], "v": [], "gm": [], "obs": [], "particles": []}
#     massive_fx_rm = {"x": [], "v": [], "gm": [], "obs": [], "particles": []}
#     massive_rx_fm = {"x": [], "v": [], "gm": [], "obs": [], "particles": []}
#     massive_fx_fm = {"x": [], "v": [], "gm": [], "obs": [], "particles": []}
#     massive_rx_rm = {"x": [], "v": [], "gm": [], "obs": [], "particles": []}

#     for p in particles:
#         if p.free_orbit and p.free_gm:
#             massive_fx_fm["x"].append(p.x)
#             massive_fx_fm["v"].append(p.v)
#             massive_fx_fm["gm"].append(p.gm)
#             massive_fx_fm["obs"].append(p.observations)
#             massive_fx_fm["particles"].append(p)
#         elif p.free_orbit and not p.free_gm:
#             if p.gm == 0:
#                 tracer_fx_rm["x"].append(p.x)
#                 tracer_fx_rm["v"].append(p.v)
#                 tracer_fx_rm["gm"].append(p.gm)
#                 tracer_fx_rm["obs"].append(p.observations)
#                 tracer_fx_rm["particles"].append(p)
#             else:
#                 massive_fx_rm["x"].append(p.x)
#                 massive_fx_rm["v"].append(p.v)
#                 massive_fx_rm["gm"].append(p.gm)
#                 massive_fx_rm["obs"].append(p.observations)
#                 massive_fx_rm["particles"].append(p)
#         elif not p.free_orbit and p.free_gm:
#             massive_rx_fm["x"].append(p.x)
#             massive_rx_fm["v"].append(p.v)
#             massive_rx_fm["gm"].append(p.gm)
#             massive_rx_fm["obs"].append(p.observations)
#             massive_rx_fm["particles"].append(p)
#         elif not p.free_orbit and not p.free_gm:
#             if p.gm == 0:
#                 tracer_rx_rm["x"].append(p.x)
#                 tracer_rx_rm["v"].append(p.v)
#                 tracer_rx_rm["gm"].append(p.gm)
#                 tracer_rx_rm["obs"].append(p.observations)
#                 tracer_rx_rm["particles"].append(p)
#             else:
#                 massive_rx_rm["x"].append(p.x)
#                 massive_rx_rm["v"].append(p.v)
#                 massive_rx_rm["gm"].append(p.gm)
#                 massive_rx_rm["obs"].append(p.observations)
#                 massive_rx_rm["particles"].append(p)

#     fixed_params = {}
#     # if len(tracer_fx_rm["x"]) > 0:
#     #     fixed_params["tracer_fx_rm__gm"] = jnp.array(tracer_fx_rm["gm"])
#     # else:
#     #     fixed_params["tracer_fx_rm__gm"] = jnp.empty((0,))
#     if len(tracer_rx_rm["x"]) > 0:
#         fixed_params["tracer_rx_rm__x"] = jnp.array(tracer_rx_rm["x"])
#         fixed_params["tracer_rx_rm__v"] = jnp.array(tracer_rx_rm["v"])
#         # fixed_params["tracer_rx_rm__gm"] = jnp.array(tracer_rx_rm["gm"])
#     else:
#         fixed_params["tracer_rx_rm__x"] = jnp.empty((0, 3))
#         fixed_params["tracer_rx_rm__v"] = jnp.empty((0, 3))
#         # fixed_params["tracer_rx_rm__gm"] = jnp.empty((0,))
#     if len(massive_fx_rm["x"]) > 0:
#         fixed_params["massive_fx_rm__gm"] = jnp.array(massive_fx_rm["gm"])
#     else:
#         fixed_params["massive_fx_rm__gm"] = jnp.empty((0,))
#     if len(massive_rx_fm["x"]) > 0:
#         fixed_params["massive_rx_fm__x"] = jnp.array(massive_rx_fm["x"])
#         fixed_params["massive_rx_fm__v"] = jnp.array(massive_rx_fm["v"])
#     else:
#         fixed_params["massive_rx_fm__x"] = jnp.empty((0, 3))
#         fixed_params["massive_rx_fm__v"] = jnp.empty((0, 3))
#     if len(massive_rx_rm["x"]) > 0:
#         fixed_params["massive_rx_rm__x"] = jnp.array(massive_rx_rm["x"])
#         fixed_params["massive_rx_rm__v"] = jnp.array(massive_rx_rm["v"])
#         fixed_params["massive_rx_rm__gm"] = jnp.array(massive_rx_rm["gm"])
#     else:
#         fixed_params["massive_rx_rm__x"] = jnp.empty((0, 3))
#         fixed_params["massive_rx_rm__v"] = jnp.empty((0, 3))
#         fixed_params["massive_rx_rm__gm"] = jnp.empty((0,))

#     free_params = {}
#     if len(tracer_fx_rm["x"]) > 0:
#         free_params["tracer_fx_rm__x"] = jnp.array(tracer_fx_rm["x"])
#         free_params["tracer_fx_rm__v"] = jnp.array(tracer_fx_rm["v"])
#     else:
#         fixed_params["tracer_fx_rm__x"] = jnp.empty((0, 3))
#         fixed_params["tracer_fx_rm__v"] = jnp.empty((0, 3))

#     if len(massive_fx_rm["x"]) > 0:
#         free_params["massive_fx_rm__x"] = jnp.array(massive_fx_rm["x"])
#         free_params["massive_fx_rm__v"] = jnp.array(massive_fx_rm["v"])
#     else:
#         fixed_params["massive_fx_rm__x"] = jnp.empty((0, 3))
#         fixed_params["massive_fx_rm__v"] = jnp.empty((0, 3))

#     if len(massive_rx_fm["x"]) > 0:
#         free_params["massive_rx_fm__gm"] = jnp.array(massive_rx_fm["gm"])
#     else:
#         fixed_params["massive_rx_fm__gm"] = jnp.empty((0,))

#     if len(massive_fx_fm["x"]) > 0:
#         free_params["massive_fx_fm__x"] = jnp.array(massive_fx_fm["x"])
#         free_params["massive_fx_fm__v"] = jnp.array(massive_fx_fm["v"])
#         free_params["massive_fx_fm__gm"] = jnp.array(massive_fx_fm["gm"])
#     else:
#         fixed_params["massive_fx_fm__x"] = jnp.empty((0, 3))
#         fixed_params["massive_fx_fm__v"] = jnp.empty((0, 3))
#         fixed_params["massive_fx_fm__gm"] = jnp.empty((0,))

#     if fit_planet_gms:
#         free_params["planet_gms"] = _planet_gms
#     else:
#         fixed_params["planet_gms"] = _planet_gms

#     if fit_asteroid_gms:
#         free_params["asteroid_gms"] = _asteroid_gms
#     else:
#         fixed_params["asteroid_gms"] = _asteroid_gms


#     ordered_tracer_obs = tracer_fx_rm["obs"] + tracer_rx_rm["obs"]
#     ordered_massive_obs = (
#         massive_fx_rm["obs"]
#         + massive_rx_fm["obs"]
#         + massive_fx_fm["obs"]
#         + massive_rx_rm["obs"]
#     )

#     reordered_particles = (
#         tracer_fx_rm["particles"]
#         + tracer_rx_rm["particles"]
#         + massive_fx_rm["particles"]
#         + massive_rx_fm["particles"]
#         + massive_fx_fm["particles"]
#         + massive_rx_rm["particles"]
#     )

#     return free_params, fixed_params, ordered_tracer_obs, ordered_massive_obs, reordered_particles


# def prep_system_GJ_integrator(
#     integrator_order,
#     targeted_timestep,
#     ordered_tracer_obs,
#     ordered_massive_obs,
# ):
#     def _prep_system_GJ_integrator(ordered_obs):
#         # The the data needed to integrate each individual particle to the times it
#         # was observed. Each set of arrays for an individual particle is padded with 999s
#         # so that there is the same number of jumps between epochs, but then the number of
#         # epochs and the number of jumps per epoch will vary particle-to-particle
#         individual_integrator_prep = []
#         # times=jnp.concatenate((jnp.array([system_time]), o.times)),
#         for o in tqdm(
#             ordered_obs, desc="Pre-computing perturber positions", position=0
#         ):
#             z = prep_uneven_GJ_integrator(
#                 times=o.times,
#                 low_order=integrator_order,
#                 high_order=integrator_order,
#                 low_order_cutoff=1e-6,
#                 targeted_low_order_timestep=targeted_timestep,
#                 targeted_high_order_timestep=targeted_timestep,
#                 coeffs_dict=coeffs_dict,
#                 ragged=False,
#             )
#             individual_integrator_prep.append(z)

#         # Now pad those precomputed (and already padded) arrays so that every particle
#         # has the same number of epochs and the same number of jumps per epoch
#         most_jumps = 0
#         most_steps_per_epoch = 0
#         for p in individual_integrator_prep:
#             p = p[1]  # Planet_Xs. p[0] is Valid_Steps
#             if p.shape[0] > most_jumps:
#                 most_jumps = p.shape[0]
#             if p.shape[2] > most_steps_per_epoch:
#                 most_steps_per_epoch = p.shape[2]
#         Valid_Steps = []
#         Planet_Xs = []
#         Planet_Vs = []
#         Planet_As = []
#         Asteroid_Xs = []
#         Planet_Xs_Warmup = []
#         Asteroid_Xs_Warmup = []
#         Dts_Warmup = []
#         for p in individual_integrator_prep:
#             Valid_Steps.append(
#                 jnp.pad(
#                     p[0],
#                     (0, most_jumps - p[0].shape[0]),
#                     mode="constant",
#                     constant_values=999,
#                 )
#             )
#             Planet_Xs.append(
#                 jnp.pad(
#                     p[1],
#                     (
#                         (0, most_jumps - p[1].shape[0]),
#                         (0, 0),
#                         (0, most_steps_per_epoch - p[1].shape[2]),
#                         (0, 0),
#                     ),
#                     mode="constant",
#                     constant_values=999,
#                 )
#             )
#             Planet_Vs.append(
#                 jnp.pad(
#                     p[2],
#                     (
#                         (0, most_jumps - p[2].shape[0]),
#                         (0, 0),
#                         (0, most_steps_per_epoch - p[2].shape[2]),
#                         (0, 0),
#                     ),
#                     mode="constant",
#                     constant_values=999,
#                 )
#             )
#             Planet_As.append(
#                 jnp.pad(
#                     p[3],
#                     (
#                         (0, most_jumps - p[3].shape[0]),
#                         (0, 0),
#                         (0, most_steps_per_epoch - p[3].shape[2]),
#                         (0, 0),
#                     ),
#                     mode="constant",
#                     constant_values=999,
#                 )
#             )
#             Asteroid_Xs.append(
#                 jnp.pad(
#                     p[4],
#                     (
#                         (0, most_jumps - p[4].shape[0]),
#                         (0, 0),
#                         (0, most_steps_per_epoch - p[4].shape[2]),
#                         (0, 0),
#                     ),
#                     mode="constant",
#                     constant_values=999,
#                 )
#             )
#             Planet_Xs_Warmup.append(
#                 jnp.pad(
#                     p[5],
#                     (
#                         (0, most_jumps - p[5].shape[0]),
#                         (0, 0),
#                         (0, 0),
#                         (0, 0),
#                         (0, 0),
#                         (0, 0),
#                     ),
#                     mode="constant",
#                     constant_values=999,
#                 )
#             )
#             Asteroid_Xs_Warmup.append(
#                 jnp.pad(
#                     p[6],
#                     (
#                         (0, most_jumps - p[6].shape[0]),
#                         (0, 0),
#                         (0, 0),
#                         (0, 0),
#                         (0, 0),
#                         (0, 0),
#                     ),
#                     mode="constant",
#                     constant_values=999,
#                 )
#             )
#             Dts_Warmup.append(
#                 jnp.pad(
#                     p[7],
#                     ((0, most_jumps - p[7].shape[0]), (0, 0), (0, 0)),
#                 )
#             )
#         if len(Valid_Steps) > 0:
#             Valid_Steps = jnp.stack(Valid_Steps)
#             Planet_Xs = jnp.stack(Planet_Xs)
#             Planet_Vs = jnp.stack(Planet_Vs)
#             Planet_As = jnp.stack(Planet_As)
#             Asteroid_Xs = jnp.stack(Asteroid_Xs)
#             Planet_Xs_Warmup = jnp.stack(Planet_Xs_Warmup)
#             Asteroid_Xs_Warmup = jnp.stack(Asteroid_Xs_Warmup)
#             Dts_Warmup = jnp.stack(Dts_Warmup)
#         else:
#             Valid_Steps = jnp.empty((0,))
#             Planet_Xs = jnp.empty((0, 0, 0, 0, 0))
#             Planet_Vs = jnp.empty((0, 0, 0, 0, 0))
#             Planet_As = jnp.empty((0, 0, 0, 0, 0))
#             Asteroid_Xs = jnp.empty((0, 0, 0, 0, 0))
#             Planet_Xs_Warmup = jnp.empty((0, 0, 0, 0, 0, 0))
#             Asteroid_Xs_Warmup = jnp.empty((0, 0, 0, 0, 0, 0))
#             Dts_Warmup = jnp.empty((0, 0, 0))

#         # unrelated, we need the initial times, jump times, RAs, Decs, astrometric
#         # uncertainties, and location from which each observation was taken for each
#         # particle. Every but the initial times need to be padded to match the number of
#         # epochs, which is now the same for all particles
#         init_times = []
#         jump_times = []
#         ras = []
#         decs = []
#         astrometric_uncertainties = []
#         observer_positions = []
#         observed_planet_xs = []
#         observed_asteroid_xs = []
#         for o in ordered_obs:
#             init_times.append(o.times[0])
#             jump_times.append(
#                 jnp.pad(
#                     o.times[1:],
#                     (0, most_jumps - len(o.times) + 1),
#                     mode="constant",
#                     constant_values=999,
#                 )
#             )
#             ras.append(
#                 jnp.pad(
#                     o.ra,
#                     (0, most_jumps - len(o.times) + 1),
#                     mode="constant",
#                     constant_values=999,
#                 ),
#             )
#             decs.append(
#                 jnp.pad(
#                     o.dec,
#                     (0, most_jumps - len(o.times) + 1),
#                     mode="constant",
#                     constant_values=999,
#                 ),
#             )
#             astrometric_uncertainties.append(
#                 jnp.pad(
#                     o.astrometric_uncertainties,
#                     (0, most_jumps - len(o.times) + 1),
#                     mode="constant",
#                     constant_values=jnp.inf,
#                 ),
#             )
#             observer_positions.append(
#                 jnp.pad(
#                     o.observer_positions,
#                     ((0, most_jumps - len(o.times) + 1), (0, 0)),
#                     mode="constant",
#                     constant_values=999,
#                 ),
#             )
#             planetxs, _, _ = planet_state(
#                 planet_params=_planet_params,
#                 times=o.times,
#                 velocity=False,
#                 acceleration=False,
#             )
#             asteroidxs, _, _ = planet_state(
#                 planet_params=_asteroid_params,
#                 times=o.times,
#                 velocity=False,
#                 acceleration=False,
#             )
#             observed_planet_xs.append(
#                 jnp.pad(
#                     planetxs,
#                     ((0,0), (0, most_jumps - len(o.times) + 1), (0, 0)),
#                     mode="constant",
#                     constant_values=999,
#                 ),
#             )
#             observed_asteroid_xs.append(
#                 jnp.pad(
#                     asteroidxs,
#                     ((0,0), (0, most_jumps - len(o.times) + 1), (0, 0)),
#                     mode="constant",
#                     constant_values=999,
#                 ),
#             )

#         if len(init_times) > 0:
#             init_times = jnp.array(init_times)
#             jump_times = jnp.stack(jump_times)
#             ras = jnp.stack(ras)
#             decs = jnp.stack(decs)
#             astrometric_uncertainties = jnp.stack(astrometric_uncertainties)
#             observer_positions = jnp.stack(observer_positions)
#             observed_planet_xs = jnp.stack(observed_planet_xs)
#             observed_asteroid_xs = jnp.stack(observed_asteroid_xs)
#         else:
#             init_times = jnp.empty((0,))
#             jump_times = jnp.empty((0, 0))
#             ras = jnp.empty((0, 0))
#             decs = jnp.empty((0, 0))
#             astrometric_uncertainties = jnp.empty((0, 0))
#             observer_positions = jnp.empty((0, 0, 0))
#             observed_planet_xs = jnp.empty((0, 0, 0, 0))
#             observed_asteroid_xs = jnp.empty((0, 0, 0, 0))

#         return (
#             Valid_Steps,
#             Planet_Xs,
#             Planet_Vs,
#             Planet_As,
#             Asteroid_Xs,
#             Planet_Xs_Warmup,
#             Asteroid_Xs_Warmup,
#             Dts_Warmup,
#             init_times,
#             jump_times,
#             ras,
#             decs,
#             astrometric_uncertainties,
#             observer_positions,
#             observed_planet_xs,
#             observed_asteroid_xs,
#         )

#     (
#         tracer_Valid_Steps,
#         tracer_Planet_Xs,
#         tracer_Planet_Vs,
#         tracer_Planet_As,
#         tracer_Asteroid_Xs,
#         tracer_Planet_Xs_Warmup,
#         tracer_Asteroid_Xs_Warmup,
#         tracer_Dts_Warmup,
#         tracer_Init_Times,
#         tracer_Jump_Times,
#         tracer_RAs,
#         tracer_Decs,
#         tracer_Astrometric_Uncertainties,
#         tracer_Observer_Positions,
#         tracer_Observed_Planet_Xs,
#         tracer_Observed_Asteroid_Xs,
#     ) = _prep_system_GJ_integrator(ordered_tracer_obs)
#     (
#         massive_Valid_Steps,
#         massive_Planet_Xs,
#         massive_Planet_Vs,
#         massive_Planet_As,
#         massive_Asteroid_Xs,
#         massive_Planet_Xs_Warmup,
#         massive_Asteroid_Xs_Warmup,
#         massive_Dts_Warmup,
#         massive_Init_Times,
#         massive_Jump_Times,
#         massive_RAs,
#         massive_Decs,
#         massive_Astrometric_Uncertainties,
#         massive_Observer_Positions,
#         massive_Observed_Planet_Xs,
#         massive_Observed_Asteroid_Xs,
#     ) = _prep_system_GJ_integrator(ordered_massive_obs)

#     padded_supporting_data = {
#         "tracer_Valid_Steps": tracer_Valid_Steps,
#         "tracer_Planet_Xs": tracer_Planet_Xs,
#         "tracer_Planet_Vs": tracer_Planet_Vs,
#         "tracer_Planet_As": tracer_Planet_As,
#         "tracer_Asteroid_Xs": tracer_Asteroid_Xs,
#         "tracer_Planet_Xs_Warmup": tracer_Planet_Xs_Warmup,
#         "tracer_Asteroid_Xs_Warmup": tracer_Asteroid_Xs_Warmup,
#         "tracer_Dts_Warmup": tracer_Dts_Warmup,
#         "tracer_Init_Times": tracer_Init_Times,
#         "tracer_Jump_Times": tracer_Jump_Times,
#         "tracer_RAs": tracer_RAs,
#         "tracer_Decs": tracer_Decs,
#         "tracer_Astrometric_Uncertainties": tracer_Astrometric_Uncertainties,
#         "tracer_Observer_Positions": tracer_Observer_Positions,
#         "tracer_Observed_Planet_Xs": tracer_Observed_Planet_Xs,
#         "tracer_Observed_Asteroid_Xs": tracer_Observed_Asteroid_Xs,
#         "massive_Valid_Steps": massive_Valid_Steps,
#         "massive_Planet_Xs": massive_Planet_Xs,
#         "massive_Planet_Vs": massive_Planet_Vs,
#         "massive_Planet_As": massive_Planet_As,
#         "massive_Asteroid_Xs": massive_Asteroid_Xs,
#         "massive_Planet_Xs_Warmup": massive_Planet_Xs_Warmup,
#         "massive_Asteroid_Xs_Warmup": massive_Asteroid_Xs_Warmup,
#         "massive_Dts_Warmup": massive_Dts_Warmup,
#         "massive_Init_Times": massive_Init_Times,
#         "massive_Jump_Times": massive_Jump_Times,
#         "massive_RAs": massive_RAs,
#         "massive_Decs": massive_Decs,
#         "massive_Astrometric_Uncertainties": massive_Astrometric_Uncertainties,
#         "massive_Observer_Positions": massive_Observer_Positions,
#         "massive_Observed_Planet_Xs": massive_Observed_Planet_Xs,
#         "massive_Observed_Asteroid_Xs": massive_Observed_Asteroid_Xs,
#     }

#     return padded_supporting_data

# free_params, fixed_params, ordered_tracer_obs, ordered_massive_obs, reordered_particles = (
#     create_free_fixed_params()
# )
# padded_supporting_data = prep_system_GJ_integrator(
#     integrator_order=8,
#     targeted_timestep=3.0,
#     ordered_tracer_obs=ordered_tracer_obs,
#     ordered_massive_obs=ordered_massive_obs,
# )

# def _resids(
#     tracer_fx_rm__x,
#     tracer_fx_rm__v,
#     tracer_rx_rm__x,
#     tracer_rx_rm__v,
#     massive_fx_rm__x,
#     massive_fx_rm__v,
#     massive_fx_rm__gm,
#     massive_rx_fm__x,
#     massive_rx_fm__v,
#     massive_rx_fm__gm,
#     massive_fx_fm__x,
#     massive_fx_fm__v,
#     massive_fx_fm__gm,
#     massive_rx_rm__x,
#     massive_rx_rm__v,
#     massive_rx_rm__gm,
#     tracer_Valid_Steps,
#     tracer_Planet_Xs,
#     tracer_Planet_Vs,
#     tracer_Planet_As,
#     tracer_Asteroid_Xs,
#     tracer_Planet_Xs_Warmup,
#     tracer_Asteroid_Xs_Warmup,
#     tracer_Dts_Warmup,
#     tracer_Init_Times,
#     tracer_Jump_Times,
#     tracer_RAs,
#     tracer_Decs,
#     tracer_Astrometric_Uncertainties,
#     tracer_Observer_Positions,
#     tracer_Observed_Planet_Xs,
#     tracer_Observed_Asteroid_Xs,
#     massive_Valid_Steps,
#     massive_Planet_Xs,
#     massive_Planet_Vs,
#     massive_Planet_As,
#     massive_Asteroid_Xs,
#     massive_Planet_Xs_Warmup,
#     massive_Asteroid_Xs_Warmup,
#     massive_Dts_Warmup,
#     massive_Init_Times,
#     massive_Jump_Times,
#     massive_RAs,
#     massive_Decs,
#     massive_Astrometric_Uncertainties,
#     massive_Observer_Positions,
#     massive_Observed_Planet_Xs,
#     massive_Observed_Asteroid_Xs,
#     planet_gms,
#     asteroid_gms,
# ):
#     tracer_xs = jnp.concatenate((tracer_fx_rm__x, tracer_rx_rm__x))
#     tracer_vs = jnp.concatenate((tracer_fx_rm__v, tracer_rx_rm__v))
#     massive_xs = jnp.concatenate(
#         (massive_fx_rm__x, massive_rx_fm__x, massive_fx_fm__x, massive_rx_rm__x)
#     )
#     massive_vs = jnp.concatenate(
#         (massive_fx_rm__v, massive_rx_fm__v, massive_fx_fm__v, massive_rx_rm__v)
#     )
#     massive_gms = jnp.concatenate(
#         (massive_fx_rm__gm, massive_rx_fm__gm, massive_fx_fm__gm, massive_rx_rm__gm)
#     )

#     def _tracer_scan_func(carry, scan_over):
#         (
#             x0,
#             v0,
#             init_time,
#             jump_times,
#             valid_steps,
#             planet_xs,
#             planet_vs,
#             planet_as,
#             asteroid_xs,
#             planet_xs_warmup,
#             asteroid_xs_warmup,
#             dts_warmup,
#             observer_positions,
#             ras,
#             decs,
#             observed_planet_xs,
#             observed_asteroid_xs,
#         ) = scan_over

#         x0 = jnp.concatenate((jnp.array([x0]), massive_xs))
#         v0 = jnp.concatenate((jnp.array([v0]), massive_vs))
#         gms = jnp.concatenate((jnp.array([0.0]), massive_gms))

#         x, v = gj_integrate_multiple(
#             x0=x0,
#             v0=v0,
#             gms=gms,
#             valid_steps=valid_steps,
#             b_jk=GJ8_B,
#             a_jk=GJ8_A,
#             t0=init_time,
#             times=jump_times,
#             planet_xs=planet_xs,
#             planet_vs=planet_vs,
#             planet_as=planet_as,
#             asteroid_xs=asteroid_xs,
#             planet_xs_warmup=planet_xs_warmup,
#             asteroid_xs_warmup=asteroid_xs_warmup,
#             dts_warmup=dts_warmup,
#             warmup_C=Y4_C,
#             warmup_D=Y4_D,
#             planet_gms=STANDARD_PLANET_GMS,
#             asteroid_gms=STANDARD_ASTEROID_GMS,
#             use_GR=True,
#         )


#         # we only care about the first particle, the tracer
#         # add the first position back in
#         x = jnp.concatenate((x0[0][None], x[0, :, :]))
#         v = jnp.concatenate((v0[0][None], v[0, :, :]))

#         # calc_ra, calc_dec = on_sky(
#         #     xs=x,
#         #     vs=v,
#         #     gms=jnp.zeros(x.shape[0]),
#         #     times=jnp.concatenate(
#         #         (init_time[None], jump_times)
#         #     ),
#         #     observer_positions=observer_positions,
#         #     planet_params=planet_params,
#         #     asteroid_params=asteroid_params,
#         #     planet_gms=planet_gms,
#         #     asteroid_gms=asteroid_gms,
#         # )
#         calc_ra, calc_dec = on_sky(
#             xs=x,
#             vs=v,
#             gms=jnp.zeros(x.shape[0]),
#             observer_positions=observer_positions,
#             planet_xs=observed_planet_xs,
#             asteroid_xs=observed_asteroid_xs,
#             planet_gms=planet_gms,
#             asteroid_gms=asteroid_gms,
#         )


#         resids = sky_error(
#             calc_ra=calc_ra,
#             calc_dec=calc_dec,
#             true_ra=ras,
#             true_dec=decs,
#         )

#         return None, (x, v, calc_ra, calc_dec, resids)

#     tmp = jax.lax.scan(
#         _tracer_scan_func,
#         None,
#         (
#             tracer_xs,
#             tracer_vs,
#             tracer_Init_Times,
#             tracer_Jump_Times,
#             tracer_Valid_Steps,
#             tracer_Planet_Xs,
#             tracer_Planet_Vs,
#             tracer_Planet_As,
#             tracer_Asteroid_Xs,
#             tracer_Planet_Xs_Warmup,
#             tracer_Asteroid_Xs_Warmup,
#             tracer_Dts_Warmup,
#             tracer_Observer_Positions,
#             tracer_RAs,
#             tracer_Decs,
#             tracer_Observed_Planet_Xs,
#             tracer_Observed_Asteroid_Xs,
#         ),
#     )[1]
#     tracer_xs = tmp[0]
#     tracer_vs = tmp[1]
#     tracer_ras = tmp[2]
#     tracer_decs = tmp[3]
#     tracer_resids = tmp[4]

#     def _massive_scan_func(carry, scan_over):
#         ind = scan_over
#         x, v = gj_integrate_multiple(
#             x0=massive_xs,
#             v0=massive_vs,
#             gms=massive_gms,
#             valid_steps=massive_Valid_Steps[ind],
#             b_jk=GJ8_B,
#             a_jk=GJ8_A,
#             t0=massive_Init_Times[ind],
#             times=massive_Jump_Times[ind],
#             planet_xs=massive_Planet_Xs[ind],
#             planet_vs=massive_Planet_Vs[ind],
#             planet_as=massive_Planet_As[ind],
#             asteroid_xs=massive_Asteroid_Xs[ind],
#             planet_xs_warmup=massive_Planet_Xs_Warmup[ind],
#             asteroid_xs_warmup=massive_Asteroid_Xs_Warmup[ind],
#             dts_warmup=massive_Dts_Warmup[ind],
#             warmup_C=Y4_C,
#             warmup_D=Y4_D,
#             planet_gms=STANDARD_PLANET_GMS,
#             asteroid_gms=STANDARD_ASTEROID_GMS,
#             use_GR=True,
#         )

#         # add the first position back in
#         x = jnp.concatenate((massive_xs[ind][None, :], x[ind, :, :]))
#         v = jnp.concatenate((massive_vs[ind][None, :], v[ind, :, :]))

#         # calc_ra, calc_dec = on_sky(
#         #     xs=x,
#         #     vs=v,
#         #     gms=jnp.zeros(x.shape[0]),
#         #     times=jnp.concatenate(
#         #         (massive_Init_Times[ind][None], massive_Jump_Times[ind])
#         #     ),
#         #     observer_positions=massive_Observer_Positions[ind],
#         #     planet_params=planet_params,
#         #     asteroid_params=asteroid_params,
#         #     planet_gms=planet_gms,
#         #     asteroid_gms=asteroid_gms,
#         # )
#         calc_ra, calc_dec = on_sky(
#             xs=x,
#             vs=v,
#             gms=jnp.ones(x.shape[0])*massive_gms[ind],
#             observer_positions=massive_Observer_Positions[ind],
#             planet_xs=massive_Observed_Planet_Xs[ind],
#             asteroid_xs=massive_Observed_Asteroid_Xs[ind],
#             planet_gms=planet_gms,
#             asteroid_gms=asteroid_gms,
#         )

#         resids = sky_error(
#             calc_ra=calc_ra,
#             calc_dec=calc_dec,
#             true_ra=massive_RAs[ind],
#             true_dec=massive_Decs[ind],
#         )

#         return None, (x, v, calc_ra, calc_dec, resids)

#     tmp = jax.lax.scan(
#         _massive_scan_func,
#         None,
#         jnp.arange(massive_xs.shape[0]),
#     )[1]
#     massive_xs = tmp[0]
#     massive_vs = tmp[1]
#     massive_RAs = tmp[2]
#     massive_Decs = tmp[3]
#     massive_resids = tmp[4]

#     loglike = 0

#     sigma2 = tracer_Astrometric_Uncertainties**2
#     loglike += -0.5 * jnp.sum(tracer_resids**2 / sigma2 )

#     sigma2 = massive_Astrometric_Uncertainties**2
#     loglike += -0.5 * jnp.sum(massive_resids**2 / sigma2 )

#     return tracer_resids, massive_resids, loglike


# Q = _resids(**free_params, **fixed_params, **padded_supporting_data)

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import mpmath
from typing import List, Tuple
from tqdm import tqdm


def create_gauss_radau_spacings(internal_points: int):
    # for ias15 H spacings, internal_points = 7
    # matches https://archive.org/details/gaussianquadratu00stro/page/340/mode/2up
    mpmath.mp.dps = 75

    n = internal_points + 1  # include the endpoint
    f = lambda x: (mpmath.legendre(n - 1, x) + mpmath.legendre(n, x)) / (x + 1)

    slices = mpmath.linspace(-1, 1, 1000)

    sols = []
    for i in tqdm(range(1, len(slices) - 1)):
        try:
            s = mpmath.findroot(
                f,
                (slices[i], slices[i + 1]),
                solver="secant",
                tol=mpmath.mpf("1e-50"),
            )
            possibly_new = s not in sols
            if possibly_new:
                for sol in sols:
                    if mpmath.fabs(sol - s) < mpmath.mpf("1e-50"):
                        possibly_new = False
                        break
            if possibly_new:
                assert mpmath.fabs(f(s)) < mpmath.mpf("1e-50")
                sols.append(s)

            del s
        except:
            pass

    assert len(sols) == n - 1

    sols.sort()
    sols = [((s + 1) / 2) for s in sols]  # rescale to 0, 1 from the range -1, 1
    sols = [mpmath.mpf(0)] + sols  # add the endpoint
    return sols


def create_ias15_r_array(h: List[mpmath.mpf]) -> List[mpmath.mpf]:
    mpmath.mp.dps = 75
    n = len(h)
    r = []

    # Calculate all pairwise differences where j > k
    for j in range(1, n):
        for k in range(j):
            r.append(h[j] - h[k])

    return r


def create_ias15_c_d_arrays(
    h: List[mpmath.mpf],
) -> Tuple[List[mpmath.mpf], List[mpmath.mpf]]:
    mpmath.mp.dps = 75

    n = len(h)

    size = 1  # Initial element
    for j in range(2, n - 1):
        size += j  # j elements per iteration (1 + (j-2) + 1)

    c = [mpmath.mpf(0)] * size
    d = [mpmath.mpf(0)] * size

    # Initial values
    c[0] = -h[1]
    d[0] = h[1]

    l = 0  # Current position in arrays

    # Main recurrence relations
    for j in range(2, n - 1):
        # First element for this j
        l += 1
        c[l] = -h[j] * c[l - j + 1]
        d[l] = h[1] * d[l - j + 1]

        # Middle elements
        for k in range(2, j):
            l += 1
            c[l] = c[l - j] - h[j] * c[l - j + 1]
            d[l] = d[l - j] + h[k] * d[l - j + 1]

        # Last element for this j
        l += 1
        c[l] = c[l - j] - h[j]
        d[l] = d[l - j] + h[j]

    return c, d


def create_ias15_constants(
    n_internal_points: int,
) -> Tuple[List[mpmath.mpf], List[mpmath.mpf], List[mpmath.mpf], List[mpmath.mpf]]:
    mpmath.mp.dps = 75
    h = create_gauss_radau_spacings(n_internal_points)
    r = create_ias15_r_array(h)
    c, d = create_ias15_c_d_arrays(h)

    return h, r, c, d


def create_yoshida_coeffs(Ws):
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

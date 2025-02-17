"""Generate the high-precision coefficients for IAS15-style integrators."""

import jax

jax.config.update("jax_enable_x64", True)

import mpmath
from tqdm import tqdm


def create_gauss_radau_spacings(internal_points: int) -> list[mpmath.mpf]:
    """Compute the spacings for the Gauss-Radau quadrature rule for an arbitrary number of internal points.

    IAS15 uses 7 internal points, which results in a 15th order integrator. If we wanted
    a higher or lower order integrator, we could change the number of internal points.
    But, then we need to know where to actually evaluate those points: this function
    uses the mpmath library to compute those locations, which are the roots of the
    Legendre polynomial of order n-1 and n, divided by x+1.

    Confirmed that the resulting values match those tabulated in Table 12 of
    `Stroud and Secrest 1966 <https://archive.org/details/gaussianquadratu00stro/page/340/mode/2up>`_.

    Args:
        internal_points (int):
            The number of internal points to use.

    Returns:
        list[mpmath.mpf]:
            The spacings for the Gauss-Radau quadrature rule. The 'H' array in IAS15.
    """
    # for ias15 H spacings, internal_points = 7
    # matches https://archive.org/details/gaussianquadratu00stro/page/340/mode/2up
    mpmath.mp.dps = 75

    n = internal_points + 1  # include the endpoint

    def f(x):
        return (mpmath.legendre(n - 1, x) + mpmath.legendre(n, x)) / (x + 1)

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
        except Exception:
            pass

    assert len(sols) == n - 1

    sols.sort()
    sols = [((s + 1) / 2) for s in sols]  # rescale to 0, 1 from the range -1, 1
    sols = [mpmath.mpf(0), *sols]  # add the endpoint
    return sols


def create_iasnn_r_array(h: list[mpmath.mpf]) -> list[mpmath.mpf]:
    """Create the equivalent of the 'R' array in IAS15 for an arbitrary-order integrator.

    Args:
        h (list[mpmath.mpf]):
            The spacings for the Gauss-Radau quadrature rule.

    Returns:
        list[mpmath.mpf]:
            The 'R' array for the integrator.
    """
    mpmath.mp.dps = 75
    n = len(h)
    r = []

    # Calculate all pairwise differences where j > k
    for j in range(1, n):
        for k in range(j):
            r.append(h[j] - h[k])

    return r


def create_iasnn_c_d_arrays(
    h: list[mpmath.mpf],
) -> tuple[list[mpmath.mpf], list[mpmath.mpf]]:
    """Create the equivalent of the 'C' and 'D' arrays in IAS15 for an arbitrary-order integrator.

    Args:
        h (list[mpmath.mpf]):
            The spacings for the Gauss-Radau quadrature rule.

    Returns:
        tuple[list[mpmath.mpf], list[mpmath.mpf]]:
            The 'C' and 'D' arrays for the integrator.
    """
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

    idx = 0  # Current position in arrays

    # Main recurrence relations
    for j in range(2, n - 1):
        # First element for this j
        idx += 1
        c[idx] = -h[j] * c[idx - j + 1]
        d[idx] = h[1] * d[idx - j + 1]

        # Middle elements
        for k in range(2, j):
            idx += 1
            c[idx] = c[idx - j] - h[j] * c[idx - j + 1]
            d[idx] = d[idx - j] + h[k] * d[idx - j + 1]

        # Last element for this j
        idx += 1
        c[idx] = c[idx - j] - h[j]
        d[idx] = d[idx - j] + h[j]

    return c, d


def create_iasnn_constants(
    n_internal_points: int,
) -> tuple[list[mpmath.mpf], list[mpmath.mpf], list[mpmath.mpf], list[mpmath.mpf]]:
    """Create the equivalent of the 'H', 'R', 'C', and 'D' arrays in IAS15 for an arbitrary-order integrator.

    Just a wrapper around create_gauss_radau_spacings, create_iasnn_r_array, and
    create_iasnn_c_d_arrays.

    Args:
        n_internal_points (int):
            The number of internal points to use.

    Returns:
        tuple[list[mpmath.mpf], list[mpmath.mpf], list[mpmath.mpf], list[mpmath.mpf]]:
            The 'H', 'R', 'C', and 'D' arrays for the integrator.
    """
    mpmath.mp.dps = 75
    h = create_gauss_radau_spacings(n_internal_points)
    r = create_iasnn_r_array(h)
    c, d = create_iasnn_c_d_arrays(h)

    return h, r, c, d


# unused now that the leapfrog integrator is gone
# def create_yoshida_coeffs(Ws):
#     """
#     Convert the Ws from Tables 1 and 2 of Yoshida (1990) into C and D coefficients

#     Saving this for later reference, but it isn't called anymore- values were
#     precomputed and saved in jorbit.data.constants.

#     Parameters:
#         WS (jnp.ndarray):
#             An array of "W" values from Tables 1 and 2 of Yoshida (1990)

#     Returns:
#         Tuple[jnp.ndarray, jnp.ndarray]:
#         C (jnp.ndarray):
#             The coefficients for the mid-step position updates
#         D (jnp.ndarray):
#             The coefficients for the mid-step velocity updates
#     """
#     w0 = 1 - 2 * (jnp.sum(Ws))
#     w = jnp.concatenate((jnp.array([w0]), Ws))

#     Ds = jnp.zeros(2 * len(Ws) + 1)
#     Ds = Ds.at[: len(Ws)].set(Ws[::-1])
#     Ds = Ds.at[len(Ws)].set(w0)
#     Ds = Ds.at[len(Ws) + 1 :].set(Ws)

#     Cs = jnp.zeros(2 * len(Ws) + 2)
#     for i in range(len(w) - 1):
#         Cs = Cs.at[i + 1].set(0.5 * (w[len(w) - 1 - i] + w[len(w) - 2 - i]))

#     Cs = Cs.at[int(len(Cs) / 2) :].set(Cs[: int(len(Cs) / 2)][::-1])
#     Cs = Cs.at[0].set(0.5 * w[-1])
#     Cs = Cs.at[-1].set(0.5 * w[-1])

#     # to do it at extended precision, use Decimal:
#     # tmp = 0
#     # for i in Ws:
#     #     tmp += i
#     # w0 = 1 - 2 * tmp
#     # w = [w0] + Ws

#     # Ds = [0]*(2 * len(Ws) + 1)
#     # Ds[:len(Ws)] = Ws[::-1]
#     # Ds[len(Ws)] = w0
#     # Ds[len(Ws) + 1:] = Ws

#     # Cs = [0]*(2 * len(Ws) + 2)
#     # for i in range(len(w) - 1):
#     #     Cs[i + 1] = Decimal(0.5) * (w[len(w) - 1 - i] + w[len(w) - 2 - i])
#     # Cs[int(len(Cs) / 2):] = Cs[: int(len(Cs) / 2)][::-1]
#     # Cs[0] = Decimal(0.5) * w[-1]
#     # Cs[-1] = Decimal(0.5) * w[-1]

#     return jnp.array(Cs), jnp.array(Ds)

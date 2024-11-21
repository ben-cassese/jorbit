import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


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

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from functools import partial
from typing import Any, Tuple, Optional


@jax.tree_util.register_pytree_node_class
class DoubleDouble:
    """A double-double precision number representation for JAX.

    Represents a high-precision number as a sum of two IEEE doubles (hi + lo),
    where |lo| <= ulp(hi)/2. This gives roughly twice the precision of a regular double.
    """

    def __init__(self, hi, lo=None):
        """Initialize a DoubleDouble number.

        Args:
            hi: High part (jnp.ndarray)
            lo: Low part (jnp.ndarray, optional). If None, lo is set to 0
        """
        self.hi = hi
        self.lo = jnp.zeros_like(hi) if lo is None else lo

    # @staticmethod
    # def _two_sum(a: jnp.ndarray, b: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    #     s = a + b
    #     v = s - a
    #     e = (a - (s - v)) + (b - v)
    #     return s, e

    # @staticmethod
    # def _split(a: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    #     t = 2**27 * a
    #     a_hi = t - (t - a)
    #     a_lo = a - a_hi
    #     return a_hi, a_lo

    # @staticmethod
    # def _two_prod(a: jnp.ndarray, b: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    #     # https://andrewthall.org/papers/df64_qf128.pdf
    #     x = a * b

    #     a_hi, a_lo = DoubleDouble._split(a)
    #     b_hi, b_lo = DoubleDouble._split(b)

    #     err1 = x - (a_hi * b_hi)
    #     err2 = err1 - (a_lo * b_hi)
    #     err3 = err2 - (a_hi * b_lo)

    #     y = (a_lo * b_lo) - err3

    #     return x, y

    @staticmethod
    def _mul12(x: jnp.ndarray, y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # mul12 from https://csclub.uwaterloo.ca/~pbarfuss/dekker1971.pdf
        constant = 2**27 + 1
        p = x * constant
        hx = x - p + p
        tx = x - hx

        p = y * constant
        hy = y - p + p
        ty = y - hy

        p = hx * hy
        q = hx * ty + tx * hy
        z = p + q
        zz = p - z + q + tx * ty

        return DoubleDouble(z, zz)

    @classmethod
    def from_string(cls, s: str) -> "DoubleDouble":
        """Alternative method using decimal arithmetic for string conversion.

        This method is useful when numpy float128 is not available or for even higher precision.

        Args:
            s: String representation of a number

        Returns:
            DoubleDouble: High-precision representation of the number
        """
        from decimal import Decimal, getcontext

        # Set precision high enough to capture all digits
        getcontext().prec = 40

        # Convert string to Decimal
        d = Decimal(s)

        # Extract hi part (first ~15-16 digits)
        hi = float(str(d))

        # Compute lo part
        lo = float(str(d - Decimal(str(hi))))

        # Convert to JAX arrays
        return cls(jnp.array(hi), jnp.array(lo))

    def __str__(self) -> str:
        from decimal import Decimal

        hi_dec = Decimal(str(float(self.hi)))
        lo_dec = Decimal(str(float(self.lo)))
        return str(hi_dec + lo_dec)

    def __repr__(self):
        return f"DoubleDouble({self.hi}, {self.lo})"

    @jax.jit
    def __add__(self, other):
        # add2 from https://csclub.uwaterloo.ca/~pbarfuss/dekker1971.pdf
        r = self.hi + other.hi
        s = jnp.where(
            jnp.abs(self.hi) > jnp.abs(other.hi),
            self.hi - r + other.hi + other.lo + self.lo,
            other.hi - r + self.hi + self.lo + other.lo,
        )
        z = r + s
        zz = r - z + s
        return DoubleDouble(z, zz)

    def __neg__(self):
        return DoubleDouble(-self.hi, -self.lo)

    @jax.jit
    def __sub__(self, other):
        # sub2 from https://csclub.uwaterloo.ca/~pbarfuss/dekker1971.pdf
        r = self.hi - other.hi
        s = jnp.where(
            jnp.abs(self.hi) > jnp.abs(other.hi),
            self.hi - r - other.hi - other.lo + self.lo,
            -other.hi - r + self.hi + self.lo - other.lo,
        )
        z = r + s
        zz = r - z + s
        return DoubleDouble(z, zz)
        self

    @jax.jit
    def __mul__(self, other):
        # mul2 from https://csclub.uwaterloo.ca/~pbarfuss/dekker1971.pdf
        c = DoubleDouble._mul12(self.hi, other.hi)
        cc = self.hi * other.lo + self.lo * other.hi + c.lo

        z = c.hi + cc
        zz = c.hi - z + cc

        return DoubleDouble(z, zz)

    @jax.jit
    def __truediv__(self, other):
        # div2 from https://csclub.uwaterloo.ca/~pbarfuss/dekker1971.pdf
        c = self.hi / other.hi
        u = DoubleDouble._mul12(c, other.hi)
        cc = (self.hi - u.hi - u.lo + self.lo - c * other.lo) / other.hi
        z = c + cc
        zz = c - z + cc
        return DoubleDouble(z, zz)

    @jax.jit
    def __abs__(self):
        new_hi = jnp.where(self.hi < 0, -self.hi, self.hi)
        new_lo = jnp.where(self.hi < 0, -self.lo, self.lo)
        return DoubleDouble(new_hi, new_lo)
        # return jax.lax.cond(
        #     self.hi < 0,
        #     lambda x: DoubleDouble(-self.hi, -self.lo),
        #     lambda x: self,
        #     operand=None,
        # )

    @jax.jit
    def dd_exp(self):
        raise
        # not strictly a DoubleDouble-compatible operation:
        # just exp the hi and lo parts separately, then add them
        # with a DoubleDouble-compatible addition

        # can do better w/ a Taylor series?

        exp_hi = jnp.exp(self.hi)
        exp_lo = jnp.exp(self.lo)

        # from here, same as __mul__
        c = DoubleDouble._mul12(self.hi, other.hi)
        cc = self.hi * other.lo + self.lo * other.hi + c.lo

        z = c.hi + cc
        zz = c.hi - z + cc

    @partial(jax.jit, static_argnums=(1,))
    def dd_max(self, axis: Optional[int] = None):
        hi_max = jnp.max(self.hi, axis=axis)
        max_mask = self.hi == hi_max
        lo_max = jnp.max(jnp.where(max_mask, self.lo, -jnp.inf), axis=axis)
        return DoubleDouble(hi_max, lo_max)

    def tree_flatten(self):
        """Implementation for JAX pytree."""
        children = (self.hi, self.lo)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Implementation for JAX pytree."""
        return cls(*children)

"""Experimental DoubleDouble precision arithmetic in JAX."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
class DoubleDouble:
    """An experimental class for 'DoubleDouble' precision arthmetic.

    This creates a Jax pytree object that stores two jnp.ndarrays, hi and lo, which are
    the high and low parts of a double-double precision array. Basic arithmetic
    operations are overloaded to use functions that respect the double-double precision
    rules. This is not compensated summation, but summation at "DoubleDouble" precision.

    Attributes:
        hi (jnp.ndarray):
            High part.
        lo: (jnp.ndarray):
            Low part.
    """

    def __init__(self, hi: jnp.ndarray, lo: jnp.ndarray | None = None) -> None:
        """Initialize a DoubleDouble number.

        Args:
            hi: High part (jnp.ndarray)
            lo: Low part (jnp.ndarray, optional). If None, lo is set to 0
        """
        if isinstance(hi, (int, float)) and (lo is None):
            self.hi, self.lo = DoubleDouble._split(jnp.array(hi))
        else:
            self.hi = jnp.array(hi)
            self.lo = jnp.zeros_like(hi) if lo is None else lo

    @staticmethod
    def _split(a: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Split a 64-bit floating point number into high and low components."""
        t = (2**27 + 1) * a
        a_hi = t - (t - a)
        a_lo = a - a_hi
        return a_hi, a_lo

    @staticmethod
    def _two_sum(a: jnp.ndarray, b: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Basic two-sum algorithm."""
        s = a + b
        v = s - a
        e = (a - (s - v)) + (b - v)
        return s, e

    @staticmethod
    def _mul12(x: jnp.ndarray, y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """The mul12 algorithm from `Dekker 1971 <https://csclub.uwaterloo.ca/~pbarfuss/dekker1971.pdf>`_."""
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
    def from_string(cls, s: str) -> DoubleDouble:
        """Create a DoubleDouble number from a string, similar to mpmath.mpf.

        Args:
            s (str):
                String representation of a number.

        Returns:
            DoubleDouble:
                The DoubleDouble representation.
        """
        assert isinstance(s, str)
        from decimal import Decimal, getcontext

        getcontext().prec = 50

        d = Decimal(s)
        hi = float(d)
        # Compute low part using exact subtraction
        lo = float(d - Decimal(hi))
        # Normalize the components
        hi, lo = DoubleDouble._two_sum(hi, lo)
        return cls(jnp.array(hi), jnp.array(lo))

    def __str__(self) -> str:
        """String representation of the DoubleDouble array."""
        return f"{self.hi} + {self.lo}"

    def __repr__(self) -> str:
        """Representation of the DoubleDouble array."""
        return f"DoubleDouble({self.hi}, {self.lo})"

    def __getitem__(self, index: int) -> DoubleDouble:
        """Get an item from the DoubleDouble array."""
        return DoubleDouble(self.hi[index], self.lo[index])

    def __setitem__(self, index: int, value: DoubleDouble) -> None:
        """Set an item in the DoubleDouble array (note: mutable, unlike jnp.ndarray)."""
        self.hi = self.hi.at[index].set(value.hi)
        self.lo = self.lo.at[index].set(value.lo)

    # @jax.jit
    def __add__(self, other: DoubleDouble) -> DoubleDouble:
        """Add two DoubleDouble numbers.

        Implementation of add2 from `Dekker 1971 <https://csclub.uwaterloo.ca/~pbarfuss/dekker1971.pdf>`_.
        """
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

    # @jax.jit
    def __neg__(self) -> DoubleDouble:
        """Negate a DoubleDouble number."""
        return DoubleDouble(-self.hi, -self.lo)

    # @jax.jit
    def __sub__(self, other: DoubleDouble) -> DoubleDouble:
        """Subtract two DoubleDouble numbers.

        Implementation of sub2 from `Dekker 1971 <https://csclub.uwaterloo.ca/~pbarfuss/dekker1971.pdf>`_.
        """
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

    # @jax.jit
    def __mul__(self, other: DoubleDouble) -> DoubleDouble:
        """Multiply two DoubleDouble numbers.

        Implementation of mul2 from `Dekker 1971 <https://csclub.uwaterloo.ca/~pbarfuss/dekker1971.pdf>`_.
        """
        # mul2 from https://csclub.uwaterloo.ca/~pbarfuss/dekker1971.pdf
        c = DoubleDouble._mul12(self.hi, other.hi)
        cc = self.hi * other.lo + self.lo * other.hi + c.lo

        z = c.hi + cc
        zz = c.hi - z + cc

        return DoubleDouble(z, zz)

    # @jax.jit
    def __truediv__(self, other: DoubleDouble) -> DoubleDouble:
        """Divide two DoubleDouble numbers.

        Implementation of div2 from `Dekker 1971 <https://csclub.uwaterloo.ca/~pbarfuss/dekker1971.pdf>`_.
        """
        # div2 from https://csclub.uwaterloo.ca/~pbarfuss/dekker1971.pdf
        c = self.hi / other.hi
        u = DoubleDouble._mul12(c, other.hi)
        cc = (self.hi - u.hi - u.lo + self.lo - c * other.lo) / other.hi
        z = c + cc
        zz = c - z + cc
        return DoubleDouble(z, zz)

    # @jax.jit
    def __abs__(self) -> DoubleDouble:
        """Absolute value of a DoubleDouble number."""
        new_hi = jnp.where(self.hi < 0, -self.hi, self.hi)
        new_lo = jnp.where(self.hi < 0, -self.lo, self.lo)
        return DoubleDouble(new_hi, new_lo)

    def __lt__(self, other: DoubleDouble) -> bool:
        """Less than comparison of two DoubleDouble numbers."""
        return (self.hi < other.hi) | ((self.hi == other.hi) & (self.lo < other.lo))

    def __le__(self, other: DoubleDouble) -> bool:
        """Less than or equal to comparison of two DoubleDouble numbers."""
        return (self.hi < other.hi) | ((self.hi == other.hi) & (self.lo <= other.lo))

    def __gt__(self, other: DoubleDouble) -> bool:
        """Greater than comparison of two DoubleDouble numbers."""
        return (self.hi > other.hi) | ((self.hi == other.hi) & (self.lo > other.lo))

    def __ge__(self, other: DoubleDouble) -> bool:
        """Greater than or equal to comparison of two DoubleDouble numbers."""
        return (self.hi > other.hi) | ((self.hi == other.hi) & (self.lo >= other.lo))

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the DoubleDouble array."""
        return self.hi.shape

    def tree_flatten(self) -> tuple:
        """Implementation for JAX pytree."""
        children = (self.hi, self.lo)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple) -> DoubleDouble:
        """Implementation for JAX pytree."""
        return cls(*children)


# @jax.jit
def dd_max(x: DoubleDouble, axis: int | None = None) -> DoubleDouble:
    """Sort-of implements jnp.max on a DoubleDouble array.

    Args:
        x: DoubleDouble array
        axis: Axis to reduce over

    Returns:
        DoubleDouble: The maximum value
    """
    hi_max = jnp.max(x.hi, axis=axis)
    max_mask = x.hi == hi_max
    lo_max = jnp.max(jnp.where(max_mask, x.lo, -jnp.inf), axis=axis)
    return DoubleDouble(hi_max, lo_max)


# @partial(jax.jit, static_argnames=("axis",))
def dd_sum(x: DoubleDouble, axis: int | None = None) -> DoubleDouble:
    """Sort-of implements jnp.sum on a DoubleDouble array.

    Args:
        x (DoubleDouble):
            The DoubleDouble array to sum.
        axis (int | None):
            The axis to sum over. If None, the array is flattened.

    Returns:
        DoubleDouble: The sum of the array along the given axis.
    """
    # needed to respect DoubleDouble addition rules when doing sums
    # again- this is *not* compensated summation, but summation at "DoubleDouble" precision
    if axis is None:
        x = DoubleDouble(x.hi.flatten(), x.lo.flatten())
        axis = 0

    # Move the axis to be summed to the front
    transposed = DoubleDouble(jnp.swapaxes(x.hi, 0, axis), jnp.swapaxes(x.lo, 0, axis))

    def scan_fn(carry: DoubleDouble, x: DoubleDouble) -> tuple[DoubleDouble, None]:
        return carry + x, None

    result, _ = jax.lax.scan(scan_fn, transposed[0], transposed[1:])

    return result


# @jax.jit
def dd_sqrt(x: DoubleDouble) -> DoubleDouble:
    """Sort-of implements jnp.sqrt on a DoubleDouble array.

    Uses sqrt2 from `Dekker 1971 <https://csclub.uwaterloo.ca/~pbarfuss/dekker1971.pdf>`_.

    Args:
        x (DoubleDouble):
            The DoubleDouble array to take the square root of.

    Returns:
        DoubleDouble:
            The square root of the array.
    """
    # sqrt2 from https://csclub.uwaterloo.ca/~pbarfuss/dekker1971.pdf
    c = jnp.sqrt(x.hi)
    u = DoubleDouble._mul12(c, c)
    c_lo = (x.hi - u.hi - u.lo + x.lo) / (2 * c)
    y = c + c_lo
    yy = c - y + c_lo
    return DoubleDouble(y, yy)


# @partial(jax.jit, static_argnames=("axis",))
def dd_norm(x: DoubleDouble, axis: int | None = None) -> DoubleDouble:
    """Sort-of implements jnp.linalg.norm on a DoubleDouble array.

    Uses dd_sum and dd_sqrt.

    Args:
        x (DoubleDouble):
            The DoubleDouble array to take the norm of.
        axis (int | None):
            The axis to take the norm over. If None, the array is flattened.

    Returns:
        DoubleDouble:
            The norm of the array.
    """
    if axis is None:
        x = DoubleDouble(x.hi.flatten(), x.lo.flatten())
        axis = 0
    return dd_sqrt(dd_sum(x * x, axis=axis))


# @staticmethod
# def _two_sum(a: jnp.ndarray, b: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
#     s = a + b
#     v = s - a
#     e = (a - (s - v)) + (b - v)
#     return s, e

# @staticmethod
# def _split(a: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
#     t = (2**27 + 1) * a
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

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

    @staticmethod
    def _ensure_dd(x: DoubleDouble | jnp.ndarray | float | int) -> DoubleDouble:
        """Wrap a value as DoubleDouble if it isn't one already."""
        if isinstance(x, DoubleDouble):
            return x
        return DoubleDouble(jnp.asarray(x, dtype=jnp.float64))

    # @jax.jit
    def __add__(self, other: DoubleDouble) -> DoubleDouble:
        """Add two DoubleDouble numbers.

        Implementation of add2 from `Dekker 1971 <https://csclub.uwaterloo.ca/~pbarfuss/dekker1971.pdf>`_.
        """
        other = DoubleDouble._ensure_dd(other)
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

    def __radd__(self, other: DoubleDouble | jnp.ndarray | float | int) -> DoubleDouble:
        """Right-hand addition."""
        return DoubleDouble._ensure_dd(other) + self

    # @jax.jit
    def __sub__(self, other: DoubleDouble) -> DoubleDouble:
        """Subtract two DoubleDouble numbers.

        Implementation of sub2 from `Dekker 1971 <https://csclub.uwaterloo.ca/~pbarfuss/dekker1971.pdf>`_.
        """
        other = DoubleDouble._ensure_dd(other)
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

    def __rsub__(self, other: DoubleDouble | jnp.ndarray | float | int) -> DoubleDouble:
        """Right-hand subtraction."""
        return DoubleDouble._ensure_dd(other) - self

    # @jax.jit
    def __mul__(self, other: DoubleDouble) -> DoubleDouble:
        """Multiply two DoubleDouble numbers.

        Implementation of mul2 from `Dekker 1971 <https://csclub.uwaterloo.ca/~pbarfuss/dekker1971.pdf>`_.
        """
        other = DoubleDouble._ensure_dd(other)
        # mul2 from https://csclub.uwaterloo.ca/~pbarfuss/dekker1971.pdf
        c = DoubleDouble._mul12(self.hi, other.hi)
        cc = self.hi * other.lo + self.lo * other.hi + c.lo

        z = c.hi + cc
        zz = c.hi - z + cc

        return DoubleDouble(z, zz)

    def __rmul__(self, other: DoubleDouble | jnp.ndarray | float | int) -> DoubleDouble:
        """Right-hand multiplication."""
        return DoubleDouble._ensure_dd(other) * self

    # @jax.jit
    def __truediv__(self, other: DoubleDouble) -> DoubleDouble:
        """Divide two DoubleDouble numbers.

        Implementation of div2 from `Dekker 1971 <https://csclub.uwaterloo.ca/~pbarfuss/dekker1971.pdf>`_.
        """
        other = DoubleDouble._ensure_dd(other)
        # div2 from https://csclub.uwaterloo.ca/~pbarfuss/dekker1971.pdf
        c = self.hi / other.hi
        u = DoubleDouble._mul12(c, other.hi)
        cc = (self.hi - u.hi - u.lo + self.lo - c * other.lo) / other.hi
        z = c + cc
        zz = c - z + cc
        return DoubleDouble(z, zz)

    def __rtruediv__(
        self, other: DoubleDouble | jnp.ndarray | float | int
    ) -> DoubleDouble:
        """Right-hand division."""
        return DoubleDouble._ensure_dd(other) / self

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
def dd_sum(
    x: DoubleDouble, axis: int | None = None, where: jnp.ndarray | None = None
) -> DoubleDouble:
    """Sort-of implements jnp.sum on a DoubleDouble array.

    Args:
        x (DoubleDouble):
            The DoubleDouble array to sum.
        axis (int | None):
            The axis to sum over. If None, the array is flattened.
        where (jnp.ndarray | None):
            Boolean mask. Elements where mask is False are treated as zero.

    Returns:
        DoubleDouble: The sum of the array along the given axis.
    """
    # needed to respect DoubleDouble addition rules when doing sums
    # again- this is *not* compensated summation, but summation at "DoubleDouble" precision
    if where is not None:
        zero = DoubleDouble(jnp.zeros_like(x.hi), jnp.zeros_like(x.lo))
        x = dd_where(where, x, zero)

    if axis is None:
        x = DoubleDouble(x.hi.flatten(), x.lo.flatten())
        axis = 0

    # Move the axis to be summed to the front, preserving order of remaining axes
    transposed = DoubleDouble(jnp.moveaxis(x.hi, axis, 0), jnp.moveaxis(x.lo, axis, 0))

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


def dd_cross(a: DoubleDouble, b: DoubleDouble, axis: int = -1) -> DoubleDouble:
    """Cross product of two DoubleDouble arrays along the given axis.

    Both arrays must have size 3 along the specified axis.

    Args:
        a: First DoubleDouble array.
        b: Second DoubleDouble array.
        axis: Axis along which to compute the cross product (must have size 3).

    Returns:
        DoubleDouble: The cross product.
    """
    # Move target axis to the last position for indexing
    a_hi = jnp.moveaxis(a.hi, axis, -1)
    a_lo = jnp.moveaxis(a.lo, axis, -1)
    b_hi = jnp.moveaxis(b.hi, axis, -1)
    b_lo = jnp.moveaxis(b.lo, axis, -1)

    a0 = DoubleDouble(a_hi[..., 0], a_lo[..., 0])
    a1 = DoubleDouble(a_hi[..., 1], a_lo[..., 1])
    a2 = DoubleDouble(a_hi[..., 2], a_lo[..., 2])
    b0 = DoubleDouble(b_hi[..., 0], b_lo[..., 0])
    b1 = DoubleDouble(b_hi[..., 1], b_lo[..., 1])
    b2 = DoubleDouble(b_hi[..., 2], b_lo[..., 2])

    c0 = a1 * b2 - a2 * b1
    c1 = a2 * b0 - a0 * b2
    c2_val = a0 * b1 - a1 * b0

    # Stack back along the last axis, then move back to original position
    result_hi = jnp.stack([c0.hi, c1.hi, c2_val.hi], axis=-1)
    result_lo = jnp.stack([c0.lo, c1.lo, c2_val.lo], axis=-1)
    result_hi = jnp.moveaxis(result_hi, -1, axis)
    result_lo = jnp.moveaxis(result_lo, -1, axis)

    return DoubleDouble(result_hi, result_lo)


def dd_dot(a: DoubleDouble, b: DoubleDouble, axis: int = -1) -> DoubleDouble:
    """Dot product of two DoubleDouble arrays along the given axis.

    Args:
        a: First DoubleDouble array.
        b: Second DoubleDouble array.
        axis: Axis along which to compute the dot product.

    Returns:
        DoubleDouble: The dot product (with the given axis reduced).
    """
    return dd_sum(a * b, axis=axis)


def dd_where(condition: jnp.ndarray, x: DoubleDouble, y: DoubleDouble) -> DoubleDouble:
    """Element-wise selection between two DoubleDouble arrays.

    Args:
        condition: Boolean array for selection.
        x: Values where condition is True.
        y: Values where condition is False.

    Returns:
        DoubleDouble with elements selected from x or y.
    """
    return DoubleDouble(
        jnp.where(condition, x.hi, y.hi),
        jnp.where(condition, x.lo, y.lo),
    )


def dd_concatenate(arrays: list[DoubleDouble], axis: int = 0) -> DoubleDouble:
    """Concatenate a sequence of DoubleDouble arrays along an axis.

    Args:
        arrays: List of DoubleDouble arrays to concatenate.
        axis: Axis along which to concatenate.

    Returns:
        DoubleDouble: The concatenated array.
    """
    return DoubleDouble(
        jnp.concatenate([a.hi for a in arrays], axis=axis),
        jnp.concatenate([a.lo for a in arrays], axis=axis),
    )


def dd_broadcast_to(x: DoubleDouble, shape: tuple[int, ...]) -> DoubleDouble:
    """Broadcast a DoubleDouble array to a given shape.

    Args:
        x: DoubleDouble array to broadcast.
        shape: Target shape.

    Returns:
        DoubleDouble: The broadcasted array.
    """
    return DoubleDouble(
        jnp.broadcast_to(x.hi, shape),
        jnp.broadcast_to(x.lo, shape),
    )


def dd_zeros(shape: tuple[int, ...]) -> DoubleDouble:
    """Create a DoubleDouble array of zeros.

    Args:
        shape: Shape of the array.

    Returns:
        DoubleDouble: Zero array.
    """
    return DoubleDouble(jnp.zeros(shape), jnp.zeros(shape))


def dd_zeros_like(x: DoubleDouble) -> DoubleDouble:
    """Create a DoubleDouble array of zeros with the same shape as x.

    Args:
        x: DoubleDouble array whose shape to match.

    Returns:
        DoubleDouble: Zero array with same shape.
    """
    return DoubleDouble(jnp.zeros_like(x.hi), jnp.zeros_like(x.lo))


def dd_exp(x: DoubleDouble) -> DoubleDouble:
    """Approximate exp for DoubleDouble using correction term.

    Computes exp(hi + lo) = exp(hi) * (1 + lo) for small lo,
    which is accurate when |lo| << 1 (always true for well-formed DD).

    Args:
        x: DoubleDouble array.

    Returns:
        DoubleDouble: exp(x) to DoubleDouble precision.
    """
    e_hi = jnp.exp(x.hi)
    # exp(hi + lo) = exp(hi) * exp(lo) ≈ exp(hi) * (1 + lo) for small lo
    # More precisely: use mul12 for the product exp(hi) * (1 + lo)
    result_hi = DoubleDouble(e_hi)
    correction = DoubleDouble(jnp.ones_like(x.lo)) + DoubleDouble(x.lo)
    return result_hi * correction


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

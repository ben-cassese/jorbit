"""Test that the packed to unpacked designation translator is consistent."""

from jorbit.mpchecker import load_mpcorb
from jorbit.utils.mpc import (
    packed_to_unpacked_designation,
    unpacked_to_packed_designation,
)


def test_designation_translators() -> None:
    """Test that the designation translators are consistent."""
    mpcorb = load_mpcorb()
    for n in mpcorb["Packed designation"]:
        q = packed_to_unpacked_designation(n)
        m = unpacked_to_packed_designation(q)
        if n != m:
            print(n, q, m)
            raise ValueError

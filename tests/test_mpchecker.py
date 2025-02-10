from jorbit.mpchecker import load_mpcorb
from jorbit.mpchecker.parse_jorbit_ephem import (
    packed_to_unpacked_designation,
    unpacked_to_packed_designation,
)


def test_designation_translators():
    mpcorb = load_mpcorb()
    for n in mpcorb["Packed designation"]:
        q = packed_to_unpacked_designation(n)
        m = unpacked_to_packed_designation(q)
        if n != m:
            print(n, q, m)
            raise ValueError

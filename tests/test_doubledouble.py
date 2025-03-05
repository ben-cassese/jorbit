"""Tests for the DoubleDouble class."""

import jax

jax.config.update("jax_enable_x64", True)
import mpmath as mpm

from jorbit.utils.doubledouble import DoubleDouble


def test_doubledouble() -> None:
    """Simple tests of DoubleDouble arithmetic."""
    a_j = DoubleDouble.from_string("9999999999999.123456")
    b_j = DoubleDouble.from_string("201.789")
    a_m = mpm.mpf("9999999999999.123456")
    b_m = mpm.mpf("201.789")

    # division
    res_j = a_j / b_j
    res_m = a_m / b_m
    res_jm = mpm.mpf(float(res_j.hi)) + mpm.mpf(float(res_j.lo))
    assert (mpm.mpf("1") - res_jm / res_m) < mpm.mpf(1e-30)

    # multiplication
    res_j = a_j * b_j
    res_m = a_m * b_m
    res_jm = mpm.mpf(float(res_j.hi)) + mpm.mpf(float(res_j.lo))
    assert (mpm.mpf("1") - res_jm / res_m) < mpm.mpf(1e-30)

    # addition
    res_j = a_j + b_j
    res_m = a_m + b_m
    res_jm = mpm.mpf(float(res_j.hi)) + mpm.mpf(float(res_j.lo))
    assert (mpm.mpf("1") - res_jm / res_m) < mpm.mpf(1e-30)

    # subtraction
    res_j = a_j - b_j
    res_m = a_m - b_m
    res_jm = mpm.mpf(float(res_j.hi)) + mpm.mpf(float(res_j.lo))
    assert (mpm.mpf("1") - res_jm / res_m) < mpm.mpf(1e-30)

    assert a_j > b_j
    assert a_j >= b_j
    assert b_j < a_j
    assert b_j <= a_j
    assert a_j == a_j

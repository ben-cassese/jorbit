"""A function for parsing an MPC observations file."""

import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.time import Time


def read_mpc_file(
    mpc_file: str,
) -> tuple[SkyCoord, list[Time], list[str], list[u.Quantity]]:
    """Read an MPC observations file and extract the relevant data.

    Haven't checked on this in a while - it may be out of date.

    Args:
        mpc_file (str):
            Path to the MPC observations file.

    Returns:
        tuple[SkyCoord, list[Time], list[str], list[u.Quantity]]:
            A tuple containing the following elements.
            (SkyCoord, The observed coordinates;
            list[Time], The times of observation;
            list[str], The observatory locations;
            list[u.Quantity], The astrometric uncertainties)
    """
    cols = [
        (0, 5),
        (5, 12),
        (12, 13),
        (13, 14),
        (14, 15),
        (15, 32),
        (32, 44),
        (44, 56),
        (65, 70),
        (70, 71),
        (77, 80),
    ]

    names = [
        "Packed number",
        "Packed provisional designation",
        "Discovery asterisk",
        "Note 1",
        "Note 2",
        "Date of observation",
        "Observed RA (J2000.0)",
        "Observed Decl. (J2000.0)",
        "Observed magnitude",
        "Band",
        "Observatory code",
    ]

    data = pd.read_fwf(mpc_file, colspecs=cols, names=names)

    def parse_time(mpc_time: str) -> Time:
        t = mpc_time.replace(" ", "-").split(".")
        return Time(t[0], format="iso", scale="utc") + float(f"0.{t[1]}") * u.day

    def parse_uncertainty(dec_coord: str) -> u.Quantity:
        if len(dec_coord.split(".")) == 1:
            return 1 * u.arcsec
        return 10 ** (-len(dec_coord.split(".")[1])) * u.arcsec

    observed_coordinates = SkyCoord(
        data["Observed RA (J2000.0)"],
        data["Observed Decl. (J2000.0)"],
        unit=(u.hourangle, u.deg),
    )
    times = list(map(parse_time, data["Date of observation"]))
    observatory_locations = [s + "@399" for s in list(data["Observatory code"])]
    astrometric_uncertainties = list(
        map(parse_uncertainty, data["Observed Decl. (J2000.0)"])
    )
    return (
        observed_coordinates,
        times,
        observatory_locations,
        astrometric_uncertainties,
    )


def unpacked_to_packed_designation(number_str: str) -> str:
    """Convert an unpacked designation to a packed designation.

    Useful for translating between the leftmost and rightmost columns of a mpcorb file.
    Correctly handles provisional designations, low-numbered objects, medium-numbered
    objects, and high-numbered objects.

    Args:
        number_str (str):
            The unpacked designation. If is 7 digits and begins with a letter, it's
            assumed to be a provisional designation and is returned unchanged.
            Otherwise it's assumed to be a numbered object and will be packed into a 5
            digit form.

    Returns:
        str:
            The packed designation.
    """
    # If it's a provisional designation (7 characters), return as is
    # adding this isalpha check in case we reach > 10^7 numbered objects soonish
    if (len(number_str) == 7) and number_str[0].isalpha():
        return number_str

    # Convert to integer for numerical comparisons
    num = int(number_str)

    # Low numbers (purely numeric) - return as is
    if num < 100000:
        return number_str

    # Medium numbers (10000-619999) - convert to letter + 4 digits
    if num < 620000:
        # Calculate the letter prefix and remaining digits
        prefix_num = num // 10000
        remaining = num % 10000

        # Convert prefix number to letter (matching the original letter_to_number function)
        if prefix_num >= 36:  # a-z for 36+
            prefix = chr(ord("a") + (prefix_num - 36))
        else:  # A-Z for 10-35
            prefix = chr(ord("A") + (prefix_num - 10))

        # Format the remaining digits with leading zeros
        return f"{prefix}{remaining:04d}"

    # High numbers (620000+) - convert to tilde + base62
    def decimal_to_base62(n: int) -> str:
        """Convert decimal number to base62 string."""
        chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        if n == 0:
            return "0"

        result = ""
        while n > 0:
            n, remainder = divmod(n, 62)
            result = chars[remainder] + result
        return result

    # Subtract the offset and convert to base62
    base62_num = decimal_to_base62(num - 620000)
    # Pad to ensure total length of 5 characters (including the tilde)
    return f"~{base62_num:0>4}"


def packed_to_unpacked_designation(code: str) -> str:
    """Convert a packed designation to an unpacked designation.

    Useful for translating between the leftmost and rightmost columns of a mpcorb file.
    Correctly handles provisional designations, low-numbered objects, medium-numbered
    objects, and high-numbered objects.

    Args:
        code (str):
            The packed designation. 5 characters for numbered objects, 7 for
            provisional.

    Returns:
        str:
            The unpacked designation.
    """
    # if it's a provisional designation, just return it
    if len(code) == 7:
        return code

    # if it's a numbered object, it could be written 3 forms:

    # low numbered objects are just numbers
    if code.isdigit():
        return code

    # medium-numbered objects are a letter followed by 4 digits
    def letter_to_number(char: str) -> int:
        if char.isupper():
            return ord(char) - ord("A") + 10
        else:
            return ord(char) - ord("a") + 36

    if code[0].isalpha() and code[1:].isdigit():
        prefix_value = letter_to_number(code[0])
        num = (prefix_value * 10000) + int(code[1:])
        return str(num)

    # high-numbered objects are a tilde followed by a base-62 number
    def base62_to_decimal(char: str) -> int:
        if char.isdigit():
            return int(char)
        elif char.isupper():
            return ord(char) - ord("A") + 10
        else:
            return ord(char) - ord("a") + 36

    if code.startswith("~"):
        # Convert each character to its decimal value and calculate total
        total = 0
        for position, char in enumerate(reversed(code[1:])):
            decimal_value = base62_to_decimal(char)
            total += decimal_value * (62**position)
        num = total + 620000
        return str(num)

    raise ValueError(f"Invalid MPC code format: {code}")

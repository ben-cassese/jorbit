import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.time import Time


def read_mpc_file(mpc_file):
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

    def parse_time(mpc_time):
        t = mpc_time.replace(" ", "-").split(".")
        return Time(t[0], format="iso", scale="utc") + float(f"0.{t[1]}") * u.day

    def parse_uncertainty(dec_coord):
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

import pandas as pd
import polars as pl
from astropy.utils.data import download_file

from jorbit.data.constants import JORBIT_EPHEM_URL_BASE


def load_mpcorb(filename=None):
    # convert the MPCORB.DAT file into a format that can be read quickly by Polars
    column_spans = {
        "Packed designation": (0, 7),
        "H": (8, 13),
        "G": (14, 19),
        "Epoch": (20, 25),
        "M": (26, 36),
        "Peri": (37, 47),
        "Node": (48, 58),
        "Incl.": (59, 69),
        "e": (70, 79),
        "n": (80, 91),
        "a": (92, 104),
        "U": (105, 106),
        "Reference": (107, 116),
        "#Obs": (117, 122),
        "#Opp": (123, 126),
        "Arc": (127, 136),
        "rms": (137, 141),
        "Coarse Perts": (142, 145),
        "Precise Perts": (146, 149),
        "Computer": (150, 160),
        "Flags": (161, 165),
        "Unpacked Name": (166, 193),
        "last obs": (194, 201),
    }

    col_names = list(column_spans.keys())
    col_widths = [end - start + 1 for start, end in column_spans.values()]

    if filename is None:
        file_path = download_file(JORBIT_EPHEM_URL_BASE + "MPCORB.DAT", cache=True)
    else:
        file_path = filename

    df = pd.read_fwf(
        file_path, widths=col_widths, names=col_names, dtype=str, skiprows=43
    )
    df = pl.DataFrame(df)
    df.write_ipc("mpcorb.arrow")
    return df

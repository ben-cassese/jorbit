import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from astropy.time import Time
from astropy.utils.data import download_file
from jplephem.spk import SPK


def extract_data(
    center: str, target: str, ephem_file: str, earliest_time: Time, latest_time: Time
) -> tuple:
    """
    Extracts the Chebyshev coefficients for a given target and center from an SPK file.

    Args:
        center (str):
            The center ID of an ephemeris segment.
        target (str):
            The target ID of an ephemeris segment.
        ephem_file (str):
            The path to the SPK file. Uses astropy's download_file function to download
            and cache the file if it isn't present in the cache.
        earliest_time (Time):
            The start time for our region of interest. Smaller time windows will result
            in smaller in-memory ephemeris objects.
        latest_time (Time):
            The latest time for the ephemeris.

    Returns:
        tuple:
            A tuple containing the Chebyshev coefficients for the given target and
            center. More specifically, the tuple contains
            (init, intlen, coeff), where init is the initial time of the segment,
            intlen is the interval length, and coeff is the Chebyshev coefficients.
    """
    spk = SPK.open(download_file(ephem_file, cache=True))

    target_found = False
    for seg in spk.segments:
        if seg.center == center and seg.target == target:
            target_found = True
            if seg.start_jd <= earliest_time.jd and seg.end_jd >= latest_time.jd:
                init, intlen, coeff = seg._data
                return init, intlen, coeff
    if not target_found:
        raise ValueError(
            f"Target '{target}' and center '{center}' not found in SPK file '{ephem_file}'"
        )
    else:
        raise ValueError(
            f"Time range not valid for this target and center in SPK file '{ephem_file}'"
        )


def merge_data(
    inits: list, intlens: list, coeffs: list, earliest_time: Time, latest_time: Time
) -> tuple:
    """
    Merges data for multiple targets into a single set of coefficients.

    This takes all of the data extracted from individual target/center segments and
    merges them into larger jnp.ndarrays. All objects will now use the same number of
    coefficients to describe each interval, so objects that were previously missing
    high-order coefficients will have those zero padded. Also, all objects will now
    have the same number of intervals (but those intervals will have distinct inits),
    so objects that were previously missing intervals will have those also zero padded.


    Args:
        inits (list):
            A list of initial times for each target.
        intlens (list):
            A list of interval lengths for each target.
        coeffs (list):
            A list of Chebyshev coefficients for each target.
        earliest_time (Time):
            The earliest time to consider.
        latest_time (Time):
            The latest time to consider.

    Returns:
        tuple:
            A tuple containing the merged initial time, interval length, and coefficients.

    """
    inits = jnp.array(inits)
    intlens = jnp.array(intlens)

    init0 = inits[0]
    for i in inits:
        assert i == init0

    # Trim the timespans down to the earliest and latest times
    longest_intlen = jnp.max(intlens)
    ratios = longest_intlen / intlens
    early_indecies = []
    late_indecies = []
    for i in range(len(inits)):
        component_count, coefficient_count, n = coeffs[i].shape
        index, offset = jnp.divmod(
            (earliest_time.tdb.jd - 2451545.0) * 86400.0 - inits[i],
            intlens[i],
        )
        omegas = index == n
        index = jnp.where(omegas, index - 1, index)
        early_indecies.append(index)

        index, offset = jnp.divmod(
            (latest_time.tdb.jd - 2451545.0) * 86400.0 - inits[i],
            intlens[i],
        )
        omegas = index == n
        index = jnp.where(omegas, index - 1, index)
        late_indecies.append(index)

    early_indecies = (
        jnp.ones(len(early_indecies)) * jnp.min(jnp.array(early_indecies)) * ratios
    ).astype(int)
    new_inits = inits + early_indecies * intlens
    late_indecies = (
        jnp.ones(len(late_indecies)) * jnp.min(jnp.array(late_indecies)) * ratios
    ).astype(int)
    trimmed_coeffs = []
    for i in range(len(inits)):
        trimmed_coeffs.append(coeffs[i][:, :, early_indecies[i] : late_indecies[i]])

    # Add extra Chebyshev coefficients (zeros) to make the number of
    # coefficients at each time slice the same across all planets
    coeff_shapes = []
    for i in trimmed_coeffs:
        coeff_shapes.append(i.shape)
    coeff_shapes = jnp.array(coeff_shapes)
    most_coefficients, _, most_time_slices = jnp.max(coeff_shapes, axis=0)

    padded_coefficients = []
    for c in trimmed_coeffs:
        c = jnp.pad(c, ((most_coefficients - c.shape[0], 0), (0, 0), (0, 0)))
        padded_coefficients.append(c)

    # This is a little sketchy- tile each planet so that they all have the same
    # number of time slices. This means that for planets with longer original intlens,
    # we could technically feed in times outside the original timespan and get a false result
    # But, by keeping their original intlens intact, if we feed in a time within
    # the timespan, we should just always stay in the first half, quarter, whatever
    shortest_intlen = jnp.min(intlens)
    extra_padded = []
    for i in range(len(padded_coefficients)):
        extra_padded.append(
            jnp.tile(padded_coefficients[i], int(intlens[i] / shortest_intlen))
        )

    new_coeffs = jnp.array(extra_padded)

    return new_inits, intlens, new_coeffs

"""Constants used throughout the package."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

################################################################################
# Misc constants
################################################################################

SPEED_OF_LIGHT = 173.14463267424034
"""c in AU/day."""

INV_SPEED_OF_LIGHT = 1 / 173.14463267424034  # 1 / AU/day
"""1 / c in day/AU."""

# no longer used- horizons "ecliptic" J2000 is just a 84381.448" rotation about
# the ICRS x-axis
# ICRS_TO_BARY_ROT_MAT = jnp.array(erfa.ecm06(2451545.0, 0.0))  # J2000
# BARY_TO_ICRS_ROT_MAT = jnp.array(ICRS_TO_BARY_ROT_MAT.T)
# from mpmath import mp
# mp.dps = 24
# 84381.448 / (1/(2 * mp.pi / 360 / 60 / 60)) =
# 0.40909280422232897, then mp.cos and mp.sin of that
ICRS_TO_HORIZONS_ECLIPTIC_ROT_MAT = jnp.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.917482062069181818110850924, 0.397777155931913719203212896],
        [0.0, -0.397777155931913719203212896, 0.917482062069181818110850924],
    ]
)
"""
Rotation matrix to convert from ICRS to Horizons Ecliptic J2000.

The former is defined as a 84381.448" rotation about the ICRS x-axis. Values computed
using mpmath with 24 digits of precision.
"""

HORIZONS_ECLIPTIC_TO_ICRS_ROT_MAT = ICRS_TO_HORIZONS_ECLIPTIC_ROT_MAT.T
"""Transpose of ICRS_TO_HORIZONS_ECLIPTIC_ROT_MAT."""

EPSILON = jnp.array(jnp.finfo(jnp.float64).eps)
"""Machine specific precision."""

JORBIT_EPHEM_URL_BASE = (
    "https://huggingface.co/datasets/jorbit/jorbit_mpchecker/resolve/main/"
)
"""The URL root where are the jorbit ephemeris files are stored."""
JORBIT_EPHEM_CACHE_TIMEOUT = 6 * 31 * 24 * 60 * 60  # 6 months
"""The timeout for the jorbit ephemeris cache."""

################################################################################
# Ephemeris constants
################################################################################

DEFAULT_PLANET_EPHEMERIS_URL = "https://ssd.jpl.nasa.gov//ftp/eph/planets/bsp/de440.bsp"
"""The default URL for the planet ephemeris."""

DEFAULT_ASTEROID_EPHEMERIS_URL = (
    "https://ssd.jpl.nasa.gov//ftp/eph/small_bodies/asteroids_de441/sb441-n16.bsp"
)
"""The default URL for the asteroid ephemeris."""

HUGE_ASTEROID_EPHEMERIS_URL = (
    "https://ssd.jpl.nasa.gov//ftp/eph/small_bodies/asteroids_de441/sb441-n373.bsp"
)
"""The URL for the asteroid ephemeris with 373 asteroids. Currently never used."""
# also here's 441
# "https://ssd.jpl.nasa.gov//ftp/eph/planets/bsp/de441.bsp"

DE430_PLANET_EPHEMERIS_URL = (
    "https://ssd.jpl.nasa.gov//ftp/eph/planets/bsp/de430_1850-2150.bsp"
)
"""The URL for the DE430 planet ephemeris."""

DE430_ASTEROID_EPHEMERIS_URL = (
    "https://ssd.jpl.nasa.gov//ftp/eph/small_bodies/asteroids_de430/ast343de430.bsp"
)
"""The URL for the DE430 asteroid ephemeris."""


# These are NOT from the JPL ephemeris comments, and are in units of AU^3 / day^2
# They are actually from https://ssd.jpl.nasa.gov/ftp/xfr/gm_Horizons.pck,
# which lists Earth as apparently 1% different from its de440 and de441 value
ALL_PLANET_LOG_GMS = {
    "mercury": jnp.log(4.9125001948893175e-11),
    "venus": jnp.log(7.2434523326441177e-10),
    "earth_bary": jnp.log(0.0),
    "earth": jnp.log(8.887692446707103e-10),
    "moon": jnp.log(1.0931894624024349e-11),
    "mars": jnp.log(9.5495488297258106e-11),
    "jupiter": jnp.log(2.8253458252257912e-07),
    "saturn": jnp.log(8.4597059933762889e-08),
    "uranus": jnp.log(1.2920265649682398e-08),
    "neptune": jnp.log(1.5243573478851935e-08),
    "pluto": jnp.log(2.1750964648933581e-12),
    "sun": jnp.log(2.9591220828411951e-04),  # G in AU^3 / day^2
}
"""
The log GM values of the planets and sun, units of AU^3 / day^2.

These are NOT from the JPL ephemeris comments. They are actually from
https://ssd.jpl.nasa.gov/ftp/xfr/gm_Horizons.pck, which lists Earth as apparently 1%
different from its de440 and de441 value. Chasing that down was a top-5 debugging day
for sure.
"""

LARGE_ASTEROID_LOG_GMS = {
    "ceres": jnp.log(1.3964518123081067e-13),
    "pallas": jnp.log(3.0471146330043194e-14),
    "juno": jnp.log(4.2823439677995e-15),
    "vesta": jnp.log(3.85480002252579e-14),
    "iris": jnp.log(2.5416014973471494e-15),
    "hygiea": jnp.log(1.2542530761640807e-14),
    "eunomia": jnp.log(4.5107799051436795e-15),
    "psyche": jnp.log(3.544500284248897e-15),
    "euphrosyne": jnp.log(2.4067012218937573e-15),
    "europa": jnp.log(5.982431526486983e-15),
    "cybele": jnp.log(2.091717595513368e-15),
    "sylvia": jnp.log(4.834560654610551e-15),
    "thisbe": jnp.log(2.652943661035635e-15),
    "camilla": jnp.log(3.2191392075878576e-15),
    "davida": jnp.log(8.683625349228651e-15),
    "interamnia": jnp.log(6.311034342087888e-15),
}
"""Similar to ALL_PLANET_LOG_GMS but for the 16 asteroid perturbers."""

# just the sum of all the planets and large asteroids above
TOTAL_SOLAR_SYSTEM_GM = 0.000296309274879932  # 0.00029630927460766373 if just planets
"""Just the sum of ALL_PLANET_LOG_GMS and LARGE_ASTEROID_LOG_GMS."""


EARTH_J_HARMONICS = jnp.array(
    [1.0826253900000000e-03, -2.5324100000000000e-06, -1.6198980000000001e-06]
)
"""The J2, J3, and J4 harmonics of the Earth. Taken from the JPL DE440/441 ephemeris."""

SUN_J_HARMONICS = jnp.array([2.1961391516529825e-07])
"""The J2 harmonic of the Sun. Taken from the JPL DE440/441 ephemeris."""

# EARTH_RADIUS = 6.3781365999999998e+03 / 1.4959787069999999e+08
# """The radius of the Earth in AU. Taken from the JPL DE440/441 ephemeris."""

EARTH_RADIUS = 6378.137 / 149597870.700
"""The radius of the Earth in AU. Taken from Horizons web interface."""

# SUN_RADIUS = 6.9600000000000000e+05 / 1.4959787069999999e+08
# """The radius of the Sun in AU. Taken from the JPL DE440/441 ephemeris."""

SUN_RADIUS = 695700.0 / 149597870.700
"""The radius of the Sun in AU. Taken from Horizons web interface."""

EARTH_POLE_RA = 0.0
"""The right ascension of the Earth's pole in radians at the J2000 epoch."""

EARTH_POLE_DEC = 90.0 * jnp.pi / 180
"""The declination of the Earth's pole in radians at the J2000 epoch."""

SUN_POLE_RA = 286.13 * jnp.pi / 180
"""The right ascension of the Sun's pole in radians at the J2000 epoch. Chosen to match
the hard-coded value in ASSIST."""

SUN_POLE_DEC = 63.87 * jnp.pi / 180
"""The declination of the Sun's pole in radians at the J2000 epoch. Chosen to match
the hard-coded value in ASSIST."""


ALL_PLANET_IDS = {
    "mercury": 1,
    "venus": 2,
    "earth_bary": 3,
    "earth": 399,
    "moon": 301,
    "mars": 4,
    "jupiter": 5,
    "saturn": 6,
    "uranus": 7,
    "neptune": 8,
    "pluto": 9,
    "sun": 10,
}
"""The IDs of the planets and sun."""

LARGE_ASTEROID_IDS = {
    "ceres": 2000001,
    "pallas": 2000002,
    "juno": 2000003,
    "vesta": 2000004,
    "iris": 2000007,
    "hygiea": 2000010,
    "eunomia": 2000015,
    "psyche": 2000016,
    "euphrosyne": 2000031,
    "europa": 2000052,
    "cybele": 2000065,
    "sylvia": 2000087,
    "thisbe": 2000088,
    "camilla": 2000107,
    "davida": 2000511,
    "interamnia": 2000704,
}
"""The SPK ID numbers for the 16 asteroid perturbers."""

ALL_PLANET_NAMES = [
    "sun",
    "mercury",
    "venus",
    "earth_bary",
    "earth",
    "moon",
    "mars",
    "jupiter",
    "saturn",
    "uranus",
    "neptune",
    "pluto",
]
"""The ordered names of the planets and sun."""

LARGE_ASTEROID_NAMES = [
    "ceres",
    "pallas",
    "juno",
    "vesta",
    "iris",
    "hygiea",
    "eunomia",
    "psyche",
    "euphrosyne",
    "europa",
    "cybele",
    "sylvia",
    "thisbe",
    "camilla",
    "davida",
    "interamnia",
]
"""The ordered names of the 16 asteroid perturbers."""

PERTURBER_PACKED_DESIGNATIONS = [
    "00001",
    "00002",
    "00003",
    "00004",
    "00007",
    "00010",
    "00015",
    "00016",
    "00031",
    "00052",
    "00065",
    "00087",
    "00088",
    "00107",
    "00511",
    "00704",
    "D4340",
]
"""The packed designations of the 16 asteroid perturbers + Pluto."""

################################################################################
# Yoshida constants.

# No longer ever used now that we've removed the high order leapfrog integrator.
################################################################################

# # Taken from Section 4 of Yoshida 1990
# # DOI: 10.1016/0375-9601(90)90092-3
# Y4_Ws = jnp.array([1 / (2 - 2 ** (1 / 3))])

# # Taken from Table 1 of Yoshida 1990
# # DOI: 10.1016/0375-9601(90)90092-3
# Y6_Ws = jnp.array([-0.117767998417887e1, 0.23557321335935, 0.78451361047756])

# # Taken from Table 2 of Yoshida 1990
# # DOI: 10.1016/0375-9601(90)90092-3
# Y8_Ws = jnp.array(
#     [
#         0.102799849391985e0,
#         -0.196061023297549e1,
#         0.193813913762276e1,
#         -0.158240635368243e0,
#         -0.144485223686048e1,
#         0.253693336566229e0,
#         0.914844246229740e0,
#     ]
# )

# # Created using the Decimal version of
# # jorbit.utils.generate_coefficients.create_yoshida_coeffs
# Y4_C = jnp.array(
#     [
#         0.675603595979828817023843904,
#         -0.17560359597982881702384390,
#         -0.17560359597982881702384390,
#         0.675603595979828817023843904,
#     ]
# )

# # Created using the Decimal version of
# # jorbit.utils.generate_coefficients.create_yoshida_coeffs
# Y4_D = jnp.array(
#     [
#         1.351207191959657634047687808,
#         -1.70241438391931526809537562,
#         1.351207191959657634047687808,
#     ]
# )

# # Created using the Decimal version of
# # jorbit.utils.generate_coefficients.create_yoshida_coeffs
# Y6_C = jnp.array(
#     [
#         0.392256805238779981959140741,
#         0.510043411918454980824577660,
#         -0.47105338540976005035076923,
#         0.068753168252525087567050832,
#         0.068753168252525087567050832,
#         -0.47105338540976005035076923,
#         0.510043411918454980824577660,
#         0.392256805238779981959140741,
#     ]
# )

# # Created using the Decimal version of
# # jorbit.utils.generate_coefficients.create_yoshida_coeffs
# Y6_D = jnp.array(
#     [
#         0.78451361047755996391828148,
#         0.23557321335934999773087383,
#         -1.1776799841788700984324123,
#         1.31518632068392027356651397,
#         -1.1776799841788700984324123,
#         0.23557321335934999773087383,
#         0.78451361047755996391828148,
#     ]
# )

# # Created using the Decimal version of
# # jorbit.utils.generate_coefficients.create_yoshida_coeffs
# Y8_C = jnp.array(
#     [
#         0.457422123114870016191702006,
#         0.584268791397984516011732125,
#         -0.59557945014712546094592937,
#         -0.80154643611436146577453598,
#         0.889949251127258450511092746,
#         -0.01123554767636503193273256,
#         -0.92890519179175248809521292,
#         0.905626460089491464033883972,
#         0.905626460089491464033883972,
#         -0.92890519179175248809521292,
#         -0.01123554767636503193273256,
#         0.889949251127258450511092746,
#         -0.80154643611436146577453598,
#         -0.59557945014712546094592937,
#         0.584268791397984516011732125,
#         0.457422123114870016191702006,
#     ]
# )

# # Created using the Decimal version of
# # jorbit.utils.generate_coefficients.create_yoshida_coeffs
# Y8_D = jnp.array(
#     [
#         0.91484424622974003238340401,
#         0.25369333656622899964006023,
#         -1.4448522368604799215319189,
#         -0.1582406353682430100171529,
#         1.93813913762275991103933847,
#         -1.9606102329754899749048036,
#         0.10279984939198499871437775,
#         1.70845307078699792935339019,
#         0.10279984939198499871437775,
#         -1.9606102329754899749048036,
#         1.93813913762275991103933847,
#         -0.1582406353682430100171529,
#         -1.4448522368604799215319189,
#         0.25369333656622899964006023,
#         0.91484424622974003238340401,
#     ]
# )

################################################################################
# IAS15 constants
################################################################################
# https://github.com/hannorein/rebound/blob/0b5c85d836fec20bc284d1f1bb326f418e11f591/src/integrator_ias15.c
IAS15_H = jnp.array(
    [
        0.0,
        0.0562625605369221464656521910318,
        0.180240691736892364987579942780,
        0.352624717113169637373907769648,
        0.547153626330555383001448554766,
        0.734210177215410531523210605558,
        0.885320946839095768090359771030,
        0.977520613561287501891174488626,
    ]
)
"""The H array from `REBOUND <https://github.com/hannorein/rebound/blob/0b5c85d836fec20bc284d1f1bb326f418e11f591/src/integrator_ias15.c>`_."""

# https://github.com/hannorein/rebound/blob/0b5c85d836fec20bc284d1f1bb326f418e11f591/src/integrator_ias15.c
IAS15_RR = jnp.array(
    [
        0.0562625605369221464656522,
        0.1802406917368923649875799,
        0.1239781311999702185219278,
        0.3526247171131696373739078,
        0.2963621565762474909082556,
        0.1723840253762772723863278,
        0.5471536263305553830014486,
        0.4908910657936332365357964,
        0.3669129345936630180138686,
        0.1945289092173857456275408,
        0.7342101772154105315232106,
        0.6779476166784883850575584,
        0.5539694854785181665356307,
        0.3815854601022408941493028,
        0.1870565508848551485217621,
        0.8853209468390957680903598,
        0.8290583863021736216247076,
        0.7050802551022034031027798,
        0.5326962297259261307164520,
        0.3381673205085403850889112,
        0.1511107696236852365671492,
        0.9775206135612875018911745,
        0.9212580530243653554255223,
        0.7972799218243951369035945,
        0.6248958964481178645172667,
        0.4303669872307321188897259,
        0.2433104363458769703679639,
        0.0921996667221917338008147,
    ]
)
"""The RR array from `REBOUND <https://github.com/hannorein/rebound/blob/0b5c85d836fec20bc284d1f1bb326f418e11f591/src/integrator_ias15.c>`_."""

IAS15_r1 = jnp.array([IAS15_RR[0]])
IAS15_r2 = IAS15_RR[1 : 1 + 2]
IAS15_r3 = IAS15_RR[3 : 3 + 3]
IAS15_r4 = IAS15_RR[6 : 6 + 4]
IAS15_r5 = IAS15_RR[10 : 10 + 5]
IAS15_r6 = IAS15_RR[15 : 15 + 6]
IAS15_r7 = IAS15_RR[21 : 21 + 7]

# https://github.com/hannorein/rebound/blob/0b5c85d836fec20bc284d1f1bb326f418e11f591/src/integrator_ias15.c
IAS15_C = jnp.array(
    [
        -0.0562625605369221464656522,
        0.0101408028300636299864818,
        -0.2365032522738145114532321,
        -0.0035758977292516175949345,
        0.0935376952594620658957485,
        -0.5891279693869841488271399,
        0.0019565654099472210769006,
        -0.0547553868890686864408084,
        0.4158812000823068616886219,
        -1.1362815957175395318285885,
        -0.0014365302363708915424460,
        0.0421585277212687077072973,
        -0.3600995965020568122897665,
        1.2501507118406910258505441,
        -1.8704917729329500633517991,
        0.0012717903090268677492943,
        -0.0387603579159067703699046,
        0.3609622434528459832253398,
        -1.4668842084004269643701553,
        2.9061362593084293014237913,
        -2.7558127197720458314421588,
    ]
)
"""The C array from `REBOUND <https://github.com/hannorein/rebound/blob/0b5c85d836fec20bc284d1f1bb326f418e11f591/src/integrator_ias15.c>`_."""

# https://github.com/hannorein/rebound/blob/0b5c85d836fec20bc284d1f1bb326f418e11f591/src/integrator_ias15.c
IAS15_D = jnp.array(
    [
        0.0562625605369221464656522,
        0.0031654757181708292499905,
        0.2365032522738145114532321,
        0.0001780977692217433881125,
        0.0457929855060279188954539,
        0.5891279693869841488271399,
        0.0000100202365223291272096,
        0.0084318571535257015445000,
        0.2535340690545692665214616,
        1.1362815957175395318285885,
        0.0000005637641639318207610,
        0.0015297840025004658189490,
        0.0978342365324440053653648,
        0.8752546646840910912297246,
        1.8704917729329500633517991,
        0.0000000317188154017613665,
        0.0002762930909826476593130,
        0.0360285539837364596003871,
        0.5767330002770787313544596,
        2.2485887607691597933926895,
        2.7558127197720458314421588,
    ]
)
"""The D array from `REBOUND <https://github.com/hannorein/rebound/blob/0b5c85d836fec20bc284d1f1bb326f418e11f591/src/integrator_ias15.c>`_."""

IAS15_D_MATRIX = jnp.array(
    [
        [1.0, IAS15_D[0], IAS15_D[1], IAS15_D[3], IAS15_D[6], IAS15_D[10], IAS15_D[15]],
        [0.0, 1.0, IAS15_D[2], IAS15_D[4], IAS15_D[7], IAS15_D[11], IAS15_D[16]],
        [0.0, 0.0, 1.0, IAS15_D[5], IAS15_D[8], IAS15_D[12], IAS15_D[17]],
        [0.0, 0.0, 0.0, 1.0, IAS15_D[9], IAS15_D[13], IAS15_D[18]],
        [0.0, 0.0, 0.0, 0.0, 1.0, IAS15_D[14], IAS15_D[19]],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, IAS15_D[20]],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
)
"""IAS15D in matrix form for batched operations."""

IAS15_BX_DENOMS = ((jnp.arange(7) + 2) * (jnp.arange(7) + 3))[::-1][:, None, None]
"""The denominators for the coefficients in the polynomial to estimate x(t)."""

IAS15_BV_DENOMS = (jnp.arange(7) + 2)[::-1][:, None, None]
"""The denominators for the coefficients in the polynomial to estimate v(t)."""

# https://github.com/hannorein/rebound/blob/0b5c85d836fec20bc284d1f1bb326f418e11f591/src/integrator_ias15.c
IAS15_EPSILON = 10 ** (-9)
"""Constant from `REBOUND <https://github.com/hannorein/rebound/blob/0b5c85d836fec20bc284d1f1bb326f418e11f591/src/integrator_ias15.c>`_."""

IAS15_EPS_Modified = 0.1750670293218999749  # 0.1750670293218999748586614182797188957 = sqrt7(r->ri_ias15.epsilon*5040.0)
"""Precomputed implementation of sqrt7(r->ri_ias15.epsilon*5040.0)"""

IAS15_SAFETY_FACTOR = 0.25
"""Constant from `REBOUND <https://github.com/hannorein/rebound/blob/0b5c85d836fec20bc284d1f1bb326f418e11f591/src/integrator_ias15.c>`_."""

IAS15_MIN_DT = 0.0
"""Constant from `REBOUND <https://github.com/hannorein/rebound/blob/0b5c85d836fec20bc284d1f1bb326f418e11f591/src/integrator_ias15.c>`_."""

################################################################################
# experimental DoubleDouble IAS15 constants
################################################################################
from jorbit.utils.doubledouble import DoubleDouble

IASNN_DD_EPSILON = DoubleDouble(1e-32)
"""DoubleDouble type epsilon."""

################################################################################
# leapfrog integrator constants
################################################################################
# Taken from Section 4 of Yoshida 1990
# DOI: 10.1016/0375-9601(90)90092-3
Y4_Ws = jnp.array([1 / (2 - 2 ** (1 / 3))])
"""The weight vector for the 4th order leapfrog integrator, Yoshida (1990) Sec 4."""

# Taken from Table 1 of Yoshida 1990
# DOI: 10.1016/0375-9601(90)90092-3
Y6_Ws = jnp.array([-0.117767998417887e1, 0.23557321335935, 0.78451361047756])
"""The weight vector for the 6th order leapfrog integrator, Yoshida (1990) Table 1."""

# Taken from Table 2 of Yoshida 1990
# DOI: 10.1016/0375-9601(90)90092-3
Y8_Ws = jnp.array(
    [
        0.102799849391985e0,
        -0.196061023297549e1,
        0.193813913762276e1,
        -0.158240635368243e0,
        -0.144485223686048e1,
        0.253693336566229e0,
        0.914844246229740e0,
    ]
)
"""The weight vector for the 8th order leapfrog integrator, Yoshida (1990) Table 2."""

Y4_C = jnp.array(
    [
        0.675603595979828817023843904,
        -0.1756035959798288170238439045,
        -0.1756035959798288170238439045,
        0.675603595979828817023843904,
    ]
)
"""C vector for 4th order leapfrog integrator, created using jorbit.integrators.yoshida_integrator._create_yoshida_coeffs."""

Y4_D = jnp.array(
    [
        1.351207191959657634047687808,
        -1.702414383919315268095375617,
        1.351207191959657634047687808,
    ]
)
"""D vector for 4th order leapfrog integrator, created using jorbit.integrators.yoshida_integrator._create_yoshida_coeffs."""

Y6_C = jnp.array(
    [
        0.3922568052387799819591407413,
        0.5100434119184549808245776605,
        -0.4710533854097600503507692338,
        0.06875316825252508756705083270,
        0.06875316825252508756705083270,
        -0.4710533854097600503507692338,
        0.5100434119184549808245776605,
        0.3922568052387799819591407413,
    ]
)
"""C vector for 6th order leapfrog integrator, created using jorbit.integrators.yoshida_integrator._create_yoshida_coeffs."""

Y6_D = jnp.array(
    [
        0.78451361047755996391828148261993192136287689208984375,
        0.2355732133593499977308738380088470876216888427734375,
        -1.1776799841788700984324123055557720363140106201171875,
        1.315186320683920273566513971,
        -1.1776799841788700984324123055557720363140106201171875,
        0.2355732133593499977308738380088470876216888427734375,
        0.78451361047755996391828148261993192136287689208984375,
    ]
)
"""D vector for 6th order leapfrog integrator, created using jorbit.integrators.yoshida_integrator._create_yoshida_coeffs."""

Y8_C = jnp.array(
    [
        0.4574221231148700161917020068,
        0.5842687913979845160117321255,
        -0.5955794501471254609459293760,
        -0.8015464361143614657745359865,
        0.8899492511272584505110927465,
        -0.01123554767636503193273256329,
        -0.9289051917917524880952129250,
        0.9056264600894914640338839720,
        0.9056264600894914640338839720,
        -0.9289051917917524880952129250,
        -0.01123554767636503193273256329,
        0.8899492511272584505110927465,
        -0.8015464361143614657745359865,
        -0.5955794501471254609459293760,
        0.5842687913979845160117321255,
        0.4574221231148700161917020068,
    ]
)
"""C vector for 8th order leapfrog integrator, created using jorbit.integrators.yoshida_integrator._create_yoshida_coeffs."""

Y8_D = jnp.array(
    [
        0.91484424622974003238340401367167942225933074951171875,
        0.253693336566228999640060237652505747973918914794921875,
        -1.4448522368604799215319189897854812443256378173828125,
        -0.158240635368243010017152982982224784791469573974609375,
        1.93813913762275991103933847625739872455596923828125,
        -1.960610232975489974904803602839820086956024169921875,
        0.102799849391984998714377752548898570239543914794921875,
        1.708453070786997929353390191,
        0.102799849391984998714377752548898570239543914794921875,
        -1.960610232975489974904803602839820086956024169921875,
        1.93813913762275991103933847625739872455596923828125,
        -0.158240635368243010017152982982224784791469573974609375,
        -1.4448522368604799215319189897854812443256378173828125,
        0.253693336566228999640060237652505747973918914794921875,
        0.91484424622974003238340401367167942225933074951171875,
    ]
)
"""D vector for 8th order leapfrog integrator, created using jorbit.integrators.yoshida_integrator._create_yoshida_coeffs."""

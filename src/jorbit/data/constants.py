import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

################################################################################
# Misc constants
################################################################################

JORBIT_EPOCH = 1.0

SPEED_OF_LIGHT = 173.14463267424034
INV_SPEED_OF_LIGHT = 1 / 173.14463267424034  # 1 / AU/day

################################################################################
# Ephemeris constants
################################################################################

# These are from the JPL ephemeris comments, and are in units of AU^3 / day^2
# Pluto is still included in the main "de" ephemeris series, no other dwarf planets
# JOKES these are actually from https://ssd.jpl.nasa.gov/ftp/xfr/gm_Horizons.pck,
# which lists Earth as apparently 1% different from its de440 and de441 value??
ALL_PLANET_GMS = {
    "mercury": 4.9125001948893175e-11,
    "venus": 7.2434523326441177e-10,
    "earth": 8.9970113929473456e-10,
    "mars": 9.5495488297258106e-11,
    "jupiter": 2.8253458252257912e-07,
    "saturn": 8.4597059933762889e-08,
    "uranus": 1.2920265649682398e-08,
    "neptune": 1.5243573478851935e-08,
    "pluto": 2.1750964648933581e-12,
    "sun": 2.9591220828411951e-04,
}

LARGE_ASTEROID_GMS = {
    "ceres": 1.3964518123081067e-13,
    "pallas": 3.0471146330043194e-14,
    "juno": 4.2823439677995e-15,
    "vesta": 3.85480002252579e-14,
    "iris": 2.5416014973471494e-15,
    "hygiea": 1.2542530761640807e-14,
    "eunomia": 4.5107799051436795e-15,
    "psyche": 3.544500284248897e-15,
    "euphrosyne": 2.4067012218937573e-15,
    "europa": 5.982431526486983e-15,
    "cybele": 2.091717595513368e-15,
    "sylvia": 4.834560654610551e-15,
    "thisbe": 2.652943661035635e-15,
    "camilla": 3.2191392075878576e-15,
    "davida": 8.683625349228651e-15,
    "interamnia": 6.311034342087888e-15,
}

ALL_PLANET_NUMS = {
    "mercury": 1,
    "venus": 2,
    "earth": 3,
    "mars": 4,
    "jupiter": 5,
    "saturn": 6,
    "uranus": 7,
    "neptune": 8,
    "pluto": 9,
    "sun": 10,
}

LARGE_ASTEROID_NUMS = {
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

################################################################################
# Yoshida constants
################################################################################

# Taken from Section 4 of Yoshida 1990
# DOI: 10.1016/0375-9601(90)90092-3
Y4_Ws = jnp.array([1 / (2 - 2 ** (1 / 3))])

# Taken from Table 1 of Yoshida 1990
# DOI: 10.1016/0375-9601(90)90092-3
Y6_Ws = jnp.array([-0.117767998417887e1, 0.23557321335935, 0.78451361047756])

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

# Created using the Decimal version of
# jorbit.engine.yoshida_integrator._create_yoshida_coeffs
Y4_C = jnp.array(
    [
        0.675603595979828817023843904,
        -0.17560359597982881702384390,
        -0.17560359597982881702384390,
        0.675603595979828817023843904,
    ]
)

# Created using the Decimal version of
# jorbit.engine.yoshida_integrator._create_yoshida_coeffs
Y4_D = jnp.array(
    [
        1.351207191959657634047687808,
        -1.70241438391931526809537562,
        1.351207191959657634047687808,
    ]
)

# Created using the Decimal version of
# jorbit.engine.yoshida_integrator._create_yoshida_coeffs
Y6_C = jnp.array(
    [
        0.392256805238779981959140741,
        0.510043411918454980824577660,
        -0.47105338540976005035076923,
        0.068753168252525087567050832,
        0.068753168252525087567050832,
        -0.47105338540976005035076923,
        0.510043411918454980824577660,
        0.392256805238779981959140741,
    ]
)

# Created using the Decimal version of
# jorbit.engine.yoshida_integrator._create_yoshida_coeffs
Y6_D = jnp.array(
    [
        0.78451361047755996391828148,
        0.23557321335934999773087383,
        -1.1776799841788700984324123,
        1.31518632068392027356651397,
        -1.1776799841788700984324123,
        0.23557321335934999773087383,
        0.78451361047755996391828148,
    ]
)

# Created using the Decimal version of
# jorbit.engine.yoshida_integrator._create_yoshida_coeffs
Y8_C = jnp.array(
    [
        0.457422123114870016191702006,
        0.584268791397984516011732125,
        -0.59557945014712546094592937,
        -0.80154643611436146577453598,
        0.889949251127258450511092746,
        -0.01123554767636503193273256,
        -0.92890519179175248809521292,
        0.905626460089491464033883972,
        0.905626460089491464033883972,
        -0.92890519179175248809521292,
        -0.01123554767636503193273256,
        0.889949251127258450511092746,
        -0.80154643611436146577453598,
        -0.59557945014712546094592937,
        0.584268791397984516011732125,
        0.457422123114870016191702006,
    ]
)

# Created using the Decimal version of
# jorbit.engine.yoshida_integrator._create_yoshida_coeffs
Y8_D = jnp.array(
    [
        0.91484424622974003238340401,
        0.25369333656622899964006023,
        -1.4448522368604799215319189,
        -0.1582406353682430100171529,
        1.93813913762275991103933847,
        -1.9606102329754899749048036,
        0.10279984939198499871437775,
        1.70845307078699792935339019,
        0.10279984939198499871437775,
        -1.9606102329754899749048036,
        1.93813913762275991103933847,
        -0.1582406353682430100171529,
        -1.4448522368604799215319189,
        0.25369333656622899964006023,
        0.91484424622974003238340401,
    ]
)

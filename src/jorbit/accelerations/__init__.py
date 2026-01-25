"""Handles accelerations that can be used in the integrator."""

import jax

jax.config.update("jax_enable_x64", True)

from jorbit.accelerations.create_acc_funcs import (
    create_default_ephemeris_acceleration_func,
    create_ephem_grav_harmonics_acceleration_func,
    create_gr_ephemeris_acceleration_func,
    create_newtonian_ephemeris_acceleration_func,
    create_static_generic_acceleration_func,
)
from jorbit.accelerations.gr import ppn_gravity, static_ppn_gravity
from jorbit.accelerations.grav_harmonics import grav_harmonics
from jorbit.accelerations.newtonian import newtonian_gravity
from jorbit.accelerations.nongrav import nongrav_acceleration

__all__ = [
    "create_default_ephemeris_acceleration_func",
    "create_ephem_grav_harmonics_acceleration_func",
    "create_gr_ephemeris_acceleration_func",
    "create_newtonian_ephemeris_acceleration_func",
    "create_static_generic_acceleration_func",
    "grav_harmonics",
    "newtonian_gravity",
    "nongrav_acceleration",
    "ppn_gravity",
    "static_ppn_gravity",
]

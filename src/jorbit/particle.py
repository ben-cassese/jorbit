import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from astropy.time import Time

from jorbit.utils.states import KeplerianState


class Particle:
    def __init__(
        self,
        x=None,
        v=None,
        elements=None,
        time=None,
        log_gm=0,
        observations=None,
        name="",
    ):
        self.x = x
        self.v = v
        self.elements = elements
        self.log_gm = log_gm
        self.time = time
        self.observations = observations
        self.name = name

        self.checks()

    def __repr__(self):
        return f"Particle: {self.name}"

    def checks(self):
        if self.elements is not None:
            assert self.x is None, "Cannot provide both x and elements"
            assert self.v is None, "Cannot provide both v and elements"

            if isinstance(self.elements, dict):
                self.elements = KeplerianState(**self.elements)

        elif self.x is not None:
            assert self.v is not None, "Must provide both x and v"

            # self.elements =

        assert self.time is not None, "Must provide a single epoch for the particle"
        if isinstance(self.time, type(Time("2023-01-01"))):
            self.time = self.time.tdb.jd

        else:
            raise ValueError(
                "time must be either astropy.time.Time or float (interpreted as JD in"
                " TDB)"
            )

"""The JAX-compatible functions for manipulating JPL DE ephemeris data."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
class EphemerisProcessor:
    """A pytree-compatible class for processing JPL DE ephemeris data.

    This class provides functionality to evaluate Chebyshev polynomials for computing
    planetary positions and velocities from JPL Development Ephemerides (DE) data.
    It is compatible with JAX's pytree system for automatic differentiation and
    parallelization.
    """

    def __init__(self, init, intlen, coeffs, log_gms):
        """Initializes the EphemerisProcessor with Chebyshev polynomial data.

        Args:
            init (jnp.ndarray):
                Initial epoch times for each body in Julian days TDB.
            intlen (jnp.ndarray):
                Length of each Chebyshev polynomial interval in seconds.
            coeffs (jnp.ndarray):
                Chebyshev polynomial coefficients array of shape
                (3, degree+1, n_intervals) where 3 represents x,y,z coordinates.
            log_gms (jnp.ndarray):
                Natural log of gravitational parameters (GM) for each body in
                AU^3/day^2.
        """
        self.init = init
        self.intlen = intlen
        self.coeffs = coeffs
        self.log_gms = log_gms

    def tree_flatten(self):
        """Flattens the class instance for JAX pytree compatibility.

        Returns:
            tuple: A tuple of (children, auxiliary_data) where children contains
                the arrays to be transformed and auxiliary_data is None.
        """
        children = (self.init, self.intlen, self.coeffs, self.log_gms)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstructs a class instance from flattened data.

        Args:
            aux_data: Auxiliary data (unused).
            children (tuple): Tuple of arrays to reconstruct the instance.

        Returns:
            EphemerisProcessor: A new instance of the class.
        """
        return cls(*children)

    @jax.jit
    def eval_cheby(self, coefficients, x):
        """Evaluates a Chebyshev polynomial using Clenshaw's algorithm.

        Implements Clenshaw's recurrence formula for evaluating Chebyshev polynomials
        in a numerically stable way.

        Args:
            coefficients (jnp.ndarray): Chebyshev polynomial coefficients.
            x (float): Input value in the domain [-1, 1].

        Returns:
            tuple:
                A tuple containing (jnp.ndarray, The evaluated polynomial value;
                jnp.ndarray, Intermediate values used for velocity computation).
        """
        b_ii = jnp.zeros(3)
        b_i = jnp.zeros(3)

        def scan_func(X, a):
            b_i, b_ii = X
            tmp = b_i
            b_i = a + 2 * x * b_i - b_ii
            b_ii = tmp
            return (b_i, b_ii), b_i

        (b_i, b_ii), s = jax.lax.scan(scan_func, (b_i, b_ii), coefficients[:-1])
        return coefficients[-1] + x * b_i - b_ii, s

    @jax.jit
    def _individual_state(self, init, intlen, coeffs, tdb):
        """Computes position and velocity for a single body in the ephemeris at a given time.

        Args:
            init (float): Initial epoch time in Julian days TDB.
            intlen (float): Length of Chebyshev polynomial interval in seconds.
            coeffs (jnp.ndarray): Chebyshev coefficients for the body.
            tdb (float): Requested time in Julian days TDB.

        Returns:
            tuple:
                A tuple of jnp.ndarrays, first positions in AU, then velocities in
                AU/day.
        """
        tdb2 = 0.0  # leaving in case we ever decide to increase the time precision and use 2 floats
        _, _, n = coeffs.shape

        # 2451545.0 is the J2000 epoch in TDB
        index1, offset1 = jnp.divmod((tdb - 2451545.0) * 86400.0 - init, intlen)
        index2, offset2 = jnp.divmod(tdb2 * 86400.0, intlen)
        index3, offset = jnp.divmod(offset1 + offset2, intlen)
        index = (index1 + index2 + index3).astype(int)

        omegas = index == n
        index = jnp.where(omegas, index - 1, index)
        offset = jnp.where(omegas, offset + intlen, offset)

        coefficients = coeffs[:, :, index]

        s = 2.0 * offset / intlen - 1.0

        # Position
        x, As = self.eval_cheby(coefficients, s)  # in km here

        # Velocity
        Q = self.eval_cheby(2 * As, s)
        v = Q[0] - As[-1]
        v /= intlen
        v *= 2.0  # in km/s here

        # # Acceleration
        # a = self.eval_cheby(4 * Q[1], s)[0] - 2 * Q[1][-1]
        # a /= intlen**2
        # a *= 4.0  # in km/s^2 here

        # Convert to AU, AU/day, AU/day^2
        return (
            x.T * 6.684587122268446e-09,
            v.T * 0.0005775483273639937,
            # a.T * 49.900175484249054,
        )

    @jax.jit
    def state(self, tdb):
        """Computes positions and velocities for all bodies in the ephemeris at a given time.

        Args:
            tdb (float): Requested time in Julian days TDB (Barycentric Dynamical Time).

        Returns:
            tuple:
                A tuple of jnp.ndarrays, first positions in AU, then velocities in AU/day.
        """
        x, v = jax.vmap(self._individual_state, in_axes=(0, 0, 0, None))(
            self.init, self.intlen, self.coeffs, tdb
        )
        return x, v


@jax.tree_util.register_pytree_node_class
class EphemerisPostProcessor:
    """A pytree-compatible class for post-processing multiple ephemeris calculations.

    Useful when one ephemeris (e.g. the asteroids) provides coordinates based on the
    positions of an object in another (e.g. the sun).

    Attributes:
        log_gms (jnp.ndarray):
            Concatenated array of natural log of gravitational parameters (GM) from all
            ephemeris processors.
    """

    def __init__(self, ephs, postprocessing_func):
        """Initializes the EphemerisPostProcessor with multiple ephemeris processors.

        Args:
            ephs (list[EphemerisProcessor]):
                List of ephemeris processor instances.
            postprocessing_func (callable):
                Function to process the combined state vectors.
        """
        self.ephs = ephs
        self.postprocessing_func = postprocessing_func
        log_gms = jnp.empty(0)
        for eph in ephs:
            log_gms = jnp.concatenate([log_gms, eph.log_gms])
        self.log_gms = log_gms

    def tree_flatten(self):
        """Flattens the class instance for JAX pytree compatibility.

        Returns:
            tuple: A tuple of (children, auxiliary_data) where children contains
                the ephemeris processors and post-processing function, and
                auxiliary_data is None.
        """
        children = (self.ephs, self.postprocessing_func)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstructs a class instance from flattened data.

        Args:
            aux_data: Auxiliary data (unused).
            children (tuple): Tuple containing ephemeris processors and
                post-processing function.

        Returns:
            EphemerisPostProcessor: A new instance of the class.
        """
        return cls(*children)

    @jax.jit
    def state(self, tdb):
        """Computes combined and post-processed state vectors for all bodies.

        This method:
        1. Initializes empty arrays for position and velocity
        2. Computes states for each ephemeris processor
        3. Concatenates the results vertically
        4. Applies the post-processing function to the combined states

        Args:
            tdb (float): Requested time in Julian days TDB (Barycentric Dynamical Time).

        Returns:
            tuple:
                The post-processed state vectors. The exact structure depends on
                the post-processing function, but typically includes
                - jnp.ndarray: Processed position vectors, shape (n_total_bodies, 3)
                - jnp.ndarray: Processed velocity vectors, shape (n_total_bodies, 3)
        """
        x = jnp.empty((0, 3))
        v = jnp.empty((0, 3))
        # a = jnp.empty((0, 3))
        for eph in self.ephs:
            _x, _v = eph.state(tdb)
            x = jnp.vstack([x, _x])
            v = jnp.vstack([v, _v])
            # a = jnp.vstack([a, _a])
        return self.postprocessing_func(x, v)  # , a)

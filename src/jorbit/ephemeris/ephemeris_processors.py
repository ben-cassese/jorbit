import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
class EphemerisProcessor:
    def __init__(self, init, intlen, coeffs, log_gms):
        self.init = init
        self.intlen = intlen
        self.coeffs = coeffs
        self.log_gms = log_gms

    def tree_flatten(self):
        children = (self.init, self.intlen, self.coeffs, self.log_gms)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @jax.jit
    def eval_cheby(self, coefficients, x):
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
        x, v = jax.vmap(self._individual_state, in_axes=(0, 0, 0, None))(
            self.init, self.intlen, self.coeffs, tdb
        )
        return x, v


@jax.tree_util.register_pytree_node_class
class EphemerisPostProcessor:
    def __init__(self, ephs, postprocessing_func):
        self.ephs = ephs
        self.postprocessing_func = postprocessing_func
        log_gms = jnp.empty(0)
        for eph in ephs:
            log_gms = jnp.concatenate([log_gms, eph.log_gms])
        self.log_gms = log_gms

    def tree_flatten(self):
        children = (self.ephs, self.postprocessing_func)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @jax.jit
    def state(self, tdb):
        x = jnp.empty((0, 3))
        v = jnp.empty((0, 3))
        # a = jnp.empty((0, 3))
        for eph in self.ephs:
            _x, _v = eph.state(tdb)
            x = jnp.vstack([x, _x])
            v = jnp.vstack([v, _v])
            # a = jnp.vstack([a, _a])
        return self.postprocessing_func(x, v)  # , a)

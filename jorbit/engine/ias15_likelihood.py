import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def scanned_array_update(x, indecies, vals):
    def scan_fun(carry, scan_over):
        ind, val = scan_over
        return (carry.at[ind].set(val), None)

    return jax.lax.scan(scan_fun, x, (indecies, vals))[0]


def parser(
    x,
    tracer_xs,
    tracer_vs,
    massive_xs,
    massive_vs,
    massive_gms,
    free_tracer_x_inds,
    free_tracer_v_inds,
    free_massive_x_inds,
    free_massive_v_inds,
    free_massive_gm_inds,
    planet_gms,
    fit_planet_gms,
    asteroid_gms,
    fit_asteroid_gms,
):
    lb = 0
    ub = free_tracer_x_inds.shape[0]
    tracer_xs = scanned_array_update(tracer_xs, free_tracer_x_inds, x[:ub])

    lb = ub
    ub += free_tracer_v_inds.shape[0]
    tracer_vs = scanned_array_update(tracer_vs, free_tracer_v_inds, x[lb:ub])

    lb = ub
    ub += free_massive_x_inds.shape[0]
    massive_xs = scanned_array_update(massive_xs, free_massive_x_inds, x[lb:ub])

    lb = ub
    ub += free_massive_v_inds.shape[0]
    massive_vs = scanned_array_update(massive_vs, free_massive_v_inds, x[lb:ub])

    lb = ub
    ub += free_massive_gm_inds.shape[0]
    massive_gms = scanned_array_update(massive_gms, free_massive_gm_inds, x[lb:ub])

    lb = ub
    ub += jnp.sum(fit_planet_gms)
    planet_gms = scanned_array_update(planet_gms, fit_planet_gms, x[lb:ub])

    lb = ub
    ub += jnp.sum(fit_asteroid_gms)
    asteroid_gms = scanned_array_update(asteroid_gms, fit_asteroid_gms, x[lb:ub])

    return (
        tracer_xs,
        tracer_vs,
        massive_xs,
        massive_vs,
        massive_gms,
        planet_gms,
        asteroid_gms,
    )

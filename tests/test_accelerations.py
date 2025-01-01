import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpy as np
import rebound
import reboundx

from jorbit.utils.states import SystemState
from jorbit.accelerations.gr import ppn_gravity
from jorbit.accelerations.newtonian import newtonian_gravity


def _gr_agreement_w_reboundx(n_tracer, n_massive, seed):
    np.random.seed(seed)
    n = n_tracer
    m = n_massive
    massive_x = []
    massive_v = []
    ms = []
    sim = rebound.Simulation()
    for i in range(m):
        xs = np.random.normal(0, 1, 3) * 1000
        vs = np.random.normal(0, 1, 3)
        massive_x.append(xs)
        massive_v.append(vs)
        m = np.random.uniform(0, 1)
        ms.append(m)
        sim.add(m=m, x=xs[0], y=xs[1], z=xs[2], vx=vs[0], vy=vs[1], vz=vs[2])
    tracer_x = []
    tracer_v = []
    for i in range(n):
        xs = np.random.normal(0, 1, 3) * 1000
        vs = np.random.normal(0, 1, 3)
        tracer_x.append(xs)
        tracer_v.append(vs)
        sim.add(m=0.0, x=xs[0], y=xs[1], z=xs[2], vx=vs[0], vy=vs[1], vz=vs[2])
    rebx = reboundx.Extras(sim)
    gr = rebx.load_force("gr_full")
    gr.params["c"] = 10
    gr.params["max_iterations"] = 100
    rebx.add_force(gr)
    sim.integrate(1e-300)
    reb_res = jnp.array([[p.ax, p.ay, p.az] for p in sim.particles])

    tracer_x = jnp.array(tracer_x)
    tracer_v = jnp.array(tracer_v)
    massive_x = jnp.array(massive_x)
    massive_v = jnp.array(massive_v)
    ms = jnp.array(ms)
    s = SystemState(
        tracer_positions=tracer_x,
        tracer_velocities=tracer_v,
        massive_positions=massive_x,
        massive_velocities=massive_v,
        log_gms=jnp.log(ms),
        time=0.0,
        acceleration_func_kwargs={"c2": 100.0},
    )
    jorb_res = ppn_gravity(s)

    print(jnp.max(jnp.abs(jorb_res - reb_res)))
    assert jnp.allclose(jorb_res, reb_res, atol=1e-14, rtol=1e-14)


def _newton_agreement_w_rebound(n_tracer, n_massive, seed):
    np.random.seed(seed)
    n = n_tracer
    m = n_massive
    massive_x = []
    massive_v = []
    ms = []
    sim = rebound.Simulation()
    for i in range(m):
        xs = np.random.normal(0, 1, 3) * 1000
        vs = np.random.normal(0, 1, 3)
        massive_x.append(xs)
        massive_v.append(vs)
        m = np.random.uniform(0, 1)
        ms.append(m)
        sim.add(m=m, x=xs[0], y=xs[1], z=xs[2], vx=vs[0], vy=vs[1], vz=vs[2])
    tracer_x = []
    tracer_v = []
    for i in range(n):
        xs = np.random.normal(0, 1, 3) * 1000
        vs = np.random.normal(0, 1, 3)
        tracer_x.append(xs)
        tracer_v.append(vs)
        sim.add(m=0.0, x=xs[0], y=xs[1], z=xs[2], vx=vs[0], vy=vs[1], vz=vs[2])
    sim.integrate(1e-300)
    reb_res = jnp.array([[p.ax, p.ay, p.az] for p in sim.particles])

    tracer_x = jnp.array(tracer_x)
    tracer_v = jnp.array(tracer_v)
    massive_x = jnp.array(massive_x)
    massive_v = jnp.array(massive_v)
    ms = jnp.array(ms)
    s = SystemState(
        tracer_positions=tracer_x,
        tracer_velocities=tracer_v,
        massive_positions=massive_x,
        massive_velocities=massive_v,
        log_gms=jnp.log(ms),
        time=0.0,
        acceleration_func_kwargs={"c2": 100.0},
    )
    jorb_res = newtonian_gravity(s)

    print(jnp.max(jnp.abs(jorb_res - reb_res)))
    assert jnp.allclose(jorb_res, reb_res, atol=1e-14, rtol=1e-14)


def test_gr_agreement_w_reboundx():
    _gr_agreement_w_reboundx(n_tracer=1, n_massive=1, seed=0)
    _gr_agreement_w_reboundx(n_tracer=100, n_massive=1, seed=1)
    _gr_agreement_w_reboundx(n_tracer=100, n_massive=10, seed=2)
    _gr_agreement_w_reboundx(n_tracer=100, n_massive=100, seed=3)


def test_newton_agreement_w_rebound():
    _newton_agreement_w_rebound(n_tracer=1, n_massive=1, seed=0)
    _newton_agreement_w_rebound(n_tracer=100, n_massive=1, seed=1)
    _newton_agreement_w_rebound(n_tracer=100, n_massive=10, seed=2)
    _newton_agreement_w_rebound(n_tracer=100, n_massive=100, seed=3)
    _newton_agreement_w_rebound(
        n_tracer=10_000, n_massive=20, seed=4
    )  # this is about the limit of reasonable for rebound, but jorbit's can go up to >1e6 tracer particles

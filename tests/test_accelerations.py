"""Tests that the acceleration functions agree with external codes."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from astropy.time import Time

from jorbit import Ephemeris
from jorbit.accelerations.gr import ppn_gravity, static_ppn_gravity
from jorbit.accelerations.newtonian import newtonian_gravity
from jorbit.data.constants import SPEED_OF_LIGHT
from jorbit.utils.states import SystemState


def _gr_agreement_w_reboundx(n_tracer: int, n_massive: int, seed: int) -> None:
    """Test that the jorbit GR acceleration is consistent with reboundx."""
    import rebound
    import reboundx

    np.random.seed(seed)
    massive_x = []
    massive_v = []
    ms = []
    sim = rebound.Simulation()
    for _i in range(n_massive):
        xs = np.random.normal(0, 1, 3) * 1000
        vs = np.random.normal(0, 1, 3)
        massive_x.append(xs)
        massive_v.append(vs)
        m = np.random.uniform(0, 1)
        ms.append(m)
        sim.add(m=m, x=xs[0], y=xs[1], z=xs[2], vx=vs[0], vy=vs[1], vz=vs[2])
    tracer_x = []
    tracer_v = []
    for _i in range(n_tracer):
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
        fixed_perturber_positions=jnp.empty((0, 3)),
        fixed_perturber_velocities=jnp.empty((0, 3)),
        fixed_perturber_log_gms=jnp.empty((0,)),
        acceleration_func_kwargs={"c2": 100.0},
    )
    jorb_res = ppn_gravity(s)

    assert jnp.allclose(jorb_res, reb_res, atol=1e-14, rtol=1e-14)


def _newton_agreement_w_rebound(n_tracer: int, n_massive: int, seed: int) -> None:
    """Test that the jorbit Newtonian acceleration is consistent with rebound."""
    import rebound

    np.random.seed(seed)
    massive_x = []
    massive_v = []
    ms = []
    sim = rebound.Simulation()
    for _i in range(n_massive):
        xs = np.random.normal(0, 1, 3) * 1000
        vs = np.random.normal(0, 1, 3)
        massive_x.append(xs)
        massive_v.append(vs)
        m = np.random.uniform(0, 1)
        ms.append(m)
        sim.add(m=m, x=xs[0], y=xs[1], z=xs[2], vx=vs[0], vy=vs[1], vz=vs[2])
    tracer_x = []
    tracer_v = []
    for _i in range(n_tracer):
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
        fixed_perturber_positions=jnp.empty((0, 3)),
        fixed_perturber_velocities=jnp.empty((0, 3)),
        fixed_perturber_log_gms=jnp.empty((0,)),
        acceleration_func_kwargs={"c2": 100.0},
    )
    jorb_res = newtonian_gravity(s)

    assert jnp.allclose(jorb_res, reb_res, atol=1e-14, rtol=1e-14)


def test_gr_agreement_w_reboundx() -> None:
    """Test that the GR acceleration agrees across several configurations."""
    _gr_agreement_w_reboundx(n_tracer=1, n_massive=1, seed=0)
    _gr_agreement_w_reboundx(n_tracer=100, n_massive=1, seed=1)
    _gr_agreement_w_reboundx(n_tracer=100, n_massive=10, seed=2)
    _gr_agreement_w_reboundx(n_tracer=100, n_massive=100, seed=3)


def test_newton_agreement_w_rebound() -> None:
    """Test that the Newtonian acceleration agrees across several configurations."""
    _newton_agreement_w_rebound(n_tracer=1, n_massive=1, seed=0)
    _newton_agreement_w_rebound(n_tracer=100, n_massive=1, seed=1)
    _newton_agreement_w_rebound(n_tracer=100, n_massive=10, seed=2)
    _newton_agreement_w_rebound(n_tracer=100, n_massive=100, seed=3)
    _newton_agreement_w_rebound(
        n_tracer=10_000, n_massive=20, seed=4
    )  # this is about the limit of reasonable for rebound, but jorbit's can go up to >1e6 tracer particles


def test_static_gr_convergence() -> None:
    """Test that the fixed-iteration GR acceleration converges to the flexible iterations result."""
    eph = Ephemeris()
    massive_positions, massive_velocities = eph.processor.state(
        Time("2026-01-01").tdb.jd
    )
    perturber_log_gms = eph.processor.log_gms

    state = SystemState(
        massive_positions=massive_positions,
        massive_velocities=massive_velocities,
        tracer_positions=massive_positions[0][None, :] + jnp.array([[1e-2, 0, 0]]),
        tracer_velocities=jnp.array([[0.0, 0, 0]]),
        log_gms=perturber_log_gms,
        time=Time("2026-01-01").tdb.jd,
        fixed_perturber_positions=jnp.empty((0, 3)),
        fixed_perturber_velocities=jnp.empty((0, 3)),
        fixed_perturber_log_gms=jnp.empty((0,)),
        acceleration_func_kwargs={"c2": SPEED_OF_LIGHT**2},
    )

    g1 = ppn_gravity(state)
    g2 = static_ppn_gravity(state, 4)

    diff = g1 - g2
    assert jnp.allclose(diff, 0.0, atol=1e-14, rtol=1e-14)


def _make_state(
    n_massive: int, n_tracer: int, n_perturber: int, c2: float, seed: int
) -> SystemState:
    """Create a SystemState with random positions/velocities for testing."""
    rng = np.random.RandomState(seed)
    m_pos = (
        jnp.array(rng.normal(0, 10, (n_massive, 3)))
        if n_massive > 0
        else jnp.empty((0, 3))
    )
    m_vel = (
        jnp.array(rng.normal(0, 1, (n_massive, 3)))
        if n_massive > 0
        else jnp.empty((0, 3))
    )
    m_gms = (
        jnp.array(rng.uniform(0.1, 1.0, n_massive))
        if n_massive > 0
        else jnp.empty((0,))
    )
    t_pos = (
        jnp.array(rng.normal(0, 10, (n_tracer, 3)))
        if n_tracer > 0
        else jnp.empty((0, 3))
    )
    t_vel = (
        jnp.array(rng.normal(0, 1, (n_tracer, 3)))
        if n_tracer > 0
        else jnp.empty((0, 3))
    )
    p_pos = (
        jnp.array(rng.normal(0, 10, (n_perturber, 3)))
        if n_perturber > 0
        else jnp.empty((0, 3))
    )
    p_vel = (
        jnp.array(rng.normal(0, 1, (n_perturber, 3)))
        if n_perturber > 0
        else jnp.empty((0, 3))
    )
    p_gms = (
        jnp.array(rng.uniform(0.1, 1.0, n_perturber))
        if n_perturber > 0
        else jnp.empty((0,))
    )

    return SystemState(
        massive_positions=m_pos,
        massive_velocities=m_vel,
        tracer_positions=t_pos,
        tracer_velocities=t_vel,
        log_gms=jnp.log(m_gms) if n_massive > 0 else jnp.empty((0,)),
        time=0.0,
        fixed_perturber_positions=p_pos,
        fixed_perturber_velocities=p_vel,
        fixed_perturber_log_gms=jnp.log(p_gms) if n_perturber > 0 else jnp.empty((0,)),
        acceleration_func_kwargs={"c2": c2},
    )


def test_ppn_static_convergence() -> None:
    """Test that static_ppn_gravity converges to ppn_gravity across configurations."""
    configs = [
        (5, 0, 0),  # massive only
        (5, 3, 0),  # massive + tracers
        (5, 3, 4),  # perturbers + massive + tracers
        (0, 3, 4),  # perturbers + tracers only
    ]
    for n_m, n_t, n_p in configs:
        state = _make_state(n_m, n_t, n_p, c2=100.0, seed=42)
        res_ppn = ppn_gravity(state)
        res_static = static_ppn_gravity(state, 10)
        assert jnp.allclose(res_ppn, res_static, atol=1e-15), (
            f"P={n_p},M={n_m},T={n_t}: ppn vs static max diff="
            f"{float(jnp.max(jnp.abs(res_ppn - res_static)))}"
        )


def test_ppn_fixed_perturber_equivalence() -> None:
    """Test that moving particles from massive to fixed_perturber gives same tracer accel.

    When massive bodies are moved to fixed_perturber fields, the tracer
    accelerations should match to machine precision. The massive particle
    accelerations are not returned in the perturber configuration, so only
    the tracer portion is compared.
    """
    rng = np.random.RandomState(123)
    n_bodies = 5
    n_tracers = 3

    body_pos = jnp.array(rng.normal(0, 10, (n_bodies, 3)))
    body_vel = jnp.array(rng.normal(0, 1, (n_bodies, 3)))
    body_gms = jnp.array(rng.uniform(0.1, 1.0, n_bodies))
    tracer_pos = jnp.array(rng.normal(0, 10, (n_tracers, 3)))
    tracer_vel = jnp.array(rng.normal(0, 1, (n_tracers, 3)))

    # Config A: bodies as massive particles
    state_a = SystemState(
        massive_positions=body_pos,
        massive_velocities=body_vel,
        tracer_positions=tracer_pos,
        tracer_velocities=tracer_vel,
        log_gms=jnp.log(body_gms),
        time=0.0,
        fixed_perturber_positions=jnp.empty((0, 3)),
        fixed_perturber_velocities=jnp.empty((0, 3)),
        fixed_perturber_log_gms=jnp.empty((0,)),
        acceleration_func_kwargs={"c2": 100.0},
    )
    res_a = ppn_gravity(state_a)
    tracer_acc_a = res_a[n_bodies:]  # tracer portion

    # Config B: same bodies as fixed perturbers
    state_b = SystemState(
        massive_positions=jnp.empty((0, 3)),
        massive_velocities=jnp.empty((0, 3)),
        tracer_positions=tracer_pos,
        tracer_velocities=tracer_vel,
        log_gms=jnp.empty((0,)),
        time=0.0,
        fixed_perturber_positions=body_pos,
        fixed_perturber_velocities=body_vel,
        fixed_perturber_log_gms=jnp.log(body_gms),
        acceleration_func_kwargs={"c2": 100.0},
    )
    tracer_acc_b = ppn_gravity(state_b)

    assert jnp.allclose(tracer_acc_a, tracer_acc_b, atol=1e-15), (
        f"Tracer accel mismatch: max diff="
        f"{float(jnp.max(jnp.abs(tracer_acc_a - tracer_acc_b)))}"
    )

"""DoubleDouble-precision PPN gravity acceleration model.

This is a DoubleDouble-precision version of the PPN gravity model in gr.py.
All intermediate computations are performed in DoubleDouble (~31 decimal digits)
arithmetic to minimize floating-point error accumulation. This trades speed for
precision — it is significantly slower than the float64 version but should be
accurate to near the limits of the PPN formulation itself.

The physics and algorithm are identical to ppn_gravity in gr.py, which is based
on REBOUNDx (Tamayo et al. 2020) and Newhall et al. (1983).
"""

import jax

jax.config.update("jax_enable_x64", True)
from functools import partial

import jax.numpy as jnp

from jorbit.data.constants import SPEED_OF_LIGHT
from jorbit.utils.doubledouble import (
    DoubleDouble,
    dd_broadcast_to,
    dd_concatenate,
    dd_exp,
    dd_sqrt,
    dd_sum,
    dd_where,
    dd_zeros,
    dd_zeros_like,
)
from jorbit.utils.states import SystemState


def _to_dd(x: jnp.ndarray) -> DoubleDouble:
    """Convert a plain jnp.ndarray to DoubleDouble (lo = 0)."""
    return DoubleDouble(x, jnp.zeros_like(x))


def _dd_ppn_constant_terms(
    t_vel: DoubleDouble,
    t_v2: DoubleDouble,
    s_vel: DoubleDouble,
    s_gms: DoubleDouble,
    s_a_newt: DoubleDouble,
    dx: DoubleDouble,
    r: DoubleDouble,
    r2: DoubleDouble,
    r3: DoubleDouble,
    dv: DoubleDouble,
    a1_total: DoubleDouble,
    a2_per_source: DoubleDouble,
    c2: DoubleDouble,
    mask: jnp.ndarray,
) -> DoubleDouble:
    """Compute the constant PPN terms in DoubleDouble precision.

    Args:
        t_vel: Target velocities in COM frame, (N_t, 3).
        t_v2: Target velocity squared, (N_t,).
        s_vel: Source velocities in COM frame, (N_s, 3).
        s_gms: Source GMs, (N_s,).
        s_a_newt: Newtonian acceleration on each source, (N_s, 3).
        dx: Target - source displacements, (N_t, N_s, 3).
        r: Pairwise distances, (N_t, N_s).
        r2: Pairwise distances squared, (N_t, N_s).
        r3: r^3, (N_t, N_s).
        dv: Target - source velocity differences in COM frame, (N_t, N_s, 3).
        a1_total: Pre-computed total a1 sum for each target, (N_t,).
        a2_per_source: Pre-computed a2 sum for each source, (N_s,).
        c2: Speed of light squared (DD scalar).
        mask: (N_t, N_s) boolean mask for valid pairs.

    Returns:
        Constant PPN corrections on targets from sources, (N_t, 3).
    """
    N_t = dx.shape[0]
    N_s = dx.shape[1]

    # s_v2: (N_s,)
    s_v2 = dd_sum(s_vel * s_vel, axis=-1)

    # vdot: (N_t, N_s) — dot product of target and source velocities
    # t_vel[:, None, :] * s_vel[None, :, :] then sum over last axis
    t_vel_expanded = DoubleDouble(
        t_vel.hi[:, None, :], t_vel.lo[:, None, :]
    )  # (N_t, 1, 3)
    s_vel_expanded = DoubleDouble(
        s_vel.hi[None, :, :], s_vel.lo[None, :, :]
    )  # (1, N_s, 3)
    vdot = dd_sum(t_vel_expanded * s_vel_expanded, axis=-1)  # (N_t, N_s)

    # Broadcast a1 and a2 to (N_t, N_s)
    a1 = dd_broadcast_to(
        DoubleDouble(a1_total.hi[:, None], a1_total.lo[:, None]), (N_t, N_s)
    )
    a2 = dd_broadcast_to(
        DoubleDouble(a2_per_source.hi[None, :], a2_per_source.lo[None, :]),
        (N_t, N_s),
    )

    # a3 = -t_v2 / c2, broadcast to (N_t, N_s)
    neg_t_v2_over_c2 = DoubleDouble(-t_v2.hi, -t_v2.lo) / c2
    a3 = dd_broadcast_to(
        DoubleDouble(neg_t_v2_over_c2.hi[:, None], neg_t_v2_over_c2.lo[:, None]),
        (N_t, N_s),
    )

    # a4 = -2 * s_v2 / c2, broadcast to (N_t, N_s)
    neg2_sv2_over_c2 = (DoubleDouble(-2.0) * s_v2) / c2
    a4 = dd_broadcast_to(
        DoubleDouble(neg2_sv2_over_c2.hi[None, :], neg2_sv2_over_c2.lo[None, :]),
        (N_t, N_s),
    )

    # a5 = 4/c2 * vdot
    a5 = (DoubleDouble(4.0) / c2) * vdot

    # a6 = (3 / (2*c2)) * (dx . s_vel)^2 / r2
    # dx * s_vel[None, :, :] summed over last axis
    s_vel_exp2 = DoubleDouble(s_vel.hi[None, :, :], s_vel.lo[None, :, :])  # (1, N_s, 3)
    a6_0 = dd_sum(dx * s_vel_exp2, axis=-1)  # (N_t, N_s)
    a6 = (DoubleDouble(3.0) / (DoubleDouble(2.0) * c2)) * (a6_0 * a6_0) / r2

    # a7 = (dx . s_a_newt) / (2*c2)
    s_a_newt_exp = DoubleDouble(
        s_a_newt.hi[None, :, :], s_a_newt.lo[None, :, :]
    )  # (1, N_s, 3)
    a7 = dd_sum(dx * s_a_newt_exp, axis=-1) / (DoubleDouble(2.0) * c2)  # (N_t, N_s)

    factor1 = a1 + a2 + a3 + a4 + a5 + a6 + a7  # (N_t, N_s)

    # part1 = s_gms[None, :, None] * dx * factor1[:, :, None] / r3[:, :, None]
    s_gms_exp = DoubleDouble(
        s_gms.hi[None, :, None], s_gms.lo[None, :, None]
    )  # (1, N_s, 1)
    factor1_exp = DoubleDouble(
        factor1.hi[:, :, None], factor1.lo[:, :, None]
    )  # (N_t, N_s, 1)
    r3_exp = DoubleDouble(r3.hi[:, :, None], r3.lo[:, :, None])  # (N_t, N_s, 1)
    part1 = s_gms_exp * dx * factor1_exp / r3_exp  # (N_t, N_s, 3)

    # factor2 = sum(dx * (4*t_vel - 3*s_vel), axis=-1)
    t_vel_exp3d = DoubleDouble(
        t_vel.hi[:, None, :], t_vel.lo[:, None, :]
    )  # (N_t, 1, 3)
    s_vel_exp3d = DoubleDouble(
        s_vel.hi[None, :, :], s_vel.lo[None, :, :]
    )  # (1, N_s, 3)
    factor2 = dd_sum(
        dx * (DoubleDouble(4.0) * t_vel_exp3d - DoubleDouble(3.0) * s_vel_exp3d),
        axis=-1,
    )  # (N_t, N_s)

    # part2 = s_gms * (factor2 * dv / r3 + 3.5 * s_a_newt / r) / c2
    factor2_exp = DoubleDouble(
        factor2.hi[:, :, None], factor2.lo[:, :, None]
    )  # (N_t, N_s, 1)
    r_exp = DoubleDouble(r.hi[:, :, None], r.lo[:, :, None])  # (N_t, N_s, 1)
    part2 = (
        s_gms_exp
        * (
            factor2_exp * dv / r3_exp
            + (DoubleDouble(7.0) / DoubleDouble(2.0)) * s_a_newt_exp / r_exp
        )
        / c2
    )  # (N_t, N_s, 3)

    total = part1 + part2  # (N_t, N_s, 3)

    # Masked sum over source axis (axis=1)
    mask_3d = mask[:, :, None]  # (N_t, N_s, 1) — broadcast over xyz
    return dd_sum(total, axis=1, where=mask_3d)  # (N_t, 3)


def _dd_ppn_non_constant(
    s_gms: DoubleDouble,
    s_a_est: DoubleDouble,
    dx: DoubleDouble,
    r: DoubleDouble,
    r3: DoubleDouble,
    c2: DoubleDouble,
    mask: jnp.ndarray,
) -> DoubleDouble:
    """Compute non-constant PPN terms in DoubleDouble precision.

    Args:
        s_gms: Source GMs, (N_s,).
        s_a_est: Current GR correction estimate for sources, (N_s, 3).
        dx: Target - source displacements, (N_t, N_s, 3).
        r: Pairwise distances, (N_t, N_s).
        r3: r^3, (N_t, N_s).
        c2: Speed of light squared (DD scalar).
        mask: (N_t, N_s) boolean mask.

    Returns:
        Non-constant PPN corrections on targets, (N_t, 3).
    """
    # rdota = sum(dx * s_a_est[None, :, :], axis=-1)
    s_a_est_exp = DoubleDouble(
        s_a_est.hi[None, :, :], s_a_est.lo[None, :, :]
    )  # (1, N_s, 3)
    rdota = dd_sum(dx * s_a_est_exp, axis=-1)  # (N_t, N_s)

    s_gms_exp = DoubleDouble(
        s_gms.hi[None, :, None], s_gms.lo[None, :, None]
    )  # (1, N_s, 1)
    rdota_exp = DoubleDouble(
        rdota.hi[:, :, None], rdota.lo[:, :, None]
    )  # (N_t, N_s, 1)
    r3_exp = DoubleDouble(r3.hi[:, :, None], r3.lo[:, :, None])  # (N_t, N_s, 1)
    r_exp = DoubleDouble(r.hi[:, :, None], r.lo[:, :, None])  # (N_t, N_s, 1)

    non_const_terms = (s_gms_exp / (DoubleDouble(2.0) * c2)) * (
        dx * rdota_exp / r3_exp + DoubleDouble(7.0) * s_a_est_exp / r_exp
    )  # (N_t, N_s, 3)

    mask_3d = mask[:, :, None]
    return dd_sum(non_const_terms, axis=1, where=mask_3d)  # (N_t, 3)


def _dd_compute_ppn_setup(inputs: SystemState) -> tuple:
    """Compute geometry, COM frame, Newtonian accelerations, and constant PPN terms.

    All computations are performed in DoubleDouble precision. Inputs from the
    SystemState are converted to DD at the boundary.

    Args:
        inputs: SystemState with the instantaneous state.

    Returns:
        Tuple of DD arrays needed by ppn_gravity_dd.
    """
    c2_val = inputs.acceleration_func_kwargs.get("c2", SPEED_OF_LIGHT**2)
    c2 = DoubleDouble(jnp.asarray(c2_val))

    P = inputs.fixed_perturber_positions.shape[0]
    M = inputs.massive_positions.shape[0]
    T = inputs.tracer_positions.shape[0]
    N = P + M + T
    S = P + M

    # Convert all positions/velocities/GMs to DD
    p_pos = _to_dd(inputs.fixed_perturber_positions)  # (P, 3)
    p_vel = _to_dd(inputs.fixed_perturber_velocities)  # (P, 3)
    p_gms = dd_exp(
        _to_dd(inputs.fixed_perturber_log_gms)
    )  # (P,) — exp in DD for precision

    m_pos = _to_dd(inputs.massive_positions)  # (M, 3)
    m_vel = _to_dd(inputs.massive_velocities)  # (M, 3)
    m_gms = dd_exp(_to_dd(inputs.log_gms))  # (M,)

    t_pos = _to_dd(inputs.tracer_positions)  # (T, 3)
    t_vel = _to_dd(inputs.tracer_velocities)  # (T, 3)

    # All particles (iteration targets)
    all_pos = dd_concatenate([p_pos, m_pos, t_pos], axis=0)  # (N, 3)
    all_vel = dd_concatenate([p_vel, m_vel, t_vel], axis=0)  # (N, 3)

    # All sources
    src_pos = dd_concatenate([p_pos, m_pos], axis=0)  # (S, 3)
    src_vel = dd_concatenate([p_vel, m_vel], axis=0)  # (S, 3)
    src_gms = dd_concatenate([p_gms, m_gms], axis=0)  # (S,)

    # ---- Geometry: all targets → all sources (N, S) ----
    all_pos_exp = DoubleDouble(
        all_pos.hi[:, None, :], all_pos.lo[:, None, :]
    )  # (N, 1, 3)
    src_pos_exp = DoubleDouble(
        src_pos.hi[None, :, :], src_pos.lo[None, :, :]
    )  # (1, S, 3)
    dx_ns = all_pos_exp - src_pos_exp  # (N, S, 3)

    r2_ns = dd_sum(dx_ns * dx_ns, axis=-1)  # (N, S)
    r_ns = dd_sqrt(r2_ns)  # (N, S)
    r3_ns = r2_ns * r_ns  # (N, S)

    # Self-interaction mask (plain boolean, not DD)
    mask_ns = jnp.ones((N, S), dtype=bool)
    mask_ns = mask_ns.at[:S, :].set(~jnp.eye(S, dtype=bool))

    # ---- Newtonian acceleration on all targets from all sources ----
    # prefac = where(mask, 1/r3, 0)
    one_over_r3 = DoubleDouble(1.0) / r3_ns
    zero_ns = dd_zeros((N, S))
    prefac_ns = dd_where(mask_ns, one_over_r3, zero_ns)  # (N, S)

    # a_newt = -sum(prefac * dx * src_gms, axis=1)
    prefac_exp = DoubleDouble(
        prefac_ns.hi[:, :, None], prefac_ns.lo[:, :, None]
    )  # (N, S, 1)
    src_gms_exp = DoubleDouble(
        src_gms.hi[None, :, None], src_gms.lo[None, :, None]
    )  # (1, S, 1)
    a_newt_terms = prefac_exp * dx_ns * src_gms_exp  # (N, S, 3)
    a_newt_all = -dd_sum(a_newt_terms, axis=1)  # (N, 3)

    # ---- COM frame ----
    total_gm = dd_sum(src_gms)
    # v_com = sum(src_vel * src_gms[:, None], axis=0) / total_gm
    src_gms_vel_exp = DoubleDouble(src_gms.hi[:, None], src_gms.lo[:, None])  # (S, 1)
    v_com = dd_sum(src_vel * src_gms_vel_exp, axis=0) / total_gm  # (3,)

    all_vel_com = all_vel - v_com  # (N, 3) — broadcasting
    src_vel_com = src_vel - v_com  # (S, 3)
    all_v2 = dd_sum(all_vel_com * all_vel_com, axis=-1)  # (N,)

    # Velocity differences in COM frame: (N, S, 3)
    all_vel_com_exp = DoubleDouble(
        all_vel_com.hi[:, None, :], all_vel_com.lo[:, None, :]
    )  # (N, 1, 3)
    src_vel_com_exp = DoubleDouble(
        src_vel_com.hi[None, :, :], src_vel_com.lo[None, :, :]
    )  # (1, S, 3)
    dv_ns_com = all_vel_com_exp - src_vel_com_exp  # (N, S, 3)

    # ---- a1: sum over k!=i of 4*GM_k/r_ik for each target ----
    four_over_c2 = DoubleDouble(4.0) / c2
    src_gms_r_exp = DoubleDouble(src_gms.hi[None, :], src_gms.lo[None, :])  # (1, S)
    a1_terms = four_over_c2 * src_gms_r_exp / r_ns  # (N, S)
    a1_total = dd_sum(a1_terms, axis=1, where=mask_ns)  # (N,)

    # ---- a2: sum over k!=j of GM_k/r_jk for each source ----
    src_pos_exp_ss = DoubleDouble(
        src_pos.hi[:, None, :], src_pos.lo[:, None, :]
    )  # (S, 1, 3)
    src_pos_exp_ss2 = DoubleDouble(
        src_pos.hi[None, :, :], src_pos.lo[None, :, :]
    )  # (1, S, 3)
    src_dx = src_pos_exp_ss - src_pos_exp_ss2  # (S, S, 3)
    src_r2 = dd_sum(src_dx * src_dx, axis=-1)  # (S, S)
    src_r = dd_sqrt(src_r2)  # (S, S)
    src_mask = ~jnp.eye(S, dtype=bool)

    one_over_c2 = DoubleDouble(1.0) / c2
    src_gms_ss_exp = DoubleDouble(src_gms.hi[None, :], src_gms.lo[None, :])  # (1, S)
    a2_terms = one_over_c2 * src_gms_ss_exp / src_r  # (S, S)
    a2_per_source = dd_sum(a2_terms, axis=1, where=src_mask)  # (S,)

    # ---- Newtonian acceleration on sources ----
    a_newt_sources = DoubleDouble(a_newt_all.hi[:S], a_newt_all.lo[:S])  # (S, 3)

    # ---- Constant PPN terms ----
    a_const = _dd_ppn_constant_terms(
        t_vel=all_vel_com,
        t_v2=all_v2,
        s_vel=src_vel_com,
        s_gms=src_gms,
        s_a_newt=a_newt_sources,
        dx=dx_ns,
        r=r_ns,
        r2=r2_ns,
        r3=r3_ns,
        dv=dv_ns_com,
        a1_total=a1_total,
        a2_per_source=a2_per_source,
        c2=c2,
        mask=mask_ns,
    )  # (N, 3)

    return (
        c2,
        P,
        S,
        src_gms,
        dx_ns,
        r_ns,
        r3_ns,
        mask_ns,
        a_newt_all,
        a_const,
    )


@partial(jax.jit, static_argnames=["max_iterations"])
def ppn_gravity_dd(
    inputs: SystemState,
    max_iterations: int = 10,
) -> DoubleDouble:
    """Compute PPN gravity accelerations in DoubleDouble precision.

    This is a high-precision version of ppn_gravity that performs all
    intermediate computations in DoubleDouble (~31 decimal digit) arithmetic.
    The algorithm is identical: Newtonian + iterative PPN corrections with
    convergence checking, but every floating-point operation preserves the
    extra ~15 digits provided by the DD representation.

    Args:
        inputs (SystemState): The instantaneous state of the system.
        max_iterations (int): Maximum iterations for GR correction convergence.

    Returns:
        DoubleDouble:
            The 3D acceleration felt by each particle, ordered by
            massive particles first followed by tracer particles.
    """
    (
        c2,
        P,
        S,
        src_gms,
        dx_ns,
        r_ns,
        r3_ns,
        mask_ns,
        a_newt_all,
        a_const,
    ) = _dd_compute_ppn_setup(inputs)

    def compute_non_const(a_gr_sources: DoubleDouble) -> DoubleDouble:
        return _dd_ppn_non_constant(
            src_gms, a_gr_sources, dx_ns, r_ns, r3_ns, c2, mask_ns
        )

    a_gr_init = a_const  # (N, 3)

    def do_nothing(carry: tuple) -> tuple:
        return carry

    def do_iteration(carry: tuple) -> tuple:
        _a_prev, a_curr_gr, _ = carry
        a_gr_sources = DoubleDouble(a_curr_gr.hi[:S], a_curr_gr.lo[:S])
        non_const = compute_non_const(a_gr_sources)
        a_next_gr = a_const + non_const
        # Convergence check on hi parts (sufficient for termination criterion)
        ratio = jnp.max(
            jnp.abs((a_next_gr.hi - a_curr_gr.hi) / a_next_gr.hi), initial=0.0
        )
        return (a_curr_gr, a_next_gr, ratio)

    def body_fn(carry: tuple, _: None) -> tuple:
        _a_prev, _a_curr, ratio = carry
        should_continue = ratio > jnp.finfo(jnp.float64).eps
        new_carry = jax.lax.cond(should_continue, do_iteration, do_nothing, carry)
        return new_carry, None

    init_carry = (dd_zeros_like(a_gr_init), a_gr_init, 1.0)
    final_carry, _ = jax.lax.scan(body_fn, init_carry, None, length=max_iterations)
    _, a_final_gr, _ = final_carry

    # Combine Newtonian + GR, return only M+T particles (skip perturbers)
    a_total = a_newt_all + a_final_gr  # (N, 3) DD
    return DoubleDouble(a_total.hi[P:], a_total.lo[P:])


@partial(jax.jit, static_argnames=["fixed_iterations"])
def static_ppn_gravity_dd(
    inputs: SystemState, fixed_iterations: int = 3
) -> DoubleDouble:
    """Compute PPN gravity in DoubleDouble precision with fixed iteration count.

    Similar to ppn_gravity_dd but uses a fixed number of iterations and no
    logic branching, making it more suitable for use inside jax.lax.scan.

    Args:
        inputs (SystemState): The instantaneous state of the system.
        fixed_iterations (int): Fixed number of GR correction iterations.

    Returns:
        DoubleDouble:
            The 3D acceleration felt by each particle, ordered by
            massive particles first followed by tracer particles.
    """
    (
        c2,
        P,
        S,
        src_gms,
        dx_ns,
        r_ns,
        r3_ns,
        mask_ns,
        a_newt_all,
        a_const,
    ) = _dd_compute_ppn_setup(inputs)

    def scan_fn(a_curr_gr: DoubleDouble, _: None) -> tuple:
        a_gr_sources = DoubleDouble(a_curr_gr.hi[:S], a_curr_gr.lo[:S])
        non_const = _dd_ppn_non_constant(
            src_gms, a_gr_sources, dx_ns, r_ns, r3_ns, c2, mask_ns
        )
        a_next_gr = a_const + non_const
        return a_next_gr, None

    a_final_gr, _ = jax.lax.scan(scan_fn, a_const, None, length=fixed_iterations)

    a_total = a_newt_all + a_final_gr
    return DoubleDouble(a_total.hi[P:], a_total.lo[P:])


@partial(jax.jit, static_argnames=["fixed_iterations"])
def _ppn_gravity_dd_core(
    all_pos: DoubleDouble,
    all_vel: DoubleDouble,
    src_gms: DoubleDouble,
    c2: DoubleDouble,
    fixed_iterations: int = 3,
) -> DoubleDouble:
    """Compute PPN gravity in DoubleDouble precision from raw DD arrays.

    This is a core function that works directly on DoubleDouble inputs without
    requiring a SystemState. All N particles are treated as both sources and
    targets (with self-interaction masked out). Useful for the DD integrator
    where positions/velocities are already in DD format.

    Args:
        all_pos: Positions of all particles, shape (N, 3).
        all_vel: Velocities of all particles, shape (N, 3).
        src_gms: GMs of all particles (already exponentiated), shape (N,).
        c2: Speed of light squared as a DoubleDouble scalar.
        fixed_iterations: Number of GR correction iterations.

    Returns:
        DoubleDouble: Accelerations on all particles, shape (N, 3).
    """
    N = all_pos.hi.shape[0]

    # Geometry: all targets -> all sources (N, N)
    all_pos_exp = DoubleDouble(
        all_pos.hi[:, None, :], all_pos.lo[:, None, :]
    )  # (N, 1, 3)
    src_pos_exp = DoubleDouble(
        all_pos.hi[None, :, :], all_pos.lo[None, :, :]
    )  # (1, N, 3)
    dx_ns = all_pos_exp - src_pos_exp  # (N, N, 3)

    r2_ns = dd_sum(dx_ns * dx_ns, axis=-1)  # (N, N)
    r_ns = dd_sqrt(r2_ns)
    r3_ns = r2_ns * r_ns

    # Self-interaction mask
    mask_ns = ~jnp.eye(N, dtype=bool)  # (N, N)

    # Newtonian acceleration on all particles
    one_over_r3 = DoubleDouble(1.0) / r3_ns
    zero_ns = dd_zeros((N, N))
    prefac_ns = dd_where(mask_ns, one_over_r3, zero_ns)

    prefac_exp = DoubleDouble(prefac_ns.hi[:, :, None], prefac_ns.lo[:, :, None])
    src_gms_exp = DoubleDouble(src_gms.hi[None, :, None], src_gms.lo[None, :, None])
    a_newt_all = -dd_sum(prefac_exp * dx_ns * src_gms_exp, axis=1)  # (N, 3)

    # COM frame
    total_gm = dd_sum(src_gms)
    src_gms_vel_exp = DoubleDouble(src_gms.hi[:, None], src_gms.lo[:, None])
    v_com = dd_sum(all_vel * src_gms_vel_exp, axis=0) / total_gm

    all_vel_com = all_vel - v_com
    all_v2 = dd_sum(all_vel_com * all_vel_com, axis=-1)

    # Velocity differences in COM frame
    all_vel_com_exp = DoubleDouble(
        all_vel_com.hi[:, None, :], all_vel_com.lo[:, None, :]
    )
    src_vel_com_exp = DoubleDouble(
        all_vel_com.hi[None, :, :], all_vel_com.lo[None, :, :]
    )
    dv_ns_com = all_vel_com_exp - src_vel_com_exp

    # a1: sum over k!=i of 4*GM_k/r_ik
    four_over_c2 = DoubleDouble(4.0) / c2
    src_gms_r_exp = DoubleDouble(src_gms.hi[None, :], src_gms.lo[None, :])
    a1_terms = four_over_c2 * src_gms_r_exp / r_ns
    a1_total = dd_sum(a1_terms, axis=1, where=mask_ns)

    # a2: sum over k!=j of GM_k/r_jk
    one_over_c2 = DoubleDouble(1.0) / c2
    src_gms_ss_exp = DoubleDouble(src_gms.hi[None, :], src_gms.lo[None, :])
    a2_terms = one_over_c2 * src_gms_ss_exp / r_ns
    a2_per_source = dd_sum(a2_terms, axis=1, where=mask_ns)

    # Newtonian acceleration on sources (= all particles here)
    a_newt_sources = a_newt_all

    # Constant PPN terms
    a_const = _dd_ppn_constant_terms(
        t_vel=all_vel_com,
        t_v2=all_v2,
        s_vel=all_vel_com,
        s_gms=src_gms,
        s_a_newt=a_newt_sources,
        dx=dx_ns,
        r=r_ns,
        r2=r2_ns,
        r3=r3_ns,
        dv=dv_ns_com,
        a1_total=a1_total,
        a2_per_source=a2_per_source,
        c2=c2,
        mask=mask_ns,
    )

    # Fixed iterations for GR corrections
    def scan_fn(a_curr_gr: DoubleDouble, _: None) -> tuple:
        non_const = _dd_ppn_non_constant(
            src_gms, a_curr_gr, dx_ns, r_ns, r3_ns, c2, mask_ns
        )
        a_next_gr = a_const + non_const
        return a_next_gr, None

    a_final_gr, _ = jax.lax.scan(scan_fn, a_const, None, length=fixed_iterations)

    return a_newt_all + a_final_gr

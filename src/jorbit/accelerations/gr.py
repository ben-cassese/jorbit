# """General Relativity/PPN acceleration model.

# These are pythonized/jaxified versions of acceleration models within REBOUNDx,
# Tamayo et al. (2020) (DOI: 10.1093/mnras/stz2870). The gr_full function is the
# equivalent of rebx_calculate_gr_full in REBOUNDx, which is itself based on
# Newhall et al. (1984) (bibcode: 1983A&A...125..150N)
# The original code is available at https://github.com/dtamayo/reboundx/blob/502abf3066d9bae174cb20538294c916e73391cd/src/gr_full.c

# Many thanks to the REBOUNDx developers for their work, and for making it open source!
# Accessed Fall 2024
# """

# import jax

# jax.config.update("jax_enable_x64", True)
# from functools import partial

# import jax.numpy as jnp

# from jorbit.data.constants import SPEED_OF_LIGHT
# from jorbit.utils.states import SystemState


# def _ppn_constant_terms(
#     t_vel: jnp.ndarray,
#     t_v2: jnp.ndarray,
#     s_vel: jnp.ndarray,
#     s_gms: jnp.ndarray,
#     s_a_newt: jnp.ndarray,
#     dx: jnp.ndarray,
#     r: jnp.ndarray,
#     r2: jnp.ndarray,
#     r3: jnp.ndarray,
#     dv: jnp.ndarray,
#     a1_total: jnp.ndarray,
#     a2_per_source: jnp.ndarray,
#     c2: float,
#     mask: jnp.ndarray,
# ) -> jnp.ndarray:
#     """Compute the constant PPN terms from sources onto targets.

#     The "constant" terms are those that depend on the Newtonian acceleration of
#     the sources (computed once) rather than the iteratively-refined GR correction.

#     Args:
#         t_vel: Target velocities in COM frame, (N_t, 3).
#         t_v2: Target velocity squared, (N_t,).
#         s_vel: Source velocities in COM frame, (N_s, 3).
#         s_gms: Source GMs, (N_s,).
#         s_a_newt: Newtonian acceleration on each source, (N_s, 3).
#         dx: Target - source displacements, (N_t, N_s, 3).
#         r: Pairwise distances, (N_t, N_s).
#         r2: Pairwise distances squared, (N_t, N_s).
#         r3: r^3, (N_t, N_s).
#         dv: Target - source velocity differences in COM frame, (N_t, N_s, 3).
#         a1_total: Pre-computed total a1 sum for each target (over ALL sources),
#             (N_t,). Broadcast to (N_t, N_s) internally.
#         a2_per_source: Pre-computed a2 sum for each source (over ALL other
#             particles), (N_s,). Broadcast to (N_t, N_s) internally.
#         c2: Speed of light squared.
#         mask: (N_t, N_s) boolean mask for valid pairs (False excludes self).

#     Returns:
#         a_const: Constant PPN corrections on targets from these sources, (N_t, 3).
#     """
#     N_t = dx.shape[0]
#     N_s = dx.shape[1]

#     s_v2 = jnp.sum(s_vel * s_vel, axis=-1)  # (N_s,)
#     vdot = jnp.sum(t_vel[:, None, :] * s_vel[None, :, :], axis=-1)  # (N_t, N_s)

#     a1 = jnp.broadcast_to(a1_total[:, None], (N_t, N_s))
#     a2 = jnp.broadcast_to(a2_per_source[None, :], (N_t, N_s))

#     a3 = jnp.broadcast_to(-t_v2[:, None] / c2, (N_t, N_s))
#     a4 = jnp.broadcast_to(-2.0 * s_v2[None, :] / c2, (N_t, N_s))
#     a5 = (4.0 / c2) * vdot

#     a6_0 = jnp.sum(dx * s_vel[None, :, :], axis=-1)  # (N_t, N_s)
#     a6 = (3.0 / (2 * c2)) * (a6_0**2) / r2

#     a7 = jnp.sum(dx * s_a_newt[None, :, :], axis=-1) / (2 * c2)  # (N_t, N_s)

#     factor1 = a1 + a2 + a3 + a4 + a5 + a6 + a7
#     part1 = s_gms[None, :, None] * dx * factor1[:, :, None] / r3[:, :, None]

#     factor2 = jnp.sum(
#         dx * (4 * t_vel[:, None, :] - 3 * s_vel[None, :, :]), axis=-1
#     )  # (N_t, N_s)
#     part2 = (
#         s_gms[None, :, None]
#         * (
#             factor2[:, :, None] * dv / r3[:, :, None]
#             + 7.0 / 2.0 * s_a_newt[None, :, :] / r[:, :, None]
#         )
#         / c2
#     )

#     return jnp.sum(part1 + part2, axis=1, where=mask[:, :, None])


# def _ppn_non_constant(
#     s_gms: jnp.ndarray,
#     s_a_est: jnp.ndarray,
#     dx: jnp.ndarray,
#     r: jnp.ndarray,
#     r3: jnp.ndarray,
#     c2: float,
#     mask: jnp.ndarray,
# ) -> jnp.ndarray:
#     """Compute non-constant PPN terms from sources onto targets.

#     These terms depend on the current estimate of the source accelerations
#     (the GR correction part, not the Newtonian part).

#     Args:
#         s_gms: Source GMs, (N_s,).
#         s_a_est: Current GR correction estimate for sources, (N_s, 3).
#         dx: Target - source displacements, (N_t, N_s, 3).
#         r: Pairwise distances, (N_t, N_s).
#         r3: r^3, (N_t, N_s).
#         c2: Speed of light squared.
#         mask: (N_t, N_s) boolean mask for valid pairs.

#     Returns:
#         Non-constant PPN corrections on targets, (N_t, 3).
#     """
#     rdota = jnp.sum(dx * s_a_est[None, :, :], axis=-1)  # (N_t, N_s)
#     non_const_terms = (s_gms[None, :, None] / (2.0 * c2)) * (
#         dx * rdota[:, :, None] / r3[:, :, None]
#         + 7.0 * s_a_est[None, :, :] / r[:, :, None]
#     )
#     return jnp.sum(non_const_terms, axis=1, where=mask[:, :, None])


# def _compute_ppn_setup(inputs: SystemState) -> tuple:
#     """Compute geometry, COM frame, Newtonian accelerations, and constant PPN terms.

#     Fixed perturber inputs are wrapped in stop_gradient at the source, so no
#     gradients flow through perturber quantities anywhere downstream.

#     The constant PPN terms and non-constant iteration geometry are computed for
#     ALL particles (P+M+T), but tracer sources are skipped (GM=0).

#     Returns:
#         Tuple of arrays needed by ppn_gravity and static_ppn_gravity.
#     """
#     c2 = inputs.acceleration_func_kwargs.get("c2", SPEED_OF_LIGHT**2)

#     P = inputs.fixed_perturber_positions.shape[0]
#     M = inputs.massive_positions.shape[0]
#     T = inputs.tracer_positions.shape[0]
#     N = P + M + T  # all particles (targets in the iteration)
#     S = P + M  # all sources with GM > 0

#     # Fixed perturbers come from pre-computed ephemerides; we never need
#     # gradients through them, so stop_gradient at the source eliminates all
#     # downstream gradient computation through perturber quantities.
#     p_pos = jax.lax.stop_gradient(inputs.fixed_perturber_positions)  # (P, 3)
#     p_vel = jax.lax.stop_gradient(inputs.fixed_perturber_velocities)  # (P, 3)
#     p_gms = jax.lax.stop_gradient(jnp.exp(inputs.fixed_perturber_log_gms))  # (P,)

#     m_pos = inputs.massive_positions  # (M, 3)
#     m_vel = inputs.massive_velocities  # (M, 3)
#     m_gms = jnp.exp(inputs.log_gms)  # (M,)

#     t_pos = inputs.tracer_positions  # (T, 3)
#     t_vel = inputs.tracer_velocities  # (T, 3)

#     # All particles (iteration targets) = concat(perturbers, massive, tracers)
#     all_pos = jnp.concatenate([p_pos, m_pos, t_pos], axis=0)  # (N, 3)
#     all_vel = jnp.concatenate([p_vel, m_vel, t_vel], axis=0)  # (N, 3)

#     # All sources = concat(perturbers, massive)
#     src_pos = jnp.concatenate([p_pos, m_pos], axis=0)  # (S, 3)
#     src_vel = jnp.concatenate([p_vel, m_vel], axis=0)  # (S, 3)
#     src_gms = jnp.concatenate([p_gms, m_gms])  # (S,)

#     # ---- Geometry: all targets → all sources (N, S) ----
#     dx_ns = all_pos[:, None, :] - src_pos[None, :, :]  # (N, S, 3)
#     r2_ns = jnp.sum(dx_ns * dx_ns, axis=-1)  # (N, S)
#     r_ns = jnp.sqrt(r2_ns)
#     r3_ns = r2_ns * r_ns

#     # Self-interaction mask: target i == source j when i < S and i == j
#     # (targets 0..P-1 are perturbers = sources 0..P-1,
#     #  targets P..P+M-1 are massive = sources P..P+M-1,
#     #  targets P+M..N-1 are tracers = no matching source)
#     mask_ns = jnp.ones((N, S), dtype=bool)
#     mask_ns = mask_ns.at[:S, :].set(~jnp.eye(S, dtype=bool))

#     # ---- Newtonian acceleration on all targets from all sources ----
#     prefac_ns = jnp.where(mask_ns, 1.0 / r3_ns, 0.0)
#     a_newt_all = -jnp.sum(
#         prefac_ns[:, :, None] * dx_ns * src_gms[None, :, None], axis=1
#     )  # (N, 3)

#     # ---- COM frame ----
#     total_gm = jnp.sum(src_gms)
#     v_com = jnp.sum(src_vel * src_gms[:, None], axis=0) / total_gm

#     all_vel_com = all_vel - v_com
#     src_vel_com = src_vel - v_com
#     all_v2 = jnp.sum(all_vel_com * all_vel_com, axis=-1)  # (N,)

#     # Velocity differences in COM frame
#     dv_ns_com = all_vel_com[:, None, :] - src_vel_com[None, :, :]  # (N, S, 3)

#     # ---- a1: sum over k!=i of 4*GM_k/r_ik for each target ----
#     a1_total = jnp.sum(
#         (4.0 / c2) * src_gms[None, :] / r_ns, axis=1, where=mask_ns
#     )  # (N,)

#     # ---- a2: sum over k!=j of GM_k/r_jk for each source ----
#     # For source j, sum GM_k/r_jk over all other sources k != j.
#     # (Tracers have GM=0 so excluding them doesn't change the sum.)
#     src_dx = src_pos[:, None, :] - src_pos[None, :, :]  # (S, S, 3)
#     src_r2 = jnp.sum(src_dx * src_dx, axis=-1)  # (S, S)
#     src_r = jnp.sqrt(src_r2)
#     src_mask = ~jnp.eye(S, dtype=bool)
#     a2_per_source = jnp.sum(
#         (1.0 / c2) * src_gms[None, :] / src_r, axis=1, where=src_mask
#     )  # (S,)

#     # ---- Newtonian acceleration on sources (for a7 and part2 in constant terms) ----
#     a_newt_sources = a_newt_all[:S]  # (S, 3)

#     # ---- Constant PPN terms for all targets from all sources ----
#     a_const = _ppn_constant_terms(
#         t_vel=all_vel_com,
#         t_v2=all_v2,
#         s_vel=src_vel_com,
#         s_gms=src_gms,
#         s_a_newt=a_newt_sources,
#         dx=dx_ns,
#         r=r_ns,
#         r2=r2_ns,
#         r3=r3_ns,
#         dv=dv_ns_com,
#         a1_total=a1_total,
#         a2_per_source=a2_per_source,
#         c2=c2,
#         mask=mask_ns,
#     )  # (N, 3)

#     return (
#         c2,
#         P,
#         S,
#         # Non-constant iteration geometry (N targets x S sources)
#         src_gms,
#         dx_ns,
#         r_ns,
#         r3_ns,
#         mask_ns,
#         # Newtonian and constant terms
#         a_newt_all,
#         a_const,
#     )


# # equivalent of rebx_calculate_gr_full in reboundx
# @partial(jax.jit, static_argnames=["max_iterations"])
# def ppn_gravity(
#     inputs: SystemState,
#     max_iterations: int = 10,
# ) -> jnp.ndarray:
#     """Compute the acceleration felt by each particle due to PPN gravity.

#     Equivalent of rebx_calculate_gr_full in reboundx. Uses a structured approach
#     that separates perturber, massive, and tracer contributions to avoid
#     unnecessary N² interactions. Tracer sources (GM=0) are excluded from all
#     computations, reducing the source dimension from P+M+T to P+M.

#     Note: We use "stop_gradient" on perturbers that are passed as fixed inputs, so
#     any gradients with respect to these perturber quantities will not be correct. To
#     track gradients with respect to perturbers, they must be included as "massive"
#     particles, not "fixed perturbers".

#     Args:
#         inputs (SystemState): The instantaneous state of the system.
#         max_iterations (int): The maximum number of iterations for the GR corrections
#             to converge.

#     Returns:
#         jnp.ndarray:
#             The 3D acceleration felt by each particle, ordered by massive particles
#             first followed by tracer particles.
#     """
#     (
#         c2,
#         P,
#         S,
#         src_gms,
#         dx_ns,
#         r_ns,
#         r3_ns,
#         mask_ns,
#         a_newt_all,
#         a_const,
#     ) = _compute_ppn_setup(inputs)

#     def compute_non_const(a_gr_sources: jnp.ndarray) -> jnp.ndarray:
#         """Non-constant PPN from all sources onto all targets."""
#         return _ppn_non_constant(
#             src_gms,
#             a_gr_sources,
#             dx_ns,
#             r_ns,
#             r3_ns,
#             c2,
#             mask_ns,
#         )

#     # Initialize: GR correction = constant terms (matches old code's a_curr = a_const)
#     a_gr_init = a_const  # (N, 3)

#     def do_nothing(carry: tuple) -> tuple:
#         return carry

#     def do_iteration(carry: tuple) -> tuple:
#         _a_prev, a_curr_gr, _ = carry
#         # Use GR correction of sources (first S = P+M entries) for non-constant
#         a_gr_sources = a_curr_gr[:S]
#         non_const = compute_non_const(a_gr_sources)
#         a_next_gr = a_const + non_const
#         ratio = jnp.max(jnp.abs((a_next_gr - a_curr_gr) / a_next_gr), initial=0.0)
#         return (a_curr_gr, a_next_gr, ratio)

#     def body_fn(carry: tuple, _: None) -> tuple:
#         _a_prev, _a_curr, ratio = carry
#         should_continue = ratio > jnp.finfo(jnp.float64).eps
#         new_carry = jax.lax.cond(should_continue, do_iteration, do_nothing, carry)
#         return new_carry, None

#     init_carry = (jnp.zeros_like(a_gr_init), a_gr_init, 1.0)
#     final_carry, _ = jax.lax.scan(body_fn, init_carry, None, length=max_iterations)
#     _, a_final_gr, _ = final_carry

#     # Combine Newtonian + GR, return only M+T particles (skip perturbers)
#     return (a_newt_all + a_final_gr)[P:]


# @partial(jax.jit, static_argnames=["fixed_iterations"])
# def static_ppn_gravity(inputs: SystemState, fixed_iterations: int = 3) -> jnp.ndarray:
#     """Compute the acceleration felt by each particle due to PPN gravity.

#     Similar to ppn_gravity, but uses a fixed number of iterations for the GR
#     corrections to converge and contains no logic branching.

#     Args:
#         inputs (SystemState): The instantaneous state of the system.
#         fixed_iterations (int):
#             The fixed number of iterations for the GR corrections to converge.
#             Default is 3.

#     Returns:
#         jnp.ndarray:
#             The 3D acceleration felt by each particle, ordered by massive particles
#             first followed by tracer particles.
#     """
#     (
#         c2,
#         P,
#         S,
#         src_gms,
#         dx_ns,
#         r_ns,
#         r3_ns,
#         mask_ns,
#         a_newt_all,
#         a_const,
#     ) = _compute_ppn_setup(inputs)

#     def scan_fn(a_curr_gr: jnp.ndarray, _: None) -> tuple:
#         a_gr_sources = a_curr_gr[:S]
#         non_const = _ppn_non_constant(
#             src_gms,
#             a_gr_sources,
#             dx_ns,
#             r_ns,
#             r3_ns,
#             c2,
#             mask_ns,
#         )
#         a_next_gr = a_const + non_const
#         return a_next_gr, None

#     # Initialize with constant terms
#     a_final_gr, _ = jax.lax.scan(scan_fn, a_const, None, length=fixed_iterations)

#     return (a_newt_all + a_final_gr)[P:]


# @jax.jit
# def static_ppn_gravity_tracer(inputs: SystemState) -> jnp.ndarray:
#     """Compute PPN gravity on tracers from perturbers only, avoiding N² scaling.

#     Optimized for the common case where we only need GR corrections from
#     fixed perturbers onto tracer particles. Skips perturber-perturber
#     interactions entirely, reducing the computation from O(N²) to O(P*T)
#     where P is the number of perturbers and T is the number of tracers.

#     The non-constant PPN term uses Newtonian (rather than GR-corrected)
#     perturber accelerations. This introduces an O(c⁻⁴) error (~0.003 mas
#     at 1 AU), well below the 0.1 mas accuracy requirement, while avoiding
#     the expensive PxP perturber-perturber PPN computation + iteration.

#     Args:
#         inputs (SystemState): The instantaneous state of the system.
#             Must have no massive particles (massive_positions.shape[0] == 0).

#     Returns:
#         jnp.ndarray:
#             The 3D acceleration felt by each tracer particle, shape (T, 3).
#     """
#     c2 = inputs.acceleration_func_kwargs.get("c2", SPEED_OF_LIGHT**2)

#     # Perturber properties (P perturbers) — stop_gradient since we never need
#     # gradients through fixed perturber quantities.
#     p_pos = jax.lax.stop_gradient(inputs.fixed_perturber_positions)  # (P, 3)
#     p_vel = jax.lax.stop_gradient(inputs.fixed_perturber_velocities)  # (P, 3)
#     p_gms = jax.lax.stop_gradient(jnp.exp(inputs.fixed_perturber_log_gms))  # (P,)

#     # Tracer properties (T tracers)
#     t_pos = inputs.tracer_positions  # (T, 3)
#     t_vel = inputs.tracer_velocities  # (T, 3)

#     # Displacement from tracers to perturbers: (T, P, 3)
#     dx = t_pos[:, None, :] - p_pos[None, :, :]
#     r2 = jnp.sum(dx * dx, axis=-1)  # (T, P)
#     r = jnp.sqrt(r2)  # (T, P)
#     r3 = r2 * r  # (T, P)

#     # Newtonian acceleration on tracers from perturbers
#     a_newt = -jnp.sum(dx * p_gms[None, :, None] / r3[:, :, None], axis=1)  # (T, 3)

#     dv = t_vel[:, None, :] - p_vel[None, :, :]  # (T, P, 3)

#     # Center-of-mass velocity (perturbers only, tracers are massless)
#     total_gm = jnp.sum(p_gms)
#     v_com = jnp.sum(p_vel * p_gms[:, None], axis=0) / total_gm

#     # Shift to COM frame
#     p_vel_com = p_vel - v_com
#     t_vel_com = t_vel - v_com

#     # Velocity-dependent terms
#     # v² for tracers and perturbers
#     t_v2 = jnp.sum(t_vel_com * t_vel_com, axis=-1)  # (T,)
#     p_v2 = jnp.sum(p_vel_com * p_vel_com, axis=-1)  # (P,)

#     # vi·vj for (tracer_i, perturber_j)
#     vdot = jnp.sum(t_vel_com[:, None, :] * p_vel_com[None, :, :], axis=-1)  # (T, P)

#     # a1: sum over k!=i of 4*gm_k/r_ik for each tracer i
#     # Tracers only interact with perturbers (no tracer-tracer)
#     a1 = jnp.sum((4.0 / c2) * p_gms[None, :] / r, axis=1)  # (T,)
#     a1 = jnp.broadcast_to(a1[:, None], (t_pos.shape[0], p_pos.shape[0]))

#     # a2: sum over k!=j of gm_k/r_jk for each perturber j
#     # Perturber-perturber distances are independent of tracer state (derived
#     # entirely from stopped p_pos/p_gms), so already gradient-free.
#     p_dx = p_pos[:, None, :] - p_pos[None, :, :]  # (P, P, 3)
#     p_r2 = jnp.sum(p_dx * p_dx, axis=-1)  # (P, P)
#     p_r = jnp.sqrt(p_r2)  # (P, P)
#     p_mask = ~jnp.eye(p_pos.shape[0], dtype=bool)
#     a2_per_perturber = jnp.sum(
#         (1.0 / c2) * p_gms[None, :] / p_r,
#         axis=1,
#         where=p_mask,
#     )  # (P,)
#     # Also add tracer->perturber contribution (but tracer gm=0, so this is 0)
#     a2 = jnp.broadcast_to(a2_per_perturber[None, :], (t_pos.shape[0], p_pos.shape[0]))

#     a3 = jnp.broadcast_to(-t_v2[:, None] / c2, (t_pos.shape[0], p_pos.shape[0]))
#     a4 = jnp.broadcast_to(-2.0 * p_v2[None, :] / c2, (t_pos.shape[0], p_pos.shape[0]))
#     a5 = (4.0 / c2) * vdot

#     a6_0 = jnp.sum(dx * p_vel_com[None, :, :], axis=-1)  # (T, P)
#     a6 = (3.0 / (2 * c2)) * (a6_0**2) / r2

#     # Newtonian acceleration on each perturber from other perturbers.
#     # Independent of tracer state, so stop_gradient avoids VJP overhead.
#     p_prefac = jnp.where(p_mask, 1.0 / (p_r2 * p_r), 0.0)
#     a_newt_perturbers = jax.lax.stop_gradient(
#         -jnp.sum(p_prefac[:, :, None] * p_dx * p_gms[None, :, None], axis=1)
#     )  # (P, 3)

#     a7 = jnp.sum(dx * a_newt_perturbers[None, :, :], axis=-1) / (2 * c2)  # (T, P)

#     factor1 = a1 + a2 + a3 + a4 + a5 + a6 + a7
#     part1 = p_gms[None, :, None] * dx * factor1[:, :, None] / r3[:, :, None]

#     factor2 = jnp.sum(
#         dx * (4 * t_vel_com[:, None, :] - 3 * p_vel_com[None, :, :]), axis=-1
#     )  # (T, P)
#     part2 = (
#         p_gms[None, :, None]
#         * (
#             factor2[:, :, None] * dv / r3[:, :, None]
#             + 7.0 / 2.0 * a_newt_perturbers[None, :, :] / r[:, :, None]
#         )
#         / c2
#     )

#     a_const = jnp.sum(part1 + part2, axis=1)  # (T, 3)

#     # Non-constant correction: depends on perturber accelerations a_j.
#     # In principle, the non-constant term should use GR-corrected perturber
#     # accelerations (converged via iteration). However, since the non-constant
#     # term is itself O(c⁻²) and the GR correction to perturber accelerations
#     # is also O(c⁻²), using Newtonian perturber accelerations introduces only
#     # an O(c⁻⁴) error — ~10⁻¹⁰ relative, or ~0.003 mas at 1 AU, well below
#     # the 0.1 mas accuracy threshold.
#     rdota = jnp.sum(dx * a_newt_perturbers[None, :, :], axis=-1)  # (T, P)
#     non_const = jnp.sum(
#         (p_gms[None, :, None] / (2.0 * c2))
#         * (
#             dx * rdota[:, :, None] / r3[:, :, None]
#             + 7.0 * a_newt_perturbers[None, :, :] / r[:, :, None]
#         ),
#         axis=1,
#     )  # (T, 3)
#     a_final = a_const + non_const

#     return a_newt + a_final

########################################################################################
# old, correct version
"""General Relativity/PPN acceleration model.

These are pythonized/jaxified versions of acceleration models within REBOUNDx,
Tamayo et al. (2020) (DOI: 10.1093/mnras/stz2870). The gr_full function is the
equivalent of rebx_calculate_gr_full in REBOUNDx, which is itself based on
Newhall et al. (1984) (bibcode: 1983A&A...125..150N)
The original code is available at https://github.com/dtamayo/reboundx/blob/502abf3066d9bae174cb20538294c916e73391cd/src/gr_full.c

Many thanks to the REBOUNDx developers for their work, and for making it open source!
Accessed Fall 2024
"""

import jax

jax.config.update("jax_enable_x64", True)
from functools import partial

import jax.numpy as jnp

from jorbit.data.constants import SPEED_OF_LIGHT
from jorbit.utils.states import SystemState


def _ppn_constant_terms(
    t_vel: jnp.ndarray,
    t_v2: jnp.ndarray,
    s_vel: jnp.ndarray,
    s_gms: jnp.ndarray,
    s_a_newt: jnp.ndarray,
    dx: jnp.ndarray,
    r: jnp.ndarray,
    r2: jnp.ndarray,
    r3: jnp.ndarray,
    dv: jnp.ndarray,
    a1_total: jnp.ndarray,
    a2_per_source: jnp.ndarray,
    c2: float,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the constant PPN terms from sources onto targets.

    The "constant" terms are those that depend on the Newtonian acceleration of
    the sources (computed once) rather than the iteratively-refined GR correction.

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
        a1_total: Pre-computed total a1 sum for each target (over ALL sources),
            (N_t,). Broadcast to (N_t, N_s) internally.
        a2_per_source: Pre-computed a2 sum for each source (over ALL other
            particles), (N_s,). Broadcast to (N_t, N_s) internally.
        c2: Speed of light squared.
        mask: (N_t, N_s) boolean mask for valid pairs (False excludes self).

    Returns:
        a_const: Constant PPN corrections on targets from these sources, (N_t, 3).
    """
    N_t = dx.shape[0]
    N_s = dx.shape[1]

    s_v2 = jnp.sum(s_vel * s_vel, axis=-1)  # (N_s,)
    vdot = jnp.sum(t_vel[:, None, :] * s_vel[None, :, :], axis=-1)  # (N_t, N_s)

    a1 = jnp.broadcast_to(a1_total[:, None], (N_t, N_s))
    a2 = jnp.broadcast_to(a2_per_source[None, :], (N_t, N_s))

    a3 = jnp.broadcast_to(-t_v2[:, None] / c2, (N_t, N_s))
    a4 = jnp.broadcast_to(-2.0 * s_v2[None, :] / c2, (N_t, N_s))
    a5 = (4.0 / c2) * vdot

    a6_0 = jnp.sum(dx * s_vel[None, :, :], axis=-1)  # (N_t, N_s)
    a6 = (3.0 / (2 * c2)) * (a6_0**2) / r2

    a7 = jnp.sum(dx * s_a_newt[None, :, :], axis=-1) / (2 * c2)  # (N_t, N_s)

    factor1 = a1 + a2 + a3 + a4 + a5 + a6 + a7
    part1 = s_gms[None, :, None] * dx * factor1[:, :, None] / r3[:, :, None]

    factor2 = jnp.sum(
        dx * (4 * t_vel[:, None, :] - 3 * s_vel[None, :, :]), axis=-1
    )  # (N_t, N_s)
    part2 = (
        s_gms[None, :, None]
        * (
            factor2[:, :, None] * dv / r3[:, :, None]
            + 7.0 / 2.0 * s_a_newt[None, :, :] / r[:, :, None]
        )
        / c2
    )

    return jnp.sum(part1 + part2, axis=1, where=mask[:, :, None])


def _ppn_non_constant(
    s_gms: jnp.ndarray,
    s_a_est: jnp.ndarray,
    dx: jnp.ndarray,
    r: jnp.ndarray,
    r3: jnp.ndarray,
    c2: float,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Compute non-constant PPN terms from sources onto targets.

    These terms depend on the current estimate of the source accelerations
    (the GR correction part, not the Newtonian part).

    Args:
        s_gms: Source GMs, (N_s,).
        s_a_est: Current GR correction estimate for sources, (N_s, 3).
        dx: Target - source displacements, (N_t, N_s, 3).
        r: Pairwise distances, (N_t, N_s).
        r3: r^3, (N_t, N_s).
        c2: Speed of light squared.
        mask: (N_t, N_s) boolean mask for valid pairs.

    Returns:
        Non-constant PPN corrections on targets, (N_t, 3).
    """
    rdota = jnp.sum(dx * s_a_est[None, :, :], axis=-1)  # (N_t, N_s)
    non_const_terms = (s_gms[None, :, None] / (2.0 * c2)) * (
        dx * rdota[:, :, None] / r3[:, :, None]
        + 7.0 * s_a_est[None, :, :] / r[:, :, None]
    )
    return jnp.sum(non_const_terms, axis=1, where=mask[:, :, None])


def _compute_ppn_setup(inputs: SystemState) -> tuple:
    """Compute geometry, COM frame, Newtonian accelerations, and constant PPN terms.

    Fixed perturber inputs are wrapped in stop_gradient at the source, so no
    gradients flow through perturber quantities anywhere downstream.

    The constant PPN terms and non-constant iteration geometry are computed for
    ALL particles (P+M+T), but tracer sources are skipped (GM=0).

    Returns:
        Tuple of arrays needed by ppn_gravity and static_ppn_gravity.
    """
    c2 = inputs.acceleration_func_kwargs.get("c2", SPEED_OF_LIGHT**2)

    P = inputs.fixed_perturber_positions.shape[0]
    M = inputs.massive_positions.shape[0]
    T = inputs.tracer_positions.shape[0]
    N = P + M + T  # all particles (targets in the iteration)
    S = P + M  # all sources with GM > 0

    # Fixed perturbers come from pre-computed ephemerides; we never need
    # gradients through them, so stop_gradient at the source eliminates all
    # downstream gradient computation through perturber quantities.
    p_pos = jax.lax.stop_gradient(inputs.fixed_perturber_positions)  # (P, 3)
    p_vel = jax.lax.stop_gradient(inputs.fixed_perturber_velocities)  # (P, 3)
    p_gms = jax.lax.stop_gradient(jnp.exp(inputs.fixed_perturber_log_gms))  # (P,)

    m_pos = inputs.massive_positions  # (M, 3)
    m_vel = inputs.massive_velocities  # (M, 3)
    m_gms = jnp.exp(inputs.log_gms)  # (M,)

    t_pos = inputs.tracer_positions  # (T, 3)
    t_vel = inputs.tracer_velocities  # (T, 3)

    # All particles (iteration targets) = concat(perturbers, massive, tracers)
    all_pos = jnp.concatenate([p_pos, m_pos, t_pos], axis=0)  # (N, 3)
    all_vel = jnp.concatenate([p_vel, m_vel, t_vel], axis=0)  # (N, 3)

    # All sources = concat(perturbers, massive)
    src_pos = jnp.concatenate([p_pos, m_pos], axis=0)  # (S, 3)
    src_vel = jnp.concatenate([p_vel, m_vel], axis=0)  # (S, 3)
    src_gms = jnp.concatenate([p_gms, m_gms])  # (S,)

    # ---- Geometry: all targets → all sources (N, S) ----
    dx_ns = all_pos[:, None, :] - src_pos[None, :, :]  # (N, S, 3)
    r2_ns = jnp.sum(dx_ns * dx_ns, axis=-1)  # (N, S)
    r_ns = jnp.sqrt(r2_ns)
    r3_ns = r2_ns * r_ns

    # Self-interaction mask: target i == source j when i < S and i == j
    # (targets 0..P-1 are perturbers = sources 0..P-1,
    #  targets P..P+M-1 are massive = sources P..P+M-1,
    #  targets P+M..N-1 are tracers = no matching source)
    mask_ns = jnp.ones((N, S), dtype=bool)
    mask_ns = mask_ns.at[:S, :].set(~jnp.eye(S, dtype=bool))

    # ---- Newtonian acceleration on all targets from all sources ----
    prefac_ns = jnp.where(mask_ns, 1.0 / r3_ns, 0.0)
    a_newt_all = -jnp.sum(
        prefac_ns[:, :, None] * dx_ns * src_gms[None, :, None], axis=1
    )  # (N, 3)

    # ---- COM frame ----
    total_gm = jnp.sum(src_gms)
    v_com = jnp.sum(src_vel * src_gms[:, None], axis=0) / total_gm

    all_vel_com = all_vel - v_com
    src_vel_com = src_vel - v_com
    all_v2 = jnp.sum(all_vel_com * all_vel_com, axis=-1)  # (N,)

    # Velocity differences in COM frame
    dv_ns_com = all_vel_com[:, None, :] - src_vel_com[None, :, :]  # (N, S, 3)

    # ---- a1: sum over k!=i of 4*GM_k/r_ik for each target ----
    a1_total = jnp.sum(
        (4.0 / c2) * src_gms[None, :] / r_ns, axis=1, where=mask_ns
    )  # (N,)

    # ---- a2: sum over k!=j of GM_k/r_jk for each source ----
    # For source j, sum GM_k/r_jk over all other sources k != j.
    # (Tracers have GM=0 so excluding them doesn't change the sum.)
    src_dx = src_pos[:, None, :] - src_pos[None, :, :]  # (S, S, 3)
    src_r2 = jnp.sum(src_dx * src_dx, axis=-1)  # (S, S)
    src_r = jnp.sqrt(src_r2)
    src_mask = ~jnp.eye(S, dtype=bool)
    a2_per_source = jnp.sum(
        (1.0 / c2) * src_gms[None, :] / src_r, axis=1, where=src_mask
    )  # (S,)

    # ---- Newtonian acceleration on sources (for a7 and part2 in constant terms) ----
    a_newt_sources = a_newt_all[:S]  # (S, 3)

    # ---- Constant PPN terms for all targets from all sources ----
    a_const = _ppn_constant_terms(
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
        # Non-constant iteration geometry (N targets x S sources)
        src_gms,
        dx_ns,
        r_ns,
        r3_ns,
        mask_ns,
        # Newtonian and constant terms
        a_newt_all,
        a_const,
    )


# equivalent of rebx_calculate_gr_full in reboundx
@partial(jax.jit, static_argnames=["max_iterations"])
def ppn_gravity(
    inputs: SystemState,
    max_iterations: int = 10,
) -> jnp.ndarray:
    """Compute the acceleration felt by each particle due to PPN gravity.

    Equivalent of rebx_calculate_gr_full in reboundx. Uses a structured approach
    that separates perturber, massive, and tracer contributions to avoid
    unnecessary N² interactions. Tracer sources (GM=0) are excluded from all
    computations, reducing the source dimension from P+M+T to P+M.

    Note: We use "stop_gradient" on perturbers that are passed as fixed inputs, so
    any gradients with respect to these perturber quantities will not be correct. To
    track gradients with respect to perturbers, they must be included as "massive"
    particles, not "fixed perturbers".

    Args:
        inputs (SystemState): The instantaneous state of the system.
        max_iterations (int): The maximum number of iterations for the GR corrections
            to converge.

    Returns:
        jnp.ndarray:
            The 3D acceleration felt by each particle, ordered by massive particles
            first followed by tracer particles.
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
    ) = _compute_ppn_setup(inputs)

    def compute_non_const(a_gr_sources: jnp.ndarray) -> jnp.ndarray:
        """Non-constant PPN from all sources onto all targets."""
        return _ppn_non_constant(
            src_gms,
            a_gr_sources,
            dx_ns,
            r_ns,
            r3_ns,
            c2,
            mask_ns,
        )

    # Initialize: GR correction = constant terms (matches old code's a_curr = a_const)
    a_gr_init = a_const  # (N, 3)

    def do_nothing(carry: tuple) -> tuple:
        return carry

    def do_iteration(carry: tuple) -> tuple:
        _a_prev, a_curr_gr, _ = carry
        # Use GR correction of sources (first S = P+M entries) for non-constant
        a_gr_sources = a_curr_gr[:S]
        non_const = compute_non_const(a_gr_sources)
        a_next_gr = a_const + non_const
        ratio = jnp.max(jnp.abs((a_next_gr - a_curr_gr) / a_next_gr), initial=0.0)
        return (a_curr_gr, a_next_gr, ratio)

    def body_fn(carry: tuple, _: None) -> tuple:
        _a_prev, _a_curr, ratio = carry
        should_continue = ratio > jnp.finfo(jnp.float64).eps
        new_carry = jax.lax.cond(should_continue, do_iteration, do_nothing, carry)
        return new_carry, None

    init_carry = (jnp.zeros_like(a_gr_init), a_gr_init, 1.0)
    final_carry, _ = jax.lax.scan(body_fn, init_carry, None, length=max_iterations)
    _, a_final_gr, _ = final_carry

    # Combine Newtonian + GR, return only M+T particles (skip perturbers)
    return (a_newt_all + a_final_gr)[P:]


@partial(jax.jit, static_argnames=["fixed_iterations"])
def static_ppn_gravity(inputs: SystemState, fixed_iterations: int = 3) -> jnp.ndarray:
    """Compute the acceleration felt by each particle due to PPN gravity.

    Similar to ppn_gravity, but uses a fixed number of iterations for the GR
    corrections to converge and contains no logic branching.

    Args:
        inputs (SystemState): The instantaneous state of the system.
        fixed_iterations (int):
            The fixed number of iterations for the GR corrections to converge.
            Default is 3.

    Returns:
        jnp.ndarray:
            The 3D acceleration felt by each particle, ordered by massive particles
            first followed by tracer particles.
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
    ) = _compute_ppn_setup(inputs)

    def scan_fn(a_curr_gr: jnp.ndarray, _: None) -> tuple:
        a_gr_sources = a_curr_gr[:S]
        non_const = _ppn_non_constant(
            src_gms,
            a_gr_sources,
            dx_ns,
            r_ns,
            r3_ns,
            c2,
            mask_ns,
        )
        a_next_gr = a_const + non_const
        return a_next_gr, None

    # Initialize with constant terms
    a_final_gr, _ = jax.lax.scan(scan_fn, a_const, None, length=fixed_iterations)

    return (a_newt_all + a_final_gr)[P:]


def precompute_perturber_ppn(
    p_pos: jnp.ndarray,
    p_vel: jnp.ndarray,
    p_gms: jnp.ndarray,
    c2: float = SPEED_OF_LIGHT**2,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pre-compute perturber-perturber PPN quantities for a single substep.

    Computes the P*P perturber-perturber geometry, Newtonian accelerations,
    gravitational potential sums (a2), and fully converged GR corrections.
    These can then be passed to static_ppn_gravity_tracer via
    acceleration_func_kwargs to avoid redundant P*P work in the hot loop.

    Args:
        p_pos: Perturber positions, shape (P, 3).
        p_vel: Perturber velocities, shape (P, 3).
        p_gms: Perturber GM values (not log), shape (P,).
        c2: Speed of light squared.

    Returns:
        pp_a2: Gravitational potential sum per perturber, shape (P,).
            Σ_{k≠j} GM_k / (c² r_jk) for each perturber j.
        pp_a_newt: Newtonian acceleration on each perturber from others, shape (P, 3).
        pp_a_gr: Fully converged total GR correction on each perturber, shape (P, 3).
    """
    P = p_pos.shape[0]

    # Perturber-perturber geometry
    p_dx = p_pos[:, None, :] - p_pos[None, :, :]  # (P, P, 3)
    p_r2 = jnp.sum(p_dx * p_dx, axis=-1)  # (P, P)
    p_r = jnp.sqrt(p_r2)  # (P, P)
    p_r3 = p_r2 * p_r  # (P, P)
    p_mask = ~jnp.eye(P, dtype=bool)

    # a2: gravitational potential sum per perturber
    pp_a2 = jnp.sum((1.0 / c2) * p_gms[None, :] / p_r, axis=1, where=p_mask)  # (P,)

    # Newtonian acceleration on each perturber from others
    p_prefac = jnp.where(p_mask, 1.0 / p_r3, 0.0)
    pp_a_newt = -jnp.sum(
        p_prefac[:, :, None] * p_dx * p_gms[None, :, None], axis=1
    )  # (P, 3)

    # COM frame
    total_gm = jnp.sum(p_gms)
    v_com = jnp.sum(p_vel * p_gms[:, None], axis=0) / total_gm
    p_vel_com = p_vel - v_com
    p_v2 = jnp.sum(p_vel_com * p_vel_com, axis=-1)  # (P,)

    # Constant PPN terms for perturber-perturber
    pp_vdot = jnp.sum(p_vel_com[:, None, :] * p_vel_com[None, :, :], axis=-1)  # (P, P)
    pp_a1 = jnp.sum((4.0 / c2) * p_gms[None, :] / p_r, axis=1, where=p_mask)  # (P,)
    pp_a1 = jnp.broadcast_to(pp_a1[:, None], (P, P))
    pp_a2_bc = jnp.broadcast_to(pp_a2[None, :], (P, P))
    pp_a3 = jnp.broadcast_to(-p_v2[:, None] / c2, (P, P))
    pp_a4 = jnp.broadcast_to(-2.0 * p_v2[None, :] / c2, (P, P))
    pp_a5 = (4.0 / c2) * pp_vdot
    pp_dv = p_vel_com[:, None, :] - p_vel_com[None, :, :]  # (P, P, 3)
    pp_a6_0 = jnp.sum(p_dx * p_vel_com[None, :, :], axis=-1)  # (P, P)
    pp_a6 = (3.0 / (2 * c2)) * (pp_a6_0**2) / p_r2
    pp_a7 = jnp.sum(p_dx * pp_a_newt[None, :, :], axis=-1) / (2 * c2)  # (P, P)

    pp_factor1 = pp_a1 + pp_a2_bc + pp_a3 + pp_a4 + pp_a5 + pp_a6 + pp_a7
    pp_part1 = p_gms[None, :, None] * p_dx * pp_factor1[:, :, None] / p_r3[:, :, None]
    pp_factor2 = jnp.sum(
        p_dx * (4 * p_vel_com[:, None, :] - 3 * p_vel_com[None, :, :]), axis=-1
    )  # (P, P)
    pp_part2 = (
        p_gms[None, :, None]
        * (
            pp_factor2[:, :, None] * pp_dv / p_r3[:, :, None]
            + 7.0 / 2.0 * pp_a_newt[None, :, :] / p_r[:, :, None]
        )
        / c2
    )
    pp_a_const = jnp.sum(
        pp_part1 + pp_part2, axis=1, where=p_mask[:, :, None]
    )  # (P, 3)

    # Iterate non-constant terms to convergence (3 iterations)
    def pp_scan_fn(a_curr_gr: jnp.ndarray, _: None) -> tuple:
        pp_rdota = jnp.sum(p_dx * a_curr_gr[None, :, :], axis=-1)  # (P, P)
        pp_non_const = jnp.sum(
            (p_gms[None, :, None] / (2.0 * c2))
            * (
                p_dx * pp_rdota[:, :, None] / p_r3[:, :, None]
                + 7.0 * a_curr_gr[None, :, :] / p_r[:, :, None]
            ),
            axis=1,
            where=p_mask[:, :, None],
        )  # (P, 3)
        return pp_a_const + pp_non_const, None

    pp_a_gr, _ = jax.lax.scan(pp_scan_fn, pp_a_const, None, length=3)

    return pp_a2, pp_a_newt, pp_a_gr


@jax.jit
def static_ppn_gravity_tracer(inputs: SystemState) -> jnp.ndarray:
    """Compute PPN gravity on tracers from perturbers only, avoiding N² scaling.

    Optimized for the common case where we only need GR corrections from
    fixed perturbers onto tracer particles. Skips perturber-perturber
    interactions entirely, reducing the computation from O(N²) to O(P*T)
    where P is the number of perturbers and T is the number of tracers.

    Args:
        inputs (SystemState): The instantaneous state of the system.
            Must have no massive particles (massive_positions.shape[0] == 0).
        fixed_iterations (int):
            The fixed number of iterations for the GR corrections to converge.
            Default is 3.

    Returns:
        jnp.ndarray:
            The 3D acceleration felt by each tracer particle, shape (T, 3).
    """
    c2 = inputs.acceleration_func_kwargs.get("c2", SPEED_OF_LIGHT**2)

    # Perturber properties (P perturbers) — stop_gradient since we never need
    # gradients through fixed perturber quantities.
    p_pos = jax.lax.stop_gradient(inputs.fixed_perturber_positions)  # (P, 3)
    p_vel = jax.lax.stop_gradient(inputs.fixed_perturber_velocities)  # (P, 3)
    p_gms = jax.lax.stop_gradient(jnp.exp(inputs.fixed_perturber_log_gms))  # (P,)

    # Tracer properties (T tracers)
    t_pos = inputs.tracer_positions  # (T, 3)
    t_vel = inputs.tracer_velocities  # (T, 3)

    # Displacement from tracers to perturbers: (T, P, 3)
    dx = t_pos[:, None, :] - p_pos[None, :, :]
    r2 = jnp.sum(dx * dx, axis=-1)  # (T, P)
    r = jnp.sqrt(r2)  # (T, P)
    r3 = r2 * r  # (T, P)

    # Newtonian acceleration on tracers from perturbers
    a_newt = -jnp.sum(dx * p_gms[None, :, None] / r3[:, :, None], axis=1)  # (T, 3)

    dv = t_vel[:, None, :] - p_vel[None, :, :]  # (T, P, 3)

    # Center-of-mass velocity (perturbers only, tracers are massless)
    total_gm = jnp.sum(p_gms)
    v_com = jnp.sum(p_vel * p_gms[:, None], axis=0) / total_gm

    # Shift to COM frame
    p_vel_com = p_vel - v_com
    t_vel_com = t_vel - v_com

    # Velocity-dependent terms
    # v² for tracers and perturbers
    t_v2 = jnp.sum(t_vel_com * t_vel_com, axis=-1)  # (T,)
    p_v2 = jnp.sum(p_vel_com * p_vel_com, axis=-1)  # (P,)

    # vi·vj for (tracer_i, perturber_j)
    vdot = jnp.sum(t_vel_com[:, None, :] * p_vel_com[None, :, :], axis=-1)  # (T, P)

    # a1: sum over k!=i of 4*gm_k/r_ik for each tracer i
    # Tracers only interact with perturbers (no tracer-tracer)
    a1 = jnp.sum((4.0 / c2) * p_gms[None, :] / r, axis=1)  # (T,)
    a1 = jnp.broadcast_to(a1[:, None], (t_pos.shape[0], p_pos.shape[0]))

    # Read pre-computed perturber-perturber quantities from kwargs.
    # These are computed once during preprocessing by precompute_perturber_ppn
    # and placed into acceleration_func_kwargs by the integrator.
    a2_per_perturber = jax.lax.stop_gradient(inputs.acceleration_func_kwargs["pp_a2"])
    a_newt_perturbers = jax.lax.stop_gradient(
        inputs.acceleration_func_kwargs["pp_a_newt"]
    )
    a_gr_perturbers = jax.lax.stop_gradient(inputs.acceleration_func_kwargs["pp_a_gr"])

    a2 = jnp.broadcast_to(a2_per_perturber[None, :], (t_pos.shape[0], p_pos.shape[0]))

    a3 = jnp.broadcast_to(-t_v2[:, None] / c2, (t_pos.shape[0], p_pos.shape[0]))
    a4 = jnp.broadcast_to(-2.0 * p_v2[None, :] / c2, (t_pos.shape[0], p_pos.shape[0]))
    a5 = (4.0 / c2) * vdot

    a6_0 = jnp.sum(dx * p_vel_com[None, :, :], axis=-1)  # (T, P)
    a6 = (3.0 / (2 * c2)) * (a6_0**2) / r2

    a7 = jnp.sum(dx * a_newt_perturbers[None, :, :], axis=-1) / (2 * c2)  # (T, P)

    factor1 = a1 + a2 + a3 + a4 + a5 + a6 + a7
    part1 = p_gms[None, :, None] * dx * factor1[:, :, None] / r3[:, :, None]

    factor2 = jnp.sum(
        dx * (4 * t_vel_com[:, None, :] - 3 * p_vel_com[None, :, :]), axis=-1
    )  # (T, P)
    part2 = (
        p_gms[None, :, None]
        * (
            factor2[:, :, None] * dv / r3[:, :, None]
            + 7.0 / 2.0 * a_newt_perturbers[None, :, :] / r[:, :, None]
        )
        / c2
    )

    a_const = jnp.sum(part1 + part2, axis=1)  # (T, 3)

    # Non-constant correction on tracers using pre-computed converged perturber GR
    rdota = jnp.sum(dx * a_gr_perturbers[None, :, :], axis=-1)  # (T, P)
    non_const = jnp.sum(
        (p_gms[None, :, None] / (2.0 * c2))
        * (
            dx * rdota[:, :, None] / r3[:, :, None]
            + 7.0 * a_gr_perturbers[None, :, :] / r[:, :, None]
        ),
        axis=1,
    )  # (T, 3)
    a_final = a_const + non_const

    return a_newt + a_final

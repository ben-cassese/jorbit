import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from copy import deepcopy
import math
import numpy as np
import sys
import rebound


def rebx_calculate_gr_full(
    particles, C2, G, max_iterations=10, gravity_ignore_10=False
):
    N = len(particles)
    # Create a copy of particles to avoid modifying the original
    ps_b = deepcopy(particles)

    # Calculate Newtonian accelerations
    for i in range(N):
        ps_b[i].ax = 0.0
        ps_b[i].ay = 0.0
        ps_b[i].az = 0.0

    # Compute initial Newtonian accelerations
    for i in range(N):
        for j in range(i + 1, N):
            dx = ps_b[i].x - ps_b[j].x
            dy = ps_b[i].y - ps_b[j].y
            dz = ps_b[i].z - ps_b[j].z
            r2 = dx * dx + dy * dy + dz * dz
            r = math.sqrt(r2)
            prefac = G / (r2 * r)

            ps_b[i].ax -= prefac * ps_b[j].m * dx
            ps_b[i].ay -= prefac * ps_b[j].m * dy
            ps_b[i].az -= prefac * ps_b[j].m * dz

            ps_b[j].ax += prefac * ps_b[i].m * dx
            ps_b[j].ay += prefac * ps_b[i].m * dy
            ps_b[j].az += prefac * ps_b[i].m * dz

    # Transform to barycentric coordinates (placeholder - would depend on simulation structure)
    # In the original code, this uses reb_simulation_com and reb_particle_isub
    # You might need to implement this based on your specific simulation framework
    _q = rebound.Simulation()
    _q.add(ps_b)
    _q.move_to_com()
    ps_b = list(_q.particles)
    print(f"mass: {ps_b[0].m}")

    # Compute constant acceleration terms
    a_const = np.zeros((N, 3))

    for i in range(N):
        a_constx, a_consty, a_constz = 0.0, 0.0, 0.0

        for j in range(N):
            if j != i:
                dxij = ps_b[i].x - ps_b[j].x
                dyij = ps_b[i].y - ps_b[j].y
                dzij = ps_b[i].z - ps_b[j].z

                rij2 = dxij * dxij + dyij * dyij + dzij * dzij
                rij = math.sqrt(rij2)
                rij3 = rij2 * rij

                # First constant part calculations
                a1 = sum(
                    (4.0 / C2)
                    * G
                    * particles[k].m
                    / math.sqrt(
                        (ps_b[i].x - ps_b[k].x) ** 2
                        + (ps_b[i].y - ps_b[k].y) ** 2
                        + (ps_b[i].z - ps_b[k].z) ** 2
                    )
                    for k in range(N)
                    if k != i
                )

                a2 = sum(
                    (1.0 / C2)
                    * G
                    * particles[l].m
                    / math.sqrt(
                        (ps_b[l].x - ps_b[j].x) ** 2
                        + (ps_b[l].y - ps_b[j].y) ** 2
                        + (ps_b[l].z - ps_b[j].z) ** 2
                    )
                    for l in range(N)
                    if l != j
                )

                vi2 = ps_b[i].vx ** 2 + ps_b[i].vy ** 2 + ps_b[i].vz ** 2
                a3 = -vi2 / C2

                vj2 = ps_b[j].vx ** 2 + ps_b[j].vy ** 2 + ps_b[j].vz ** 2
                a4 = -2 * vj2 / C2

                a5 = (4 / C2) * (
                    ps_b[i].vx * ps_b[j].vx
                    + ps_b[i].vy * ps_b[j].vy
                    + ps_b[i].vz * ps_b[j].vz
                )

                a6_0 = dxij * ps_b[j].vx + dyij * ps_b[j].vy + dzij * ps_b[j].vz
                a6 = (3 / (2 * C2)) * a6_0 * a6_0 / rij2

                a7 = (dxij * ps_b[j].ax + dyij * ps_b[j].ay + dzij * ps_b[j].az) / (
                    2 * C2
                )

                factor1 = a1 + a2 + a3 + a4 + a5 + a6 + a7
                print(f"a1: {a1}")
                print(f"a2: {a2}")
                print(f"a3: {a3}")
                print(f"a4: {a4}")
                print(f"a5: {a5}")
                print(f"a6: {a6}")
                print(f"a7: {a7}")
                print(f"factor1: {factor1}")

                a_constx += G * particles[j].m * dxij * factor1 / rij3
                a_consty += G * particles[j].m * dyij * factor1 / rij3
                a_constz += G * particles[j].m * dzij * factor1 / rij3

                print(f"part 1: {a_constx}")
                # Second constant part calculations
                dvxij = ps_b[i].vx - ps_b[j].vx
                dvyij = ps_b[i].vy - ps_b[j].vy
                dvzij = ps_b[i].vz - ps_b[j].vz

                factor2 = (
                    dxij * (4 * ps_b[i].vx - 3 * ps_b[j].vx)
                    + dyij * (4 * ps_b[i].vy - 3 * ps_b[j].vy)
                    + dzij * (4 * ps_b[i].vz - 3 * ps_b[j].vz)
                )

                a_constx += (
                    G
                    * particles[j].m
                    / C2
                    * (factor2 * dvxij / rij3 + 7 / 2 * ps_b[j].ax / rij)
                )
                a_consty += (
                    G
                    * particles[j].m
                    / C2
                    * (factor2 * dvyij / rij3 + 7 / 2 * ps_b[j].ay / rij)
                )
                a_constz += (
                    G
                    * particles[j].m
                    / C2
                    * (factor2 * dvzij / rij3 + 7 / 2 * ps_b[j].az / rij)
                )

                z = (
                    G
                    * particles[j].m
                    / C2
                    * (factor2 * dvxij / rij3 + 7 / 2 * ps_b[j].ax / rij)
                )
                print(f"part 2: {z}")

        a_const[i] = [a_constx, a_consty, a_constz]

    # Set initial accelerations to constant terms
    for i in range(N):
        ps_b[i].ax = a_const[i][0]
        ps_b[i].ay = a_const[i][1]
        ps_b[i].az = a_const[i][2]

    print(f"a_const: {a_const[0,0]}")
    # Iterative refinement of accelerations
    for k in range(10):  # Maximum 10 iterations
        # Store old accelerations
        a_old = np.array([[p.ax, p.ay, p.az] for p in ps_b])

        # Add non-constant term
        for i in range(N):
            non_constx, non_consty, non_constz = 0.0, 0.0, 0.0

            for j in range(N):
                if j != i:
                    dxij = ps_b[i].x - ps_b[j].x
                    dyij = ps_b[i].y - ps_b[j].y
                    dzij = ps_b[i].z - ps_b[j].z

                    rij = math.sqrt(dxij * dxij + dyij * dyij + dzij * dzij)
                    rij3 = rij * rij * rij

                    dotproduct = (
                        dxij * ps_b[j].ax + dyij * ps_b[j].ay + dzij * ps_b[j].az
                    )

                    non_constx += (G * particles[j].m * dxij / rij3) * dotproduct / (
                        2 * C2
                    ) + (7 / (2 * C2)) * G * particles[j].m * ps_b[j].ax / rij
                    non_consty += (G * particles[j].m * dyij / rij3) * dotproduct / (
                        2 * C2
                    ) + (7 / (2 * C2)) * G * particles[j].m * ps_b[j].ay / rij
                    non_constz += (G * particles[j].m * dzij / rij3) * dotproduct / (
                        2 * C2
                    ) + (7 / (2 * C2)) * G * particles[j].m * ps_b[j].az / rij

            ps_b[i].ax = a_const[i][0] + non_constx
            ps_b[i].ay = a_const[i][1] + non_consty
            ps_b[i].az = a_const[i][2] + non_constz

        # Check for convergence
        maxdev = 0.0
        for i in range(N):
            dx = (
                abs((ps_b[i].ax - a_old[i][0]) / ps_b[i].ax)
                if abs(ps_b[i].ax) > sys.float_info.epsilon
                else 0
            )
            dy = (
                abs((ps_b[i].ay - a_old[i][1]) / ps_b[i].ay)
                if abs(ps_b[i].ay) > sys.float_info.epsilon
                else 0
            )
            dz = (
                abs((ps_b[i].az - a_old[i][2]) / ps_b[i].az)
                if abs(ps_b[i].az) > sys.float_info.epsilon
                else 0
            )

            maxdev = max(maxdev, dx, dy, dz)

        if maxdev < sys.float_info.epsilon:
            break

        if k == 9:
            print(
                f"Warning: 10 loops in GR calculation did not converge. Fractional Error: {maxdev}"
            )

    # Update original particles with calculated accelerations
    for i in range(N):
        particles[i].ax += ps_b[i].ax
        particles[i].ay += ps_b[i].ay
        particles[i].az += ps_b[i].az

    return particles


def test(particles, C2, G, max_iterations=10, gravity_ignore_10=False):
    N = len(particles)
    # Create a copy of particles to avoid modifying the original
    ps_b = deepcopy(particles)

    # Calculate Newtonian accelerations
    for i in range(N):
        ps_b[i].ax = 0.0
        ps_b[i].ay = 0.0
        ps_b[i].az = 0.0

    # Compute initial Newtonian accelerations
    for i in range(N):
        for j in range(i + 1, N):
            dx = ps_b[i].x - ps_b[j].x
            dy = ps_b[i].y - ps_b[j].y
            dz = ps_b[i].z - ps_b[j].z
            r2 = dx * dx + dy * dy + dz * dz
            r = math.sqrt(r2)
            prefac = G / (r2 * r)

            ps_b[i].ax -= prefac * ps_b[j].m * dx
            ps_b[i].ay -= prefac * ps_b[j].m * dy
            ps_b[i].az -= prefac * ps_b[j].m * dz

            ps_b[j].ax += prefac * ps_b[i].m * dx
            ps_b[j].ay += prefac * ps_b[i].m * dy
            ps_b[j].az += prefac * ps_b[i].m * dz

    # Transform to barycentric coordinates (placeholder - would depend on simulation structure)
    # In the original code, this uses reb_simulation_com and reb_particle_isub
    # You might need to implement this based on your specific simulation framework
    _q = rebound.Simulation()
    _q.add(ps_b)
    _q.move_to_com()
    ps_b = list(_q.particles)
    print(f"mass: {ps_b[0].m}")

    # Compute constant acceleration terms
    a_const = np.zeros((N, 3))

    # Extract attributes into arrays for vectorized computation
    ps_b_x = jnp.array([p.x for p in ps_b])
    ps_b_y = jnp.array([p.y for p in ps_b])
    ps_b_z = jnp.array([p.z for p in ps_b])
    ps_b_vx = jnp.array([p.vx for p in ps_b])
    ps_b_vy = jnp.array([p.vy for p in ps_b])
    ps_b_vz = jnp.array([p.vz for p in ps_b])
    ps_b_ax = jnp.array([p.ax for p in ps_b])
    ps_b_ay = jnp.array([p.ay for p in ps_b])
    ps_b_az = jnp.array([p.az for p in ps_b])
    particles_m = jnp.array([p.m for p in particles])

    # Compute pairwise distances
    dx = ps_b_x[:, None] - ps_b_x[None, :]
    dy = ps_b_y[:, None] - ps_b_y[None, :]
    dz = ps_b_z[:, None] - ps_b_z[None, :]
    rij2 = dx**2 + dy**2 + dz**2
    rij = jnp.sqrt(rij2)
    rij3 = rij2 * rij

    # Mask to exclude self-interactions
    mask = ~jnp.eye(N, dtype=bool)

    # Compute pairwise interactions
    def compute_constants(i):
        # a1
        a1 = jnp.sum(
            (4.0 / C2)
            * G
            * particles_m
            / jnp.sqrt(
                (ps_b_x[i] - ps_b_x) ** 2
                + (ps_b_y[i] - ps_b_y) ** 2
                + (ps_b_z[i] - ps_b_z) ** 2
            )
            * mask[i]
        )

        # a2
        a2 = jnp.sum(
            (1.0 / C2)
            * G
            * particles_m
            / jnp.sqrt(
                (ps_b_x - ps_b_x[:, None]) ** 2
                + (ps_b_y - ps_b_y[:, None]) ** 2
                + (ps_b_z - ps_b_z[:, None]) ** 2
            )
            * mask[i]
        )

        # Velocity terms
        vi2 = ps_b_vx[i] ** 2 + ps_b_vy[i] ** 2 + ps_b_vz[i] ** 2
        a3 = -vi2 / C2

        vj2 = ps_b_vx**2 + ps_b_vy**2 + ps_b_vz**2
        a4 = -2 * vj2 / C2

        a5 = (4 / C2) * (
            ps_b_vx[i] * ps_b_vx + ps_b_vy[i] * ps_b_vy + ps_b_vz[i] * ps_b_vz
        )

        a6_0 = dx[i] * ps_b_vx + dy[i] * ps_b_vy + dz[i] * ps_b_vz
        a6 = (3 / (2 * C2)) * a6_0**2 / rij2[i]

        a7 = (dx[i] * ps_b_ax + dy[i] * ps_b_ay + dz[i] * ps_b_az) / (2 * C2)

        factor1 = a1 + a2 + a3 + a4 + a5 + a6 + a7

        jax.debug.print("a1: {}", a1)
        jax.debug.print("a2: {}", a2)
        jax.debug.print("a3: {}", a3)
        jax.debug.print("a4: {}", a4)
        jax.debug.print("a5: {}", a5)
        jax.debug.print("a6: {}", a6)
        jax.debug.print("a7: {}", a7)
        jax.debug.print("factor1: {}", factor1)

        return factor1

    # Vectorized computation across all indices
    results = jax.vmap(compute_constants)(jnp.arange(N))


def test2(particles, C2, G, max_iterations=10, gravity_ignore_10=False):
    N = len(particles)
    # Create a copy of particles to avoid modifying the original
    ps_b = deepcopy(particles)

    # Calculate Newtonian accelerations
    for i in range(N):
        ps_b[i].ax = 0.0
        ps_b[i].ay = 0.0
        ps_b[i].az = 0.0

    # Compute initial Newtonian accelerations
    for i in range(N):
        for j in range(i + 1, N):
            dx = ps_b[i].x - ps_b[j].x
            dy = ps_b[i].y - ps_b[j].y
            dz = ps_b[i].z - ps_b[j].z
            r2 = dx * dx + dy * dy + dz * dz
            r = math.sqrt(r2)
            prefac = G / (r2 * r)

            ps_b[i].ax -= prefac * ps_b[j].m * dx
            ps_b[i].ay -= prefac * ps_b[j].m * dy
            ps_b[i].az -= prefac * ps_b[j].m * dz

            ps_b[j].ax += prefac * ps_b[i].m * dx
            ps_b[j].ay += prefac * ps_b[i].m * dy
            ps_b[j].az += prefac * ps_b[i].m * dz

    # Transform to barycentric coordinates (placeholder - would depend on simulation structure)
    # In the original code, this uses reb_simulation_com and reb_particle_isub
    # You might need to implement this based on your specific simulation framework
    _q = rebound.Simulation()
    _q.add(ps_b)
    _q.move_to_com()
    ps_b = list(_q.particles)

    # Compute constant acceleration terms
    a_const = np.zeros((N, 3))

    def compute_constant_acceleration(ps_b, G, C2):
        # Extract attributes into JAX arrays
        N = len(ps_b)

        x = jnp.array([p.x for p in ps_b])
        y = jnp.array([p.y for p in ps_b])
        z = jnp.array([p.z for p in ps_b])

        vx = jnp.array([p.vx for p in ps_b])
        vy = jnp.array([p.vy for p in ps_b])
        vz = jnp.array([p.vz for p in ps_b])

        ax = jnp.array([p.ax for p in ps_b])
        ay = jnp.array([p.ay for p in ps_b])
        az = jnp.array([p.az for p in ps_b])

        masses = jnp.array([p.m for p in ps_b])

        def compute_particle_acceleration(i: int):
            def compute_pairwise_acceleration(j: int):
                # Zero out self-interactions
                def zero_if_self(x):
                    return jnp.where(i == j, 0.0, x)

                # Relative distances
                dxij = x[i] - x[j]
                dyij = y[i] - y[j]
                dzij = z[i] - z[j]

                # Compute a1 (sum over all particles except self)
                a1 = zero_if_self(
                    (4.0 / C2)
                    * G
                    * masses[j]
                    / jnp.sqrt(
                        (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 + (z[i] - z[j]) ** 2
                    )
                )

                # Compute a2 (sum over all particles except the current interaction)
                a2 = jnp.sum(
                    jnp.where(
                        (jnp.arange(N) != i) & (jnp.arange(N) != j),
                        (1.0 / C2)
                        * G
                        * masses
                        / jnp.sqrt((x[j] - x) ** 2 + (y[j] - y) ** 2 + (z[j] - z) ** 2),
                        0.0,
                    )
                )

                # Velocity and acceleration terms
                vi2 = vx[i] ** 2 + vy[i] ** 2 + vz[i] ** 2
                vj2 = vx[j] ** 2 + vy[j] ** 2 + vz[j] ** 2

                a3 = -vi2 / C2
                a4 = -2 * vj2 / C2
                a5 = (4 / C2) * (vx[i] * vx[j] + vy[i] * vy[j] + vz[i] * vz[j])

                a6_0 = dxij * vx[j] + dyij * vy[j] + dzij * vz[j]
                a6 = (3 / (2 * C2)) * a6_0 * a6_0 / (dxij**2 + dyij**2 + dzij**2)

                a7 = (dxij * ax[j] + dyij * ay[j] + dzij * az[j]) / (2 * C2)

                # Compute factor1
                factor1 = a1 + a2 + a3 + a4 + a5 + a6 + a7
                jax.debug.print("a5: {x}", x=a5)

                # First part of constant acceleration
                rij2_ij = dxij**2 + dyij**2 + dzij**2
                rij_ij = jnp.sqrt(rij2_ij)
                rij3_ij = rij2_ij * rij_ij

                # Acceleration contributions
                a_const_x = zero_if_self(G * masses[j] * dxij * factor1 / rij3_ij)
                a_const_y = zero_if_self(G * masses[j] * dyij * factor1 / rij3_ij)
                a_const_z = zero_if_self(G * masses[j] * dzij * factor1 / rij3_ij)

                # Second part
                dvxij = vx[i] - vx[j]
                dvyij = vy[i] - vy[j]
                dvzij = vz[i] - vz[j]

                factor2 = (
                    dxij * (4 * vx[i] - 3 * vx[j])
                    + dyij * (4 * vy[i] - 3 * vy[j])
                    + dzij * (4 * vz[i] - 3 * vz[j])
                )

                a_const_x += zero_if_self(
                    G
                    * masses[j]
                    / C2
                    * (factor2 * dvxij / rij3_ij + 7 / 2 * ax[j] / rij_ij)
                )
                a_const_y += zero_if_self(
                    G
                    * masses[j]
                    / C2
                    * (factor2 * dvyij / rij3_ij + 7 / 2 * ay[j] / rij_ij)
                )
                a_const_z += zero_if_self(
                    G
                    * masses[j]
                    / C2
                    * (factor2 * dvzij / rij3_ij + 7 / 2 * az[j] / rij_ij)
                )

                return jnp.array([a_const_x, a_const_y, a_const_z])

            # Compute pairwise interactions for this particle
            pairwise_accel = jax.vmap(compute_pairwise_acceleration)(jnp.arange(N))

            # Sum up all pairwise contributions
            return pairwise_accel.sum(axis=0)

        # Vectorize the computation for all particles
        a_const = jax.vmap(compute_particle_acceleration)(jnp.arange(N))

        return a_const

    a_const = compute_constant_acceleration(ps_b, G, C2)
    print(f"a_const: {a_const[0,0]}")
    return a_const

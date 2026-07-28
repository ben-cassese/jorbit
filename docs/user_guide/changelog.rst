Changelog
+++++++++

**1.5.1 (07/2026)**

- **Behavior change: `System` now actually honors `de_ephemeris_version` in its force model.** `System` accepted, documented, and stored the flag, and used it for observer positions in `System.ephemeris`, but `System._setup_acceleration_func` constructed all five built-in `Ephemeris` objects without passing it. Since `Ephemeris` defaults to `"440"`, every `System` integrated under DE440 regardless of what was requested — silently, and with no warning. The result was physically inconsistent: a `System` built with `de_ephemeris_version="430"` produced DE430 observer positions against DE440 dynamics, and because the discrepancy grows slowly from zero it was easy to mistake for ordinary integration error. `System` also now validates the value the way `Particle` always has, so an unsupported version raises instead of falling through to DE440. **If you have been passing `de_ephemeris_version="430"` to `System`, your results will change** — they were DE440 before and are DE430 now. `Particle` was never affected.
- **Behavior change: `jorbit.utils.mpc.unpack_epoch` now returns TT, not UTC.** MPCORB epochs are defined in TT — the MPC's *Export Format for Minor-Planet Orbits* specifies columns 21-25 as "Epoch (in packed form, .0 TT)" — but the decoded `Time` was tagged `scale="utc"`. Every epoch decoded from MPCORB was therefore mis-placed by TT-UTC (69.184 s as of 2026), a pure systematic worth roughly 1300 km along-track (~1.2 arcsec) for a main-belt object and 1960 km (~27 arcsec) for an NEO — far above jorbit's ~1 mas accuracy target. Times returned by this function now shift by 69.184 s relative to 1.5.0. Nothing inside jorbit called `unpack_epoch`, so no other behavior is affected; note that the observation dates parsed by `read_mpc_file` are correctly UTC, since the 80-column observation format differs from MPCORB in exactly this respect.
- `Particle.is_observable` built its Sun-position `Ephemeris` without `de_ephemeris_version`, so it mixed a DE430 observer with a DE440 Sun when the particle was set to DE430. Same class of bug as the `System` one above, much smaller amplitude (solar elongation tolerates a few km of Sun position easily), now consistent.
- Documented a known limitation that was previously silent: `Particle.static_residuals` always uses the DE440 "default solar system" dynamical model, regardless of the `gravity` and `de_ephemeris_version` passed to `Particle`. This is because `create_static_default_acceleration_func` is currently the only static acceleration function in `jorbit.accelerations`; there is no static counterpart to "newtonian planets", "gr planets", etc. Behavior is unchanged — the limitation is now stated in the `Particle` docstrings, in `precompute_likelihood_data`, and in the MCMC tutorial. Use `Particle.residuals` if you need the model you asked for.
- Fixed `jorbit.__version__`, which still read `"1.4.2"` when 1.5.0 shipped. The `__init__.py` and `pyproject.toml` version literals are maintained independently, so a test now asserts they agree.

**1.5.0 (07/2026)**

- **Major memory fix: JIT compilations no longer embed the JPL ephemeris (and other large arrays) as per-compilation constants.** The acceleration-function factories in `jorbit.accelerations`, the per-`Particle` `residuals`/`loglike` callables, the per-`System` forward-model callables, and the static-residuals pipeline all previously captured large arrays (most notably the ephemeris Chebyshev coefficients) in Python closures. JAX bakes any concrete array touched during tracing into the compiled executable as a constant, so every distinct compilation retained its own ~150 MB copy for the life of the process — enough to OOM CI runners (and any long-lived session creating many `Particle`/`System` objects). All such data now flows through `jax.tree_util.Partial`-bound arguments into shared, module-level jitted functions: buffers are shared process-wide and instances with matching shapes reuse compilations. Together with a new per-module JIT-cache-clearing test fixture, peak test-suite memory drops from ~19 GB to ~5.5 GB, and first-call (compile) times for the likelihood/gradient paths roughly halve; steady-state runtime of likelihoods and gradients is unchanged (verified with back-to-back benchmarks of `loglike`, `residuals`, `static_residuals`, their gradients, and the `System` forward model). Note one small behavioral difference: `Particle.residuals`/`Particle.loglike` and the `System` forward-model callables are no longer pre-wrapped in an extra `jax.jit` (their internals are already jitted); wrapping them in your own `jax.jit` still works, but re-introduces the constant-embedding for that one compilation.
- Fixed `EphemerisPostProcessor.tree_unflatten` running `jnp.concatenate` on its children during unflattening, which crashed when JAX rebuilds the pytree with placeholder leaves (e.g. under `jax.vmap` with `in_axes=None`). `log_gms` is now stored as a pytree child and unflattening is purely structural.
- Added protections for integrations/ephemerides generated via the front end `Particle` and `System` classes no longer silently truncate when an integration requires more steps than allocated in 1) the cap of 10k steps between time steps and 2) the cap of 15k total accepted steps when using "interpolation". The actual caps remain in the back end implementations.
- Fixed a bug in the `Particle.integrate_or_interpolate` method that would produce garbage results on very long backwards integrations.

**1.4.0 (05/2026)**

- Added `uncertainty=True` as an option to `Particle.ephemeris` that returns the propagated on-sky covariance in addition to the propagated state. This is implemented by computing the Jacobian of the dynamics using forward autodiff, then linear error propagation using the initial covariance on the `CartesianState` or `KeplerianState`.
- Several misc bug fixes found by Claude's recent code audit, including some precision loss issues when combining `Observation` objects and array shape mismatches.

**1.3.0 (05/2026)**

- Instead of taking an IAS15 step to the exact observation times, the `static_step` now takes natural IAS15 adaptive steps and evaluates the relevant polynomials based on the b coefficients to compute positions/velocities at arbitrary times within a step. This cuts down the number of IAS15 steps substatially when observations are spaced within a typical maximum-allowable IAS15 step, or ~30 days. This involves lots of changes to the internal `ias_15_evolve` function and the creation of some new internal helpers like `ias15_evolve_with_dense_output`, `ias15_evolve_forced_landing` and `interpolate_from_dense_output`.
- Added option for Keplerian-only motion to `Particle` and `System`. This is significantly faster than the full N-body simulations and is useful for both short-term motion and outer solar system applications where the perturbations are small.
- Add the `.is_observable` method to `Particle` objects which checks whether the particle is at least a certain angular distance from the Sun at given times from given observatories.
- Modified `on_sky` to take optional `ltt_position_fn` instead of always defaulting to the Taylor expansion. Can be used to "interpolate" using the above-mentioned b coefficients.
- Changed the internal representations being absolute and relative time. No external API changes, but now `Particle` and `System` objects work in relative times by default, only converting to absolute times when necessary for ephemeris queries. This preserves some precision by not storing all the extra digits in mjd/jd.
- Add the "global" IAS15 step size controller in addition to the default "PRS23" controller.
- Misc bug fixes and code cleanups.

**1.2.0 (02/2026)**

- Major refactor of ias15.py, no longer resembles the original REBOUND implementation. Removed IAS15Helper, but otherwise no API changes.
- Major refactor of ppn_gravity, the outputs are unchanged but now unnecessary perturber-perturber interactions are excluded.
- Added a "static" version of IAS15 that uses fixed step sizes and a fixed number of predictor-corrector iteratins. Also added the ability to pre-compute perturber positions/velocities/ppn-specific acceleration terms at every (sub)time step prior to integrating. These features are intended to speed up likelihood evaluations in orbit fitting applications where we expect only small changes to the initial conditions of a test particle. This dramatically accelerates both likelihood and gradient evaluations. The "integrate", "loglike", and "ephemeris" attributes of Particle remain unchanged for now, but a new method called "static_residuals" has been added that uses these new tools.
- Addition of `fixed_perturber_positions`, `fixed_perturber_velocities`, and `fixed_perturber_log_gms` to `SystemState`. These quantities can be used in different acceleration functions to indicate that perturber-perturber interactions should be ignored and that gradients with respect to these quantities don't need to be tracked.
- Added option to select DE ephemeris version (DE430 or DE440) when creating `Ephemeris` and `Particle` objects. Updated relevant tutorials/tests.
- Dropped support for Python 3.9.

**1.1.0 (01/2026)**

- Added general N-body integration functionality and tutorial.
- Added helpers to convert between heliocentric and barycentric frames, and tutorial.
- Added 4th, 6th, and 8th order symplectic integrators from [Yoshida 1990](https://www.doi.org/10.1016/0375-9601(90)90092-3). Added new tutorial on picking integrators and dynamical models, and edited the orbit fitting tutorial to use the new integrators.

**1.0.0 (07/2025)**

- Paper release

**0.2.0 (03/2025)**

- Initial release!

Changelog
+++++++++

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

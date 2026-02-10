Changelog
+++++++++

**development version**

- Added option to select DE ephemeris version (DE430 or DE440) when creating `Ephemeris` and `Particle` objects. Updated relevant tutorials/tests.
- Major refactor of ias15.py, no longer resembles the original REBOUND implementation. Removed IAS15Helper, but otherwise no API changes.
- Addition of `fixed_perturber_positions`, `fixed_perturber_velocities`, and `fixed_perturber_log_gms` to `SystemState`. These quantities can be used in different acceleration functions to indicate that perturber-perturber interactions should be ignored and that gradients with respect to these quantities don't need to be tracked.
- Major refactor of ppn_gravity, the outputs are unchanged but now unnecessary perturber-perturber interactions are excluded.

**1.1.0 (01/2026)**

- Added general N-body integration functionality and tutorial.
- Added helpers to convert between heliocentric and barycentric frames, and tutorial.
- Added 4th, 6th, and 8th order symplectic integrators from [Yoshida 1990](https://www.doi.org/10.1016/0375-9601(90)90092-3). Added new tutorial on picking integrators and dynamical models, and edited the orbit fitting tutorial to use the new integrators.

**1.0.0 (07/2025)**

- Paper release

**0.2.0 (03/2025)**

- Initial release!

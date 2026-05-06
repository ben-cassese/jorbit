# REBOUND IAS15 instrumentation patch

`integrator_ias15.c.patch` is a unified diff against the unmodified
REBOUND `4.5.1` release tag's `src/integrator_ias15.c`. It adds
diagnostic dumps to `reb_integrator_ias15_step` so that every IAS15
predictor-corrector intermediate can be captured from C-side stdout.

The full instrumented file is also included as `integrator_ias15.c`
for emergency restoration if the patch fails to apply.

## What the patch adds

All dumps are gated by env var `IAS15_DUMP_PC=1`. The check is performed
once per process via a function-scope static; when unset, there is zero
overhead in the integrator's hot loop.

Two static helpers are added near `add_cs`:
- `_dump_arr` — emits one line of `key=v0,v1,…,vN` with `%.17e` formatting
  (full double-precision round-trip).
- `_dump_dp7` — convenience wrapper for the `struct reb_dp7` 7-component
  IAS15 b/g/csb/e arrays.

Four dump kinds are emitted, all prefixed with a tag in square brackets
so they can be parsed line-by-line:

### `[IAS15_INIT]` — one per accepted step

Emitted just before the `while(1)` predictor-corrector loop begins.
Captures the full step-start state.

Fields:
- `N3` (int) — `3 * sim.N`
- `dt`, `t_beginning`
- `x0[N3]`, `v0[N3]`, `a0[N3]` — barycentric particle state at step start
- `csx[N3]`, `csv[N3]`, `csa0[N3]` — REBOUND's compensated-summation accumulators
- `b{0..6}[N3]`, `csb{0..6}[N3]`, `g{0..6}[N3]` — IAS15 polynomial state at start

### `[IAS15_SUB]` — one per (PC iteration × substep n=1..7)

Emitted inside the substep loop immediately after the `switch(n)` block
mutates b/g. Captures both the predicted state the force model just saw
(*before* this substep's b/g update) and the post-update state.

Fields:
- `iter` (int, 1-indexed), `n` (int, 1-indexed substep)
- `x[N3]` (predicted position used for the force evaluation)
- `v[N3]` (predicted velocity, zero-filled if the velocity-prediction path
  was disabled)
- `at[N3]` (acceleration the force model returned at this substep)
- `g_pre[N3]` (`g.p{n-1}` snapshot *before* the substep updated it)
- `g{0..6}[N3]`, `b{0..6}[N3]`, `csb{0..6}[N3]` — all post-update

### `[IAS15_PC]` — one per PC iteration

Emitted at the bottom of each iteration of the `while(1)` PC loop.

Fields:
- `iter` (int, 1-indexed), `N3`
- `pc_err` — REBOUND's iteration convergence metric
- `b{0..6}[N3]` — converged b at the end of this iteration

### `[IAS15_FINAL]` — one per accepted step

Emitted *after* the PC loop converges and *before* the controller mutates
b for the next-step predictor. The fields here equal the converged
`_br` reachable from Python.

Fields:
- `iter_final` — number of PC iterations the step needed
- `b{0..6}[N3]`, `csb{0..6}[N3]`, `g{0..6}[N3]`, `e{0..6}[N3]`

## Why `4.5.1` specifically

The C struct `reb_simulation` layout differs between REBOUND `4.5.1` and
`4.6.0`. ASSIST's pip-installed shared library is currently built against
`4.5.1`, so a librebound patched on top of any newer tag will be
ABI-incompatible with ASSIST and will segfault at startup.

If you need to use a different REBOUND version, you'll have to reapply
the diagnostic markers by hand (the dump-point locations are stable
across recent versions; the surrounding code is what differs).

## Verifying the patch

```bash
git clone --depth 1 --branch 4.5.1 https://github.com/hannorein/rebound.git /tmp/rebound
cd /tmp/rebound
git apply --check /path/to/rebound_apophis_dumps/rebound_patches/integrator_ias15.c.patch
```

The above should print no output. If it errors with "patch does not
apply", the upstream `4.5.1` source has shifted (unlikely for a tagged
release) — fall back to copying `integrator_ias15.c` from this directory
over `/tmp/rebound/src/integrator_ias15.c` directly.

## Building

```bash
cd /tmp/rebound/src && make
```

Default OPT (no `-O3`, includes `-g`). REBOUND's pip-installed lib was
built with the same flags via `setup.py`, so the FP semantics are matched
modulo compiler version. Don't change OPT unless you record it — the
single-step sims compare byte-for-byte against the bundle's reference
output, and aggressive optimization is the most likely source of drift.

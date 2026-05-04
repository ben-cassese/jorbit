"""Parse raw IAS15_DUMP_PC stdout into structured NPZ.

The instrumented REBOUND prints four kinds of lines (see
rebound_patches/README_patch.md):
  [IAS15_INIT]  — one per step, step-start state snapshot
  [IAS15_SUB]   — one per (PC iter, substep n=1..7), per-substep state
  [IAS15_PC]    — one per PC iteration, post-iter convergence metric and b
  [IAS15_FINAL] — one per step, converged state pre-controller-mutation

This module reads a captured stdout log and converts it to NPZ. For
multi-step runs (sims 1 and 2), it emits per-step records as variable-length
object arrays. For single-step runs (sims 3 and 4), it emits plain ndarrays
with the same field names that scratch_phase7_dump.py produced.

Usage:
    python parse_dumps.py <input.log> <output.npz>

Or as a library:
    from parse_dumps import parse_log, parse_step
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _parse_kv(line: str) -> dict:
    """Parse one '[TAG] key=val key=v0,v1,...' line."""
    body = line.split("]", 1)[1]
    out: dict[str, object] = {}
    for tok in body.strip().split():
        if "=" not in tok:
            continue
        key, val = tok.split("=", 1)
        if "," in val:
            try:
                out[key] = np.array([float(s) for s in val.split(",")])
                continue
            except ValueError:
                pass
        try:
            out[key] = float(val)
        except ValueError:
            out[key] = val
    return out


def _stack_dp7(rec: dict, prefix: str) -> np.ndarray:
    """Stack {prefix}0..{prefix}6 from a record into shape (7, N3)."""
    return np.stack([rec[f"{prefix}{k}"] for k in range(7)], axis=0)


def parse_step(
    init: dict,
    subs: list[dict],
    pcs: list[dict],
    final: dict,
) -> dict:
    """Build a structured per-step record from one set of dump dicts."""
    N3 = int(init["N3"])
    n_iters = len(pcs)
    if len(subs) != n_iters * 7:
        raise ValueError(
            f"expected {n_iters * 7} SUB lines for {n_iters} PC iters, got {len(subs)}"
        )

    sub_x = np.zeros((n_iters, 7, N3))
    sub_v = np.zeros((n_iters, 7, N3))
    sub_at = np.zeros((n_iters, 7, N3))
    sub_g_pre = np.zeros((n_iters, 7, N3))
    sub_g = np.zeros((n_iters, 7, 7, N3))
    sub_b = np.zeros((n_iters, 7, 7, N3))
    sub_csb = np.zeros((n_iters, 7, 7, N3))
    for s in subs:
        k = int(s["iter"]) - 1
        n = int(s["n"]) - 1
        sub_x[k, n] = s["x"]
        sub_v[k, n] = s["v"]
        sub_at[k, n] = s["at"]
        sub_g_pre[k, n] = s["g_pre"]
        sub_g[k, n] = _stack_dp7(s, "g")
        sub_b[k, n] = _stack_dp7(s, "b")
        sub_csb[k, n] = _stack_dp7(s, "csb")

    return {
        "N3": N3,
        "n_iters": n_iters,
        "dt": float(init["dt"]),
        "t_beginning": float(init["t_beginning"]),
        "init_x0": init["x0"],
        "init_v0": init["v0"],
        "init_a0": init["a0"],
        "init_csx": init["csx"],
        "init_csv": init["csv"],
        "init_csa0": init["csa0"],
        "init_b": _stack_dp7(init, "b"),
        "init_csb": _stack_dp7(init, "csb"),
        "init_g": _stack_dp7(init, "g"),
        "sub_x": sub_x,
        "sub_v": sub_v,
        "sub_at": sub_at,
        "sub_g_pre": sub_g_pre,
        "sub_g": sub_g,
        "sub_b": sub_b,
        "sub_csb": sub_csb,
        "pc_err": np.array([p["pc_err"] for p in pcs]),
        "final_b": _stack_dp7(final, "b"),
        "final_csb": _stack_dp7(final, "csb"),
        "final_g": _stack_dp7(final, "g"),
        "final_e": _stack_dp7(final, "e"),
        "final_iter": int(final["iter_final"]),
    }


def parse_log(log_path: str | Path) -> list[dict]:
    """Read a captured stdout log and return one record per step."""
    log_path = Path(log_path)
    text = log_path.read_text()

    steps: list[dict] = []
    cur_init: dict | None = None
    cur_subs: list[dict] = []
    cur_pcs: list[dict] = []

    for line in text.splitlines():
        if line.startswith("[IAS15_INIT]"):
            if cur_init is not None:
                raise ValueError(
                    "two [IAS15_INIT] lines without intervening [IAS15_FINAL]"
                )
            cur_init = _parse_kv(line)
            cur_subs, cur_pcs = [], []
        elif line.startswith("[IAS15_SUB]"):
            cur_subs.append(_parse_kv(line))
        elif line.startswith("[IAS15_PC]"):
            cur_pcs.append(_parse_kv(line))
        elif line.startswith("[IAS15_FINAL]"):
            if cur_init is None:
                raise ValueError("[IAS15_FINAL] without preceding [IAS15_INIT]")
            steps.append(parse_step(cur_init, cur_subs, cur_pcs, _parse_kv(line)))
            cur_init = None
            cur_subs, cur_pcs = [], []

    return steps


def save_npz(steps: list[dict], out_path: str | Path) -> None:
    """Save steps to NPZ. Single-step → flat arrays. Multi-step → object arrays."""
    out_path = Path(out_path)
    if len(steps) == 0:
        raise ValueError("no steps parsed")

    if len(steps) == 1:
        # Flat ndarray fields, matches scratch_phase7_dump.py schema
        np.savez_compressed(out_path, **steps[0])
    else:
        # Object arrays for variable-length sequence
        keys = list(steps[0].keys())
        flat = {}
        # Scalar fields kept as 1D ndarray
        scalar_keys = {"N3", "n_iters", "dt", "t_beginning", "final_iter"}
        for k in keys:
            if k in scalar_keys:
                flat[k] = np.array([s[k] for s in steps])
            else:
                flat[k] = np.array([s[k] for s in steps], dtype=object)
        flat["n_steps"] = np.array(len(steps))
        np.savez_compressed(out_path, **flat)


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: python parse_dumps.py <input.log> <output.npz>", file=sys.stderr)
        sys.exit(2)
    in_path, out_path = sys.argv[1], sys.argv[2]
    steps = parse_log(in_path)
    save_npz(steps, out_path)
    print(f"parsed {len(steps)} step(s) from {in_path} -> {out_path}")
    if len(steps) >= 1:
        s0 = steps[0]
        print(f"  step[0]: dt={s0['dt']:.6g}  n_iters={s0['n_iters']}  N3={s0['N3']}")
    if len(steps) >= 2:
        sN = steps[-1]
        print(f"  step[{len(steps)-1}]: dt={sN['dt']:.6g}  n_iters={sN['n_iters']}")


if __name__ == "__main__":
    main()

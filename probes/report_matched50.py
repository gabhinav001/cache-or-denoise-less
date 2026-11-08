#!/usr/bin/env python3
"""Matched 50-state evaluation: whole-stack sweep, blockwise hooks, and both checkpoints.

Analyzes:
  1. idm_whole_agg_tau8: Re-timed whole-stack preaggregated attenuation (aggregation in timer)
  2. idm_block_skip_tau8: Explicit block skipping (c3ache:8:sx) under conditional IDM
  3. uncond_stock_2step: Short uncached stock (2 steps) under native unconditional Fast-WAM (shift 5.0)
  4. uncond_stock_10step: Full uncached stock (10 steps) under native unconditional Fast-WAM (shift 5.0)
  5. uncond_whole_tau8: Native whole-stack caching (10 steps, tau=8) under native unconditional Fast-WAM (shift 5.0)

Reports success rates, cycle-level latencies (mean, sd, refresh, reuse, p50, p95),
and exact McNemar tests on matched initial states.
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics as st
import sys
from pathlib import Path


def pct(v, q):
    if not v:
        return float("nan")
    s = sorted(v)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def exact_mcnemar(b, c):
    """Two-sided exact McNemar test p-value from discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * p)


def paired_bootstrap_ci(a1_succ, a2_succ, shared_keys, n_resamples=10000, seed=42):
    """Percentile bootstrap 95% CI for paired success difference (a1 - a2) in points."""
    import random
    rng = random.Random(seed)
    diffs = [100.0 * (1 if a1_succ[k] else 0) - 100.0 * (1 if a2_succ[k] else 0) for k in shared_keys]
    n = len(diffs)
    if n == 0:
        return (float("nan"), float("nan"))
    means = []
    for _ in range(n_resamples):
        sample = [rng.choice(diffs) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    low = means[int(round(0.025 * (n_resamples - 1)))]
    high = means[int(round(0.975 * (n_resamples - 1)))]
    return (low, high)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    files = [p for p in sorted(glob.glob(os.path.join(d, "*.json")))
             if not os.path.basename(p).startswith(("gate", "_"))]
    if not files:
        print(f"  no evaluation artefacts in {d} -- not yet collected.")
        return 0

    arms = {}
    loads = []
    gpu_names = set()
    ref_keys = None
    ref_state_hashes = None

    for p in files:
        stem = Path(p).stem
        try:
            j = json.load(open(p))
        except Exception as e:
            print(f"  [warn] failed to load {p}: {e}")
            continue
        m = j.get("meta") or {}
        pv = m.get("provenance") or {}
        if "gpu" in pv:
            gpu_names.add(pv["gpu"])
        la = (pv.get("loadavg_1_5_15") or [0])[0]
        nc = pv.get("cpu_count") or 1
        loads.append((la, nc))

        rows = j.get("rows", [])
        if not rows:
            continue

        seen_keys = set()
        state_hashes = {}
        for r in rows:
            k = (r["task"], r["episode"])
            if k in seen_keys:
                raise ValueError(f"Duplicate (task, episode) key {k} in {p}")
            seen_keys.add(k)
            state_hashes[k] = r.get("init_state")

        if ref_keys is None:
            ref_keys = sorted(seen_keys)
            ref_state_hashes = state_hashes
        else:
            if sorted(seen_keys) != ref_keys:
                raise ValueError(f"Task/episode key mismatch in {p} vs reference")
            if state_hashes != ref_state_hashes:
                raise ValueError(f"State hash mismatch in {p} vs reference")

        keyed = {(r["task"], r["episode"]): r for r in rows}
        successes = {k: r["success"] for k, r in keyed.items()}
        timings = [r["ms_per_cycle"] for r in rows if "ms_per_cycle" in r and not r.get("warmup")]
        refresh_timings = []
        reuse_timings = []
        all_cycle_timings = []

        for r in rows:
            if r.get("warmup"):
                continue
            cyc_ms = r.get("cycle_ms", [])
            cyc_ref = r.get("cycle_refresh", [])
            for ms, ref in zip(cyc_ms, cyc_ref):
                all_cycle_timings.append(ms)
                if ref:
                    refresh_timings.append(ms)
                else:
                    reuse_timings.append(ms)

        mean_ms = st.mean(timings) if timings else float("nan")
        sd_ms = st.stdev(timings) if len(timings) > 1 else 0.0
        n_succ = sum(successes.values())
        n_tot = len(successes)
        sr = 100.0 * n_succ / n_tot if n_tot else 0.0

        arms[stem] = {
            "file": p,
            "stem": stem,
            "steps": m.get("num_inference_steps"),
            "config": m.get("configs", [""])[0],
            "n_succ": n_succ,
            "n_tot": n_tot,
            "sr": sr,
            "mean_ms": mean_ms,
            "sd_ms": sd_ms,
            "refresh_ms": st.mean(refresh_timings) if refresh_timings else float("nan"),
            "reuse_ms": st.mean(reuse_timings) if reuse_timings else float("nan"),
            "p50_ms": pct(all_cycle_timings, 0.50),
            "p95_ms": pct(all_cycle_timings, 0.95),
            "keyed_success": successes,
        }

    hi = max(la / nc for la, nc in loads) if loads else 0
    print(f"\n================================================================================")
    print(f"  MATCHED 50-STATE EVALUATION REPORT")
    print(f"  GPU: {', '.join(sorted(gpu_names)) or 'unknown'} | Max load per core: {hi:.2f}")
    if hi > 0.5:
        print(f"  WARNING: Contended host (>0.5 load/core); wall-clock timing subject to noise.")
    print(f"================================================================================\n")

    print(f"  {'Arm':26} {'Steps':>5} {'Success (%)':>14} {'Mean (ms)':>10} {'SD':>6} {'Refresh':>9} {'Reuse':>9} {'p95':>8}")
    print(f"  {'-'*26} {'-'*5} {'-'*14} {'-'*10} {'-'*6} {'-'*9} {'-'*9} {'-'*8}")

    display_order = [
        "idm_stock_1step",
        "idm_stock_2step",
        "idm_stock_4step",
        "idm_stock_10step",
        "idm_whole_plain_tau0",
        "idm_whole_plain_tau2",
        "idm_whole_plain_tau4",
        "idm_whole_plain_tau8",
        "idm_whole_agg_tau4",
        "idm_whole_agg_tau8",
        "idm_whole_4step_tau4",
        "idm_block_plain_tau8",
        "idm_block_skip_tau8",
        "uncond_stock_1step",
        "uncond_stock_2step",
        "uncond_stock_4step",
        "uncond_stock_10step",
        "uncond_whole_tau0",
        "uncond_whole_tau2",
        "uncond_whole_tau4",
        "uncond_whole_tau8",
        "uncond_whole_4step_tau4",
    ]
    sorted_keys = [k for k in display_order if k in arms] + [k for k in arms if k not in display_order]

    for k in sorted_keys:
        a = arms[k]
        succ_str = f"{a['sr']:.1f}% ({a['n_succ']}/{a['n_tot']})"
        ref_str = f"{a['refresh_ms']:.1f}" if math.isfinite(a['refresh_ms']) else "-"
        reu_str = f"{a['reuse_ms']:.1f}" if math.isfinite(a['reuse_ms']) else "-"
        p95_str = f"{a['p95_ms']:.1f}" if math.isfinite(a['p95_ms']) else "-"
        print(f"  {a['stem']:26} {a['steps']:5} {succ_str:>14} {a['mean_ms']:10.1f} {a['sd_ms']:6.1f} {ref_str:>9} {reu_str:>9} {p95_str:>8}")

    print(f"\n--------------------------------------------------------------------------------")
    print(f"  KEY CONTRASTS ON MATCHED INITIAL STATES")
    print(f"--------------------------------------------------------------------------------")

    pairs = [
        ("idm_stock_1step", "idm_stock_2step",
         "1-step vs 2-step uncached (Optional-IDM, s=1.0)"),
        ("idm_stock_2step", "idm_stock_10step",
         "2-step vs 10-step full compute (Conditional IDM, s=1.0)"),
        ("idm_stock_2step", "idm_whole_plain_tau0",
         "IDM: 2-step uncached vs whole-stack plain tau=0 (initial refresh only)"),
        ("idm_stock_2step", "idm_whole_plain_tau2",
         "IDM: 2-step uncached vs whole-stack plain tau=2"),
        ("idm_stock_2step", "idm_whole_plain_tau4",
         "IDM: 2-step uncached vs whole-stack plain tau=4"),
        ("idm_stock_2step", "idm_whole_plain_tau8",
         "IDM: 2-step uncached vs whole-stack plain tau=8"),
        ("idm_whole_agg_tau4", "idm_whole_plain_tau4",
         "IDM: Whole-stack preagg attenuation vs plain caching (tau=4, s=1.0)"),
        ("idm_whole_agg_tau8", "idm_whole_plain_tau8",
         "IDM: Whole-stack preagg attenuation vs plain caching (tau=8, s=1.0)"),
        ("idm_stock_2step", "idm_whole_agg_tau4",
         "IDM: 2-step uncached vs Whole-stack preagg attenuation (tau=4, s=1.0)"),
        ("idm_stock_2step", "idm_whole_agg_tau8",
         "IDM: 2-step uncached vs Whole-stack preagg attenuation (tau=8, s=1.0)"),
        ("idm_stock_4step", "idm_whole_4step_tau4",
         "IDM: 4-step short schedule uncached vs whole-stack cached (tau=4)"),
        ("idm_whole_agg_tau8", "idm_block_skip_tau8",
         "Whole-stack substitution vs blockwise skipping (hook placement)"),
        ("idm_block_skip_tau8", "idm_block_plain_tau8",
         "Blockwise explicit skip attenuation vs plain caching (tau=8, s=1.0)"),
        ("uncond_stock_1step", "uncond_stock_2step",
         "1-step vs 2-step uncached (original checkpoint, s=5.0)"),
        ("uncond_stock_2step", "uncond_stock_10step",
         "2-step vs 10-step uncached (original checkpoint, s=5.0)"),
        ("uncond_stock_2step", "uncond_whole_tau0",
         "Native: Short uncached (2-step) vs whole-stack cache tau=0 (initial refresh only)"),
        ("uncond_stock_2step", "uncond_whole_tau2",
         "Native: Short uncached (2-step) vs whole-stack cache tau=2"),
        ("uncond_stock_2step", "uncond_whole_tau4",
         "Native: Short uncached (2-step) vs whole-stack cache tau=4"),
        ("uncond_stock_2step", "uncond_whole_tau8",
         "Native: Short uncached (2-step) vs whole-stack cache tau=8"),
        ("uncond_stock_4step", "uncond_whole_4step_tau4",
         "Native: 4-step short schedule uncached vs whole-stack cached (tau=4)"),
    ]

    for a1_key, a2_key, desc in pairs:
        if a1_key not in arms or a2_key not in arms:
            continue
        a1, a2 = arms[a1_key], arms[a2_key]
        shared_keys = sorted(set(a1["keyed_success"]) & set(a2["keyed_success"]))
        if not shared_keys:
            continue
        w1 = sum(1 for k in shared_keys if a1["keyed_success"][k] and not a2["keyed_success"][k])
        w2 = sum(1 for k in shared_keys if a2["keyed_success"][k] and not a1["keyed_success"][k])
        p_val = exact_mcnemar(w1, w2)
        diff_pts = a1["sr"] - a2["sr"]
        ci_low, ci_high = paired_bootstrap_ci(a1["keyed_success"], a2["keyed_success"], shared_keys)
        speedup = a2["mean_ms"] / a1["mean_ms"] if a1["mean_ms"] > 0 else float("nan")

        print(f"\n  Contrast: {desc}")
        print(f"    {a1_key} ({a1['sr']:.1f}%, {a1['mean_ms']:.1f} ms) vs {a2_key} ({a2['sr']:.1f}%, {a2['mean_ms']:.1f} ms)")
        print(f"    Diff: {diff_pts:+.1f} pts [95% CI: {ci_low:+.1f}, {ci_high:+.1f}] | Discordant: {w1}/{w2} | McNemar p={p_val:.4f}")
        print(f"    Speedup: {speedup:.2f}x ({a1_key} relative to {a2_key})")

    print(f"\n================================================================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

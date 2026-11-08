#!/usr/bin/env python3
"""Evaluation report: Same-host matched uncached baselines and composed 2-step caching.

Analyzes 8 arms on 100 matched states (10 tasks x 10 episodes) on a single RTX A6000:
  1. Optional-IDM 1-step uncached
  2. Optional-IDM 2-step uncached
  3. Optional-IDM K=2, tau=2, prefix=1 (composed)
  4. Optional-IDM K=2, tau=4, prefix=1 (composed)
  5. Native/Original 1-step uncached
  6. Native/Original 2-step uncached
  7. Native/Original K=2, tau=2, prefix=1 (composed)
  8. Native/Original K=2, tau=4, prefix=1 (composed)
Also checks gate.json for cache identity verification and memory integrity.
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


def print_contrast(name, arm1, arm2, arms):
    if arm1 not in arms or arm2 not in arms:
        return
    a1 = arms[arm1]
    a2 = arms[arm2]
    shared = [k for k in a1["succ"] if k in a2["succ"]]
    n_shared = len(shared)
    if n_shared == 0:
        return
    s1 = sum(1 for k in shared if a1["succ"][k])
    s2 = sum(1 for k in shared if a2["succ"][k])
    rate1 = 100.0 * s1 / n_shared
    rate2 = 100.0 * s2 / n_shared
    diff = rate1 - rate2

    b = sum(1 for k in shared if a1["succ"][k] and not a2["succ"][k])
    c = sum(1 for k in shared if not a1["succ"][k] and a2["succ"][k])
    p_val = exact_mcnemar(b, c)
    ci_low, ci_high = paired_bootstrap_ci(a1["succ"], a2["succ"], shared)

    speedup = a2["mean"] / a1["mean"] if a1["mean"] > 0 else float("nan")

    print(f"\n  Contrast: {name}")
    print(f"    {arm1} ({rate1:.1f}%, {a1['mean']:.1f} ms) vs {arm2} ({rate2:.1f}%, {a2['mean']:.1f} ms)")
    print(f"    Diff: {diff:+.1f} pts [95% CI: {ci_low:+.1f}, {ci_high:+.1f}] | Discordant: {b}/{c} | McNemar p = {p_val:.4f}")
    print(f"    Speedup: {speedup:.2f}x ({arm1} relative to {arm2})")


def analyze_dir(d_path: Path):
    print("=" * 80)
    print("  MATCHED SAME-HOST EVALUATION: UNCACHED VS COMPOSED 2-STEP CACHING")
    print("=" * 80)

    gate_file = d_path / "gate.json"
    if gate_file.exists():
        gate = json.loads(gate_file.read_text())
        print(f"\n  Cache Identity Gate: {'PASS' if gate.get('pass') else 'FAIL'}")
        print(f"    Identity error: {gate.get('identity_err', float('nan')):.3e} (threshold: 5e-3)")
        print(f"    Hardware: {gate.get('gpu', 'unknown')}")
        print(f"    Memory integrity: {gate.get('cache_integrity')}")
    else:
        print("\n  Cache Identity Gate: [not found]")

    arms = {}
    for p in sorted(d_path.glob("*.json")):
        if p.name == "gate.json":
            continue
        d = json.loads(p.read_text())
        if "rows" not in d:
            continue
        name = p.stem
        rows = d["rows"]
        succ = { (r["task"], r["init_state"]): bool(r["success"]) for r in rows }
        n = len(rows)
        sr = 100.0 * sum(succ.values()) / n if n > 0 else 0.0

        non_warm = [r for r in rows if not r.get("warmup")]
        ms_list = [r["ms_per_cycle"] for r in non_warm if "ms_per_cycle" in r]
        mean_lat = st.mean(ms_list) if ms_list else 0.0
        sd_lat = st.stdev(ms_list) if len(ms_list) > 1 else 0.0
        med_lat = st.median(ms_list) if ms_list else 0.0
        p95_lat = pct(ms_list, 0.95)

        arms[name] = {
            "n": n,
            "succ": succ,
            "sr": sr,
            "mean": mean_lat,
            "sd": sd_lat,
            "med": med_lat,
            "p95": p95_lat,
            "meta": d.get("meta", {}),
        }

    print("\n  Arm Overview (100 matched states per arm):")
    print("  " + "-" * 76)
    print(f"  {'Arm':30s} {'Success':12s} {'Mean':9s} {'SD':8s} {'Median':8s} {'p95':7s}")
    print("  " + "-" * 76)
    for name, a in arms.items():
        print(f"  {name:30s} {a['sr']:5.1f}% ({int(round(a['sr']*a['n']/100))}/{a['n']})  "
              f"{a['mean']:6.1f} ms  {a['sd']:5.1f}   {a['med']:6.1f}   {a['p95']:5.1f}")
    print("  " + "-" * 76)

    # Key Contrasts
    print("\n  KEY SAME-HOST CONTRASTS:")
    print_contrast("IDM: Composed tau=4 vs 2-step uncached",
                   "idm_whole_2step_p1_tau4", "idm_stock_2step", arms)
    print_contrast("IDM: Composed tau=4 vs 1-step uncached",
                   "idm_whole_2step_p1_tau4", "idm_stock_1step", arms)
    print_contrast("IDM: Composed tau=2 vs 2-step uncached",
                   "idm_whole_2step_p1_tau2", "idm_stock_2step", arms)
    print_contrast("Native: Composed tau=4 vs 2-step uncached",
                   "uncond_whole_2step_p1_tau4", "uncond_stock_2step", arms)
    print_contrast("Native: Composed tau=4 vs 1-step uncached",
                   "uncond_whole_2step_p1_tau4", "uncond_stock_1step", arms)
    print_contrast("Native: Composed tau=2 vs 2-step uncached",
                   "uncond_whole_2step_p1_tau2", "uncond_stock_2step", arms)
    print("=" * 80)


def main():
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
    else:
        root = Path(__file__).resolve().parent.parent
        p = root / "results" / "matched_composed"
    if not p.is_dir():
        print(f"Directory not found: {p}", file=sys.stderr)
        sys.exit(1)
    analyze_dir(p)


if __name__ == "__main__":
    main()

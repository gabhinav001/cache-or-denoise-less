#!/usr/bin/env python3
"""Extended evaluation report: High-powered 100-state frontier and causal schedule-shift intervention.

Analyzes:
  Part A: 100-state matched frontier (10 tasks x 10 episodes)
  Part B: Causal 2x2 factorial schedule-shift intervention (10 tasks x 5 episodes)
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
    if len(shared) != a1["n"] or len(shared) != a2["n"]:
        raise ValueError(
            f"Incomplete pair population in contrast '{name}': {arm1} has {a1['n']} rows, "
            f"{arm2} has {a2['n']} rows, with {len(shared)} shared states. "
            f"Paired evaluation requires identical, validated state coverage."
        )
    b = sum(1 for k in shared if a1["succ"][k] and not a2["succ"][k])
    c = sum(1 for k in shared if not a1["succ"][k] and a2["succ"][k])
    sr1_shared = 100.0 * sum(1 for k in shared if a1["succ"][k]) / len(shared)
    sr2_shared = 100.0 * sum(1 for k in shared if a2["succ"][k]) / len(shared)
    diff = sr1_shared - sr2_shared
    p_val = exact_mcnemar(b, c)
    low, high = paired_bootstrap_ci(a1["succ"], a2["succ"], shared)
    speedup = a2["mean"] / a1["mean"] if a1["mean"] > 0 else float("nan")

    print(f"\n  Contrast: {name}")
    print(f"    {arm1} ({sr1_shared:.1f}%, {a1['mean']:.1f} ms) vs {arm2} ({sr2_shared:.1f}%, {a2['mean']:.1f} ms)")
    p_str = "< 0.0001" if p_val < 0.0001 else f"= {p_val:.4f}"
    print(f"    Diff: {diff:+.1f} pts [95% CI: {low:+.1f}, {high:+.1f}] | Discordant: {b}/{c} | McNemar p {p_str}")
    print(f"    Speedup: {speedup:.2f}x ({arm1} relative to {arm2})")


def process_arms(files):
    arms = {}
    ref_keys = {}
    ref_state_hashes = {}
    ref_arm_names = {}
    for p in files:
        base = os.path.splitext(os.path.basename(p))[0]
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception as e:
            continue
        rows = d.get("rows", [])
        if not rows:
            continue
        post = [r for r in rows if not r.get("warmup", False)]
        ep_means = [r["ms_per_cycle"] for r in post if "ms_per_cycle" in r]
        all_cyc = []
        ref_cyc = []
        reuse_cyc = []
        for r in post:
            for ms, is_ref in zip(r.get("cycle_ms", []), r.get("cycle_refresh", [])):
                all_cyc.append(ms)
                if is_ref:
                    ref_cyc.append(ms)
                else:
                    reuse_cyc.append(ms)

        seen_keys = set()
        state_hashes = {}
        for r in rows:
            k = (r["task"], r["episode"])
            if k in seen_keys:
                raise ValueError(f"Duplicate (task, episode) key {k} in {p}")
            seen_keys.add(k)
            state_hashes[k] = r.get("init_state")

        cohort = "causal" if base.startswith("causal_") else "frontier"
        expected_len = 50 if cohort == "causal" else 100
        if len(rows) != expected_len:
            raise ValueError(
                f"Arm {base} ({cohort} cohort) has {len(rows)} rows; expected exactly {expected_len} states."
            )

        if cohort not in ref_state_hashes:
            ref_state_hashes[cohort] = state_hashes
            ref_keys[cohort] = sorted(seen_keys)
            ref_arm_names[cohort] = base
        else:
            if sorted(seen_keys) != ref_keys[cohort]:
                discrepant = set(ref_keys[cohort]) ^ seen_keys
                raise ValueError(
                    f"Task/episode key mismatch in {p} vs reference {ref_arm_names[cohort]}: {discrepant}"
                )
            if state_hashes != ref_state_hashes[cohort]:
                raise ValueError(
                    f"State hash mismatch in {p} vs reference {ref_arm_names[cohort]}"
                )

        succ_by_state = {}
        for r in rows:
            key = (r["task"], r["episode"], r.get("init_state", ""))
            succ_by_state[key] = bool(r["success"])

        arms[base] = {
            "n": len(rows),
            "sr": 100.0 * sum(1 for r in rows if r["success"]) / len(rows),
            "n_succ": sum(1 for r in rows if r["success"]),
            "mean": st.mean(ep_means) if ep_means else float("nan"),
            "sd": st.stdev(ep_means) if len(ep_means) > 1 else 0.0,
            "ref": st.mean(ref_cyc) if ref_cyc else float("nan"),
            "reuse": st.mean(reuse_cyc) if reuse_cyc else float("nan"),
            "p95": pct(all_cyc, 0.95),
            "succ": succ_by_state,
            "steps": d.get("meta", {}).get("num_inference_steps", "?"),
            "shift": d.get("meta", {}).get("runtime", {}).get("sigma_shift_override", "?"),
            "ckpt": d.get("meta", {}).get("runtime", {}).get("model_class", "?")
        }
    return arms


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    files = [p for p in sorted(glob.glob(os.path.join(d, "*.json")))
             if not os.path.basename(p).startswith(("gate", "_"))]
    if not files:
        print(f"  no evaluation artefacts in {d} -- not yet collected.")
        return 0

    all_arms = process_arms(files)
    part_a_arms = {k: v for k, v in all_arms.items() if not k.startswith("causal_")}
    part_b_arms = {k: v for k, v in all_arms.items() if k.startswith("causal_")}

    print("=" * 80)
    print("  MATCHED 100-STATE FRONTIER AND SHIFT INTERVENTION REPORT")
    print("=" * 80)

    if part_a_arms:
        print("\n--------------------------------------------------------------------------------")
        print("  PART A: HIGH-POWERED MATCHED FRONTIER EVALUATION")
        print("--------------------------------------------------------------------------------")
        print(f"  {'Arm':<28} {'Steps':>5} {'Shift':>5} {'Success (N)':>16} {'Mean':>8} {'SD':>6} {'Refresh':>8} {'Reuse':>8} {'p95':>8}")
        print("  " + "-" * 98)
        for name, a in part_a_arms.items():
            succ_str = f"{a['sr']:.1f}% ({a['n_succ']}/{a['n']})"
            ref_str = f"{a['ref']:.1f}" if not math.isnan(a['ref']) else "-"
            reuse_str = f"{a['reuse']:.1f}" if not math.isnan(a['reuse']) else "-"
            print(f"  {name:<28} {str(a['steps']):>5} {str(a['shift']):>5} {succ_str:>16} {a['mean']:8.1f} {a['sd']:6.1f} {ref_str:>8} {reuse_str:>8} {a['p95']:8.1f}")

        print("\n  KEY HIGH-POWERED FRONTIER CONTRASTS:")
        print_contrast("IDM: 2-step uncached vs 10-step full compute", "idm_stock_2step", "idm_stock_10step", part_a_arms)
        print_contrast("IDM: 2-step uncached vs whole-stack plain tau=2", "idm_stock_2step", "idm_whole_plain_tau2", part_a_arms)
        print_contrast("IDM: 2-step uncached vs whole-stack attenuated tau=8", "idm_stock_2step", "idm_whole_agg_tau8", part_a_arms)
        print_contrast("Native: 1-step uncached vs 2-step uncached", "uncond_stock_1step", "uncond_stock_2step", part_a_arms)
        print_contrast("Native: 1-step uncached vs 10-step full compute", "uncond_stock_1step", "uncond_stock_10step", part_a_arms)
        print_contrast("Native: 1-step uncached vs 4-step cached (tau=4)", "uncond_stock_1step", "uncond_whole_4step_tau4", part_a_arms)
        print_contrast("Native: 4-step uncached vs 4-step cached (tau=4)", "uncond_stock_4step", "uncond_whole_4step_tau4", part_a_arms)

    if part_b_arms:
        print("\n--------------------------------------------------------------------------------")
        print("  PART B: CAUSAL SCHEDULE-SHIFT INTERVENTION (2x2 FACTORIAL)")
        print("--------------------------------------------------------------------------------")
        print(f"  {'Arm':<32} {'Ckpt':>8} {'Shift':>6} {'Arm Config':>12} {'Success (N)':>16} {'Mean':>8}")
        print("  " + "-" * 86)
        for name, a in part_b_arms.items():
            succ_str = f"{a['sr']:.1f}% ({a['n_succ']}/{a['n']})"
            ckpt_name = "IDM" if "idm" in name else "Native"
            cfg_name = "whole_tau8" if "whole" in name else "stock_2step"
            print(f"  {name:<32} {ckpt_name:>8} {str(a['shift']):>6} {cfg_name:>12} {succ_str:>16} {a['mean']:8.1f}")

        print("\n  CAUSAL SHIFT CONTRASTS (Holding Checkpoint and Initial States Fixed):")
        print_contrast("Causal Shift: IDM tau=8 plain caching (Shift 1.0 -> 5.0)", "causal_idm_s5_whole_tau8", "causal_idm_s1_whole_tau8", part_b_arms)
        print_contrast("Causal Shift: Native tau=8 plain caching (Shift 5.0 -> 1.0)", "causal_uncond_s5_whole_tau8", "causal_uncond_s1_whole_tau8", part_b_arms)
        print_contrast("Causal Shift: IDM 2-step uncached (Shift 1.0 -> 5.0)", "causal_idm_s5_stock2", "causal_idm_s1_stock2", part_b_arms)
        print_contrast("Causal Shift: Native 2-step uncached (Shift 5.0 -> 1.0)", "causal_uncond_s5_stock2", "causal_uncond_s1_stock2", part_b_arms)

    print("=" * 80)
    return 0


if __name__ == "__main__":
    main()

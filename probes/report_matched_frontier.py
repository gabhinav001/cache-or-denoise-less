#!/usr/bin/env python3
"""Matched frontier analysis: paired accuracy and latency across implementations.

Compares:
  - 2-step stock (clean uncached baseline)
  - 10-step stock (full compute uncached baseline)
  - 10-step whole-stack plain caching (tau=8)
  - 10-step whole-stack preaggregated attenuation (tau=8)
  - 10-step blockwise plain caching (tau=8)
  - 10-step blockwise attenuated caching (tau=8)

All arms evaluated on identical benchmark states with CUDA synchronization.
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
    # 2 * sum_{i=0}^k binom(n, i) * 0.5^n
    p = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * p)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    files = [p for p in sorted(glob.glob(os.path.join(d, "*.json")))
             if not os.path.basename(p).startswith(("gate", "_"))]
    if not files:
        print(f"  no matched frontier artefacts in {d} -- not yet collected.")
        return 0

    arms = {}
    loads = []
    gpu_names = set()

    for p in files:
        stem = Path(p).stem
        j = json.load(open(p))
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
    print(f"  MATCHED FRONTIER REPORT")
    print(f"  GPU: {', '.join(sorted(gpu_names)) or 'unknown'} | Max load per core: {hi:.2f}")
    if hi > 0.5:
        print(f"  WARNING: Contended host (>0.5 load/core); wall-clock timing subject to noise.")
    print(f"================================================================================\n")

    print(f"  {'Arm':22} {'Steps':>5} {'Success (%)':>14} {'Mean (ms)':>10} {'SD':>6} {'Refresh':>9} {'Reuse':>9} {'p95':>8}")
    print(f"  {'-'*22} {'-'*5} {'-'*14} {'-'*10} {'-'*6} {'-'*9} {'-'*9} {'-'*8}")

    display_order = [
        "stock_2step",
        "stock_10step",
        "whole_plain_tau8",
        "whole_agg_tau8",
        "block_plain_tau8",
        "block_shrink_tau8",
    ]
    sorted_keys = [k for k in display_order if k in arms] + [k for k in arms if k not in display_order]

    for k in sorted_keys:
        a = arms[k]
        succ_str = f"{a['sr']:.1f}% ({a['n_succ']}/{a['n_tot']})"
        ref_str = f"{a['refresh_ms']:.1f}" if math.isfinite(a['refresh_ms']) else "-"
        reu_str = f"{a['reuse_ms']:.1f}" if math.isfinite(a['reuse_ms']) else "-"
        p95_str = f"{a['p95_ms']:.1f}" if math.isfinite(a['p95_ms']) else "-"
        print(f"  {a['stem']:22} {a['steps']:5} {succ_str:>14} {a['mean_ms']:10.1f} {a['sd_ms']:6.1f} {ref_str:>9} {reu_str:>9} {p95_str:>8}")

    print(f"\n--------------------------------------------------------------------------------")
    print(f"  KEY PAIRWISE CONTRASTS ON MATCHED INITIAL STATES")
    print(f"--------------------------------------------------------------------------------")

    pairs = [
        ("stock_2step", "stock_10step", "Short uncached vs Full uncached"),
        ("stock_2step", "whole_plain_tau8", "Short uncached vs Whole-stack plain"),
        ("stock_2step", "whole_agg_tau8", "Short uncached vs Whole-stack aggregated attenuation"),
        ("whole_agg_tau8", "whole_plain_tau8", "Whole-stack attenuation vs Plain whole-stack"),
        ("block_shrink_tau8", "block_plain_tau8", "Blockwise attenuation vs Plain blockwise"),
        ("whole_agg_tau8", "block_shrink_tau8", "Whole-stack aggregation vs Blockwise attenuation (BF16 check)"),
    ]

    for a1_key, a2_key, desc in pairs:
        if a1_key not in arms or a2_key not in arms:
            continue
        a1, a2 = arms[a1_key], arms[a2_key]
        shared_keys = sorted(set(a1["keyed_success"]) & set(a2["keyed_success"]))
        if not shared_keys:
            continue
        # wins of a1 over a2, wins of a2 over a1
        w1 = sum(1 for k in shared_keys if a1["keyed_success"][k] and not a2["keyed_success"][k])
        w2 = sum(1 for k in shared_keys if a2["keyed_success"][k] and not a1["keyed_success"][k])
        p_val = exact_mcnemar(w1, w2)
        diff_pts = a1["sr"] - a2["sr"]
        speedup = a2["mean_ms"] / a1["mean_ms"] if a1["mean_ms"] > 0 else float("nan")

        print(f"\n  Contrast: {desc}")
        print(f"    {a1_key} ({a1['sr']:.1f}%, {a1['mean_ms']:.1f} ms) vs {a2_key} ({a2['sr']:.1f}%, {a2['mean_ms']:.1f} ms)")
        print(f"    Diff: {diff_pts:+.1f} pts success | Discordant: {w1}/{w2} | McNemar p={p_val:.4f}")
        print(f"    Speedup: {speedup:.2f}x ({a1_key} relative to {a2_key})")

    print(f"\n================================================================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

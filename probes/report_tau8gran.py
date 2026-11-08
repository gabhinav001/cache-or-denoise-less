"""Is the weak plain-cache baseline a property of the method or of our cache?

Five arms on the same episodes: full compute, our per-block plain cache, a
whole-stack plain cache (the published form), that cache attenuated, and our
per-cell attenuated cache. Comparisons are paired on (task, episode) within a
seed, and the seed axis carries the interval -- never pooled with anything else.

The question the table answers first: does `whole_tau8` sit near the published
96.4 while `c3ache_tau8` sits near 67.6? If so the baseline was ours.

usage: report_tau8gran.py [results/tau8gran]
"""
from __future__ import annotations

import glob, json, math, os, sys
from collections import defaultdict

ARMS = ["full", "c3ache_tau8", "whole_tau8", "whole_tau8_shrink", "c3ache_tau8_shrink"]
PUBLISHED = 96.4


def load(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "gran*.json"))):
        seed = json.load(open(p)).get("meta", {}).get("seed")
        rows = json.load(open(p))["rows"]
        by = defaultdict(dict)
        for r in rows:
            by[r["config"]][(r["task"], r["episode"])] = bool(r["success"])
        out[seed if seed is not None else os.path.basename(p)] = by
    return out


def mean(v): return sum(v) / len(v) if v else float("nan")


def tinterval(v):
    """Seed-level 95% t interval; the seed axis is the only replication here."""
    n = len(v)
    if n < 2: return (float("nan"), float("nan"))
    m = mean(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}.get(n, 2.0)
    h = t * sd / math.sqrt(n)
    return (m - h, m + h)


def main(d):
    runs = load(d)
    if not runs:
        print(f"  SKIP  no gran*.json under {d}")
        print("  Collect with: ./repro/run_experiment.sh tau8gran")
        return 0
    print(f"\n  {len(runs)} seed(s): {sorted(runs)}\n")
    print(f"    {'arm':<22} " + "".join(f"{s:>8}" for s in sorted(runs)) + f"{'mean':>9}")
    per_arm = {}
    for arm in ARMS:
        vals = []
        cells = []
        for s in sorted(runs):
            b = runs[s].get(arm)
            cells.append(f"{100*mean(list(b.values())):8.1f}" if b else f"{'-':>8}")
            if b: vals.append(100 * mean(list(b.values())))
        per_arm[arm] = vals
        print(f"    {arm:<22} " + "".join(cells) + f"{mean(vals):9.1f}")

    print(f"\n  published plain cache: {PUBLISHED}")
    for arm in ("c3ache_tau8", "whole_tau8"):
        if per_arm.get(arm):
            print(f"    {arm:<22} {mean(per_arm[arm]):6.1f}  "
                  f"gap {mean(per_arm[arm]) - PUBLISHED:+.1f}")

    # paired differences on the seed axis
    def paired(a, b):
        diffs = []
        for s in sorted(runs):
            A, B = runs[s].get(a), runs[s].get(b)
            if not A or not B: continue
            keys = sorted(set(A) & set(B))
            if keys:
                diffs.append(100 * mean([A[k] - B[k] for k in keys]))
        return diffs

    print("\n  paired seed-level differences (percentage points)")
    for a, b, why in (("whole_tau8", "c3ache_tau8", "granularity, plain"),
                      ("whole_tau8_shrink", "whole_tau8", "attenuation on the published form"),
                      ("c3ache_tau8_shrink", "c3ache_tau8", "attenuation on ours"),
                      ("c3ache_tau8_shrink", "whole_tau8_shrink", "per-cell over whole-stack")):
        dz = paired(a, b)
        if not dz: continue
        lo, hi = tinterval(dz)
        print(f"    {a} - {b:<22} {mean(dz):+6.1f}  [{lo:+.1f},{hi:+.1f}]   {why}")

    gran = paired("whole_tau8", "c3ache_tau8")
    if gran:
        lo, hi = tinterval(gran)
        print()
        if lo > 0:
            print("  The published granularity beats ours on the same episodes. The weak")
            print("  plain-cache baseline is an artefact of per-block substitution, and")
            print("  every claim resting on that baseline has to be restated.")
        elif hi < 0:
            print("  Our granularity beats the published one; the gap is not explained by")
            print("  per-block substitution.")
        else:
            print("  The interval spans zero: this collection does not separate the two")
            print("  granularities. It does not establish that they agree.")
    return 0


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else os.path.join(here, "..", "results", "tau8gran")))

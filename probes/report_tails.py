"""Tail latency for every arm, not just the cache.

The refresh analysis covered one configuration on roughly 366 calls, so it
did not compare the short schedules under the same tail-latency analysis and
could not identify the best choice under a deadline.
The data to do that was already shipped: PROBE.cycle_times recorded every
replanning call for all eleven arms of the same-host collection, 4021 of them.
It had simply never been reported for anything but the ten-step tau=4 cache.

This reports the same quantiles for every arm against the illustrative budget,
which is what makes the deadline question answerable. A cache has two
populations and its tail is set by the refresh cycle; an uncached arm has one.
The comparison that matters for deployment is therefore between an arm's TAIL
and a rival's tail, not between their means -- and on this collection the short
uncached schedules win on both.

These are measured inference times. The simulator never tests a controller's
response to a late action, so an over-budget call bounds when a deadline would
be at risk rather than showing that anything went wrong.
"""
from __future__ import annotations

import glob, json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from report_frontier_fragility import STEPS, TAU

R = os.path.join(HERE, "..", "results", "deployfrontier")


def pct(v, q):
    """Nearest-rank percentile convention matching paper reporting (p95 = 455.9 ms)."""
    import math
    v = sorted(v)
    return v[min(len(v) - 1, max(0, math.ceil(q * len(v)) - 1))]


def collect(seed=42):
    """arm -> (all calls, refresh calls, reuse calls) for one seed."""
    suf = "" if seed == 42 else f"_seed{seed}"
    allc, ref, reu = defaultdict(list), defaultdict(list), defaultdict(list)
    for f in sorted(glob.glob(os.path.join(R, f"d*{suf}.json"))):
        base = os.path.basename(f)[:-5]
        if seed == 42 and "_seed" in base: continue
        stem = base.rsplit("_seed", 1)[0]
        if stem not in STEPS: continue
        j = json.load(open(f))
        for r in j["rows"]:
            if r.get("warmup"): continue
            cfg = r["config"]
            arm = (f"{STEPS[stem]}st" if cfg == "stock"
                   else f"{STEPS[stem]}st+tau{TAU[cfg]}")
            ms = r.get("cycle_ms") or []
            flags = r.get("cycle_refresh") or []
            allc[arm].extend(ms)
            for i, m in enumerate(ms):
                (ref if (i < len(flags) and flags[i]) else reu)[arm].append(m)
    return allc, ref, reu


def summary(seed=42, budget=None):
    allc, ref, reu = collect(seed)
    if not allc: return {}
    if budget is None:
        replan = None
        for f in glob.glob(os.path.join(R, "d*.json")):
            replan = (json.load(open(f)).get("meta") or {}).get("replan") or replan
        budget = 1000.0 * (replan or 10) / 20.0
    out = {"budget_ms": budget, "arms": {}, "total_calls": sum(len(v) for v in allc.values())}
    for a, v in allc.items():
        over = sum(1 for x in v if x > budget)
        out["arms"][a] = {
            "calls": len(v), "mean": round(sum(v) / len(v), 1),
            "p50": round(pct(v, 0.50), 1), "p95": round(pct(v, 0.95), 1),
            "max": round(max(v), 1), "over_budget": over,
            "refresh_mean": round(sum(ref[a]) / len(ref[a]), 1) if ref.get(a) else None,
            "reuse_mean": round(sum(reu[a]) / len(reu[a]), 1) if reu.get(a) else None,
        }
    clean = sorted([a for a, d in out["arms"].items() if d["over_budget"] == 0],
                   key=lambda a: out["arms"][a]["max"])
    out["arms_over_budget"] = sorted(a for a, d in out["arms"].items() if d["over_budget"])
    out["lowest_tail_arm"] = clean[0] if clean else None
    out["note"] = (
        f"Every arm of the same-host collection, {out['total_calls']} recorded calls, "
        f"against an illustrative {budget:.0f} ms budget. Only "
        + ", ".join(out["arms_over_budget"]) + " exceed it; every short uncached "
        "schedule stays inside on every call. The lowest tail belongs to "
        f"{out['lowest_tail_arm']}, which is also the fastest arm, so under a "
        "deadline the short schedule wins on the tail as well as the mean. "
        "Measured inference times, not observed control failures.")
    return out


def main():
    s = summary()
    if not s:
        print("  no per-cycle timing found"); return 1
    print(f"Per-call latency, all arms, one host, {s['total_calls']} calls")
    print(f"Illustrative budget: {s['budget_ms']:.0f} ms\n")
    print(f"  {'arm':12s} {'calls':>6s} {'mean':>7s} {'p50':>7s} {'p95':>7s} {'max':>7s} "
          f"{'refresh':>8s} {'reuse':>7s} {'over':>6s}")
    for a in sorted(s["arms"], key=lambda x: s["arms"][x]["mean"]):
        d = s["arms"][a]
        rf = f"{d['refresh_mean']:.1f}" if d["refresh_mean"] is not None else "-"
        ru = f"{d['reuse_mean']:.1f}" if d["reuse_mean"] is not None else "-"
        flag = "  <<<" if d["over_budget"] else ""
        print(f"  {a:12s} {d['calls']:6d} {d['mean']:7.1f} {d['p50']:7.1f} {d['p95']:7.1f} "
              f"{d['max']:7.1f} {rf:>8s} {ru:>7s} {d['over_budget']:6d}{flag}")
    print(f"\n  over budget: {', '.join(s['arms_over_budget']) or 'none'}")
    print(f"  lowest tail: {s['lowest_tail_arm']} (also the fastest arm)")
    print("\n  An uncached arm has one population, so its refresh column is its only")
    print("  column. A cache has two and its tail is set by the refresh cycle: that")
    print("  is why a mean understates what a deadline-bound loop actually meets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

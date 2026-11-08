#!/usr/bin/env python3
"""Latency per arm, with a refusal to report it from a contended host.

usage: report_frontier_timing.py <dir>

Every number here is wall clock, which is a property of the hardware AND of
whatever else was running on it. This box was measured at load 63 on 36 CPUs
and, hours later, at 1.8; the full-compute arm read 5722 ms/cycle on one task
and 360 on another. So the load is checked first and a busy run is refused
rather than averaged in.
"""
import glob, json, os, statistics as st, sys

BUSY = 0.5          # load average above this share of cores -> refuse


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    files = [p for p in sorted(glob.glob(os.path.join(d, "*.json")))
             if not os.path.basename(p).startswith(("gate", "_"))]
    if not files:
        print(f"  no frontier_timing artefacts in {d} -- not yet collected.")
        return 0

    rows, refused = [], []
    for p in files:
        j = json.load(open(p))
        m = j.get("meta") or {}
        prov = m.get("provenance") or {}
        la = prov.get("loadavg_1_5_15")
        nc = prov.get("cpu_count") or 1
        steps = m.get("num_inference_steps")
        tim = {}
        for r in j["rows"]:
            if "ms_per_cycle" in r and not r.get("warmup"):
                tim.setdefault(r["config"], []).append(r["ms_per_cycle"])
        busy = (la is not None and nc and (la[0] / nc) > BUSY)
        for c, v in tim.items():
            rec = {"file": os.path.basename(p)[:-5], "arm": c, "steps": steps,
                   "ms": st.mean(v), "sd": st.stdev(v) if len(v) > 1 else 0.0,
                   "n": len(v), "load": (la[0] if la else None), "cpus": nc}
            (refused if busy else rows).append(rec)

    if refused:
        print("  REFUSED -- collected on a contended host\n")
        print(f"    {'run':10} {'arm':26} {'ms/cycle':>9} {'load':>7} {'cpus':>5}")
        for r in refused:
            print(f"    {r['file']:10} {r['arm'][:26]:26} {r['ms']:9.1f} "
                  f"{(r['load'] or 0):7.1f} {r['cpus']:5}")
        print(f"\n    Load above {BUSY:.0%} of the core count. These are not")
        print("    measurements of this software and are not reported as such.")

    if not rows:
        print("\n  Nothing usable. Re-collect on a quiet host.")
        return 1

    print("\n  LATENCY (quiet host only)\n")
    print(f"    {'arm':30} {'steps':>5} {'ms/cycle':>9} {'sd':>7} {'n':>4} {'load':>6}")
    for r in sorted(rows, key=lambda r: r["ms"]):
        lbl = f"{r['arm']} ({r['steps']} steps)" if r["arm"] == "full" else r["arm"]
        print(f"    {lbl[:30]:30} {str(r['steps']):>5} {r['ms']:9.1f} {r['sd']:7.2f} "
              f"{r['n']:4} {(r['load'] or 0):6.1f}")

    base = next((r for r in rows if r["arm"] == "full" and r["steps"] == 10), None)
    if base:
        print(f"\n    Speedups against full compute at 10 steps ({base['ms']:.1f} ms):\n")
        for r in sorted(rows, key=lambda r: r["ms"]):
            if r is base: continue
            lbl = f"{r['arm']} ({r['steps']} steps)" if r["arm"] == "full" else r["arm"]
            print(f"      {lbl[:30]:30} {base['ms']/r['ms']:6.3f}x")

    worst = max(r["sd"] / r["ms"] for r in rows)
    print(f"\n    Worst relative sd {100*worst:.1f}%. latency.sh reaches 1-2% on a")
    print("    quiet box; anything far above that means the host moved during")
    print("    the run and the ratios should not be quoted to three digits.")
    print("\n    These numbers pair with the SUCCESS column from baselines.sh,")
    print("    which is seed-dependent and load-independent. Latency is the")
    print("    opposite on both counts, which is why they are collected apart.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

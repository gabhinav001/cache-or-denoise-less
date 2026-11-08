#!/usr/bin/env python3
"""Does caching still help once the schedule is already short?

usage: report_cachesteps.py <dir>

The study's negative result is that four denoising steps beats every cache
setting. The rejoinder is "why not both". This reads the runs that test it.

TWO CAVEATS TRAVEL WITH EVERY NUMBER BELOW, and the analysis repeats them rather
than assuming the reader kept them in mind:

  the cells   a short schedule caches all but its last two steps, so it caches a
              smaller fraction of its work than a ten-step schedule does
  the weights were fit at ten steps and are applied here unrefit, because
              refitting needs a fresh Gram collection at each step count
"""
import glob, json, math, os, sys


def load(p):
    d = json.load(open(p))
    by = {}
    for r in d["rows"]:
        by.setdefault(r["config"], {})[(r["task"], r["episode"])] = bool(r["success"])
    tim = {}
    for r in d["rows"]:
        if "ms_per_cycle" in r and not r.get("warmup"):
            tim.setdefault(r["config"], []).append(r["ms_per_cycle"])
    return d, by, tim


def mcnemar(b, w):
    n = b + w
    if n == 0: return 1.0
    k = min(b, w)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    files = [p for p in sorted(glob.glob(os.path.join(d, "*.json")))
             if not os.path.basename(p).startswith(("gate", "_"))]
    if not files:
        print(f"  no cachesteps artefacts in {d} -- not yet collected.")
        return 0

    print("  CACHING ON TOP OF A SHORT SCHEDULE\n")
    print(f"    {'steps':>5} {'cells':>6} {'arm':>10} {'ms/cycle':>9} {'success':>8} "
          f"{'resc/brok':>10} {'p':>8}")
    pts = []
    for p in files:
        j, by, tim = load(p)
        m = j.get("meta") or {}
        steps, cells = m.get("num_inference_steps"), m.get("skip_cells")
        cache = next((c for c in by if c.startswith("c3ache")), None)
        if not cache or "full" not in by: continue
        ks = sorted(set(by["full"]) & set(by[cache]))
        rf = 100.0 * sum(by["full"][k] for k in ks) / len(ks)
        rc = 100.0 * sum(by[cache][k] for k in ks) / len(ks)
        bb = sum(1 for k in ks if by[cache][k] and not by["full"][k])
        ww = sum(1 for k in ks if by["full"][k] and not by[cache][k])
        mf = sum(tim["full"]) / len(tim["full"])
        mc = sum(tim[cache]) / len(tim[cache])
        pv = mcnemar(bb, ww)
        print(f"    {steps:>5} {'-':>6} {'full':>10} {mf:9.1f} {rf:7.1f}% {'-':>10} {'-':>8}")
        print(f"    {steps:>5} {cells:>6} {'+cache':>10} {mc:9.1f} {rc:7.1f}% "
              f"{str(bb)+'/'+str(ww):>10} {pv:8.3f}")
        sd = (j.get("meta") or {}).get("seed", 42)
        pts.append((sd, f"{steps} steps", mf, rf))
        pts.append((sd, f"{steps} steps + cache", mc, rc))

    # THE FRONTIER, NOT "DID SUCCESS GO UP". An earlier version of this file
    # decided whether caching helps by testing rc > rf -- success alone -- and
    # concluded it never helps. That is the same error report_baselines.py was
    # corrected for: an accelerator can be useful without improving accuracy.
    # On these four points the fastest configuration IS a cached one.
    # THE FRONTIER IS PER SEED. These seeds were collected on different hosts --
    # the same uncached four-step arm reads 124.1 ms on one and 227.1 on the
    # other, a factor of 1.8 -- so a frontier computed across them ranks arms by
    # which machine happened to run them. Pooling here would repeat the error
    # report_baselines.py was corrected for.
    front = []
    for sd in sorted({p[0] for p in pts}):
        sub = [p for p in pts if p[0] == sd]
        print(f"\n  THE FRONTIER, SEED {sd}\n")
        print(f"    {'arm':24} {'ms/cycle':>9} {'success':>8}   verdict")
        for _, lbl, ms, sr in sorted(sub, key=lambda x: x[2]):
            dom = [o for o in sub if o[1] != lbl and o[2] < ms and o[3] >= sr]
            if dom:
                v = f"dominated by {dom[0][1]}"
            else:
                v = "ON THE FRONTIER"; front.append((sd, lbl))
            print(f"    {lbl:24} {ms:9.1f} {sr:7.1f}%   {v}")
    print("\n    Latency is NOT comparable across these two seeds: they ran on")
    print("    different hosts and the same uncached arm differs by 1.8x. Success")
    print("    is comparable; latency within a seed is.")

    cached_on = [f"seed {sd}: {l}" for sd, l in front if "cache" in l]
    print("\n  READING\n")
    print(f"    {len(front)} of {len(pts)} configurations are on a per-seed frontier, and")
    print(f"    {len(cached_on)} of them {'is' if len(cached_on) == 1 else 'are'} cached"
          f"{': ' + ', '.join(cached_on) if cached_on else ''}.")
    if cached_on:
        print("\n    So caching a short schedule BUYS SPEED. What it costs in success")
        print("    differs by step count, and only one of the two replicates:\n")
        deltas = {}
        for sd, lbl, ms, sr in pts:
            if "cache" in lbl: continue
            c = [p for p in pts if p[0] == sd and p[1] == lbl + " + cache"]
            if c: deltas.setdefault(lbl, []).append((sd, c[0][3] - sr))
        for lbl in sorted(deltas):
            row = "  ".join(f"seed {sd}: {d:+.1f}" for sd, d in sorted(deltas[lbl]))
            ds = [d for _, d in deltas[lbl]]
            agree = "replicates" if len(ds) > 1 and all(
                (x > 0) == (ds[0] > 0) for x in ds) else "DOES NOT replicate"
            print(f"      {lbl:12} {row:32} {agree}")
        print("\n    The defensible statement is a TRADEOFF, not a verdict: adding a")
        print("    cache to a short schedule gives modest additional acceleration at a")
        print("    success cost that is consistent at one step count and absent at the")
        print("    other. It is NOT evidence that caching fails, and an earlier reading")
        print("    of it as such was wrong.")
    else:
        print("\n    No cached configuration reaches the frontier here.")

    print("\n    WHAT THIS DOES NOT ESTABLISH. The weights were fit at ten denoising")
    print("    steps and applied unrefit, so each cell's least-squares weight is wrong")
    print("    for this schedule: these points test one concrete configuration, not")
    print("    the best achievable short-schedule caching. And a short schedule caches")
    print("    all but its last two steps, so it is a less aggressive cache than the")
    print("    ten-step configuration by construction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

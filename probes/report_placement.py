"""Matched-budget placement comparison (paper Table 5).

Five placements each skip exactly 240 of 300 block evaluations, so compute is
identical and only which cells are cached differs. Deviation is relative L2 of
the action chunk against full compute on the same frame, which makes every
comparison paired: the sign test below is over paired frames, ties dropped.
"""
from __future__ import annotations
import json, math, statistics as st, sys


def sign_test(rows, a, b):
    w = sum(1 for r in rows if r[b] < r[a])
    l = sum(1 for r in rows if r[b] > r[a])
    n = w + l
    if n == 0: return w, l, 1.0
    k = min(w, l)
    return w, l, min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main(path):
    d = json.load(open(path))
    rows = d["placement"]
    names = [k for k in rows[0] if k not in ("ep", "suite", "episode", "rep", "cycle")]
    print(f"{len(rows)} paired frames | {len({r['ep'] for r in rows})} episodes "
          f"| {len({r['rep'] for r in rows})} repeats | {len({r['suite'] for r in rows})} suites\n")
    means = {n: st.mean(r[n] for r in rows) for n in names}
    base = min(means, key=means.get)
    print(f"{'placement':12s} {'mean dev':>9s} {'vs best':>9s} {'wins':>6s} {'sign test p':>13s}")
    for n in sorted(names, key=lambda x: means[x]):
        if n == base:
            print(f"{n:12s} {means[n]:9.4f} {'---':>9s} {'---':>6s} {'---':>13s}")
            continue
        w, l, p = sign_test(rows, n, base)
        print(f"{n:12s} {means[n]:9.4f} {100*(means[n]/means[base]-1):+8.1f}% "
              f"{100*l/(w+l):5.0f}% {p:13.1e}")
    # The oracle arm is only present when probes/fw_scale.py was given a cell set
    # via +PROBE.oracle. That file was never committed and nothing in this
    # repository regenerates it, so a fresh collection has four placements rather
    # than five and this comparison cannot be made. Reported rather than raised:
    # the other four placements are still a valid matched-budget comparison, and
    # crashing here would throw them away too.
    if "oracle" in names:
        w, l, p = sign_test(rows, "random", "oracle")
        print(f"\noracle vs random: {w} wins / {l} losses, p = {p:.1e}")
    else:
        print("\n  no oracle placement in this artefact -- it requires a cell set at")
        print("  +PROBE.oracle that was never committed (see repro/experiments/placement.sh).")
        print("  The four placements above are unaffected; the oracle comparison is not")
        print("  reproducible from this repository and the write-up says so.")

    # within-process determinism: same configuration, repeated inside one run
    from collections import defaultdict
    by = defaultdict(dict)
    for r in rows: by[(r["ep"], r["cycle"])][r["rep"]] = r[base]
    full = [v for v in by.values() if len(v) == max(len(x) for x in by.values())]
    spread = max(max(v.values()) - min(v.values()) for v in full)
    print(f"determinism: {len(full)} (episode, cycle) pairs repeated, "
          f"max spread {spread:.3e}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../results/fw_scale_a100.json")

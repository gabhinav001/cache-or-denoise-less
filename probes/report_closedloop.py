"""Paired analysis of closed-loop arms that differ only in the substitution rule.

Both arms of a pair run the same tasks, the same initial states and the same
seed, so every episode yields a matched pair of outcomes. Episodes where the two
arms agree carry no information about the difference between them; only the
disagreements do. That is McNemar's test, and it is the right one here because
the pairing removes task and initial-state difficulty from the comparison
entirely.

The bootstrap interval is over episodes rather than over the discordant counts,
so it reflects the sampling variability an independent replication would face.
"""
from __future__ import annotations

import json, math, random, sys
from collections import defaultdict


def load(path):
    d = json.load(open(path))
    by = defaultdict(dict)
    for r in d["rows"]:
        by[(r["task"], r["episode"])][r["config"]] = r["success"]
    return d, by


def rate(by, cfg):
    v = [x[cfg] for x in by.values() if cfg in x]
    return (100 * sum(v) / len(v), len(v)) if v else (float("nan"), 0)


def mcnemar(by, a, b):
    n01 = n10 = 0
    for v in by.values():
        if a in v and b in v:
            if not v[a] and v[b]: n01 += 1
            elif v[a] and not v[b]: n10 += 1
    n = n01 + n10
    if n == 0: return n01, n10, 1.0
    k = min(n01, n10)
    return n01, n10, min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def boot(by, a, b, iters=20000, seed=0):
    rng = random.Random(seed)
    pairs = [(v[a], v[b]) for v in by.values() if a in v and b in v]
    if not pairs: return float("nan"), float("nan"), float("nan")
    d = sum(y - x for x, y in pairs) / len(pairs)
    s = []
    for _ in range(iters):
        idx = [rng.randrange(len(pairs)) for _ in pairs]
        s.append(sum(pairs[i][1] - pairs[i][0] for i in idx) / len(pairs))
    s.sort()
    return d, s[int(0.025 * iters)], s[int(0.975 * iters)]


def main(path):
    d, by = load(path)
    cfgs = sorted({c for v in by.values() for c in v})
    print(f"{path}\n  configs: {cfgs}")
    print(f"  {len(by)} distinct (task, episode) slots\n")
    print("  config                   success      n")
    for c in cfgs:
        r, n = rate(by, c)
        print(f"    {c:22s} {r:6.1f}%  {n:5d}")

    pairs = [(c, c + "_shrink") for c in cfgs if c + "_shrink" in cfgs]
    print("\n  paired comparisons (same task, same initial state, same seed)")
    for a, b in pairs:
        ra, _ = rate(by, a); rb, _ = rate(by, b)
        n01, n10, p = mcnemar(by, a, b)
        diff, lo, hi = boot(by, a, b)
        print(f"\n    {a} -> {b}")
        print(f"      {ra:.1f}%  ->  {rb:.1f}%   ({100*diff:+.1f} points, "
              f"95% CI [{100*lo:+.1f}, {100*hi:+.1f}])")
        print(f"      rescued {n01}, broken {n10}, McNemar exact p = {p:.3f}")

    print("\n  per-task breakdown")
    tasks = sorted({k[0] for k in by})
    hdr = "    task  " + "  ".join(f"{c[:16]:>16s}" for c in cfgs)
    print(hdr)
    for t in tasks:
        cells = []
        for c in cfgs:
            v = [x[c] for k, x in by.items() if k[0] == t and c in x]
            cells.append(f"{100*sum(v)/len(v):15.0f}%" if v else " " * 16)
        print(f"    {t:4d}  " + "  ".join(cells))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../traces/tau48.json")

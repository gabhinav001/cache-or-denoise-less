"""Paired statistics for the open-loop shrinkage test.

Both arms see the same cycle, the same cache contents and the same seed, and
differ only in the weight applied to the substituted residual. So the right test
is paired: for each held-out cycle, did attenuating help or hurt? The sign test
makes no assumption about the shape of the deviation distribution, which is
heavily right-skewed.
"""
from __future__ import annotations

import json, math, sys
from collections import defaultdict


def sign_test(pairs):
    """Two-sided exact binomial on wins vs losses, ties dropped."""
    w = sum(1 for a, b in pairs if b < a)
    l = sum(1 for a, b in pairs if b > a)
    n = w + l
    if n == 0: return w, l, 1.0
    k = min(w, l)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return w, l, min(1.0, 2.0 * tail / (2.0 ** n))


def main(path):
    d = json.load(open(path))
    rows = [r for r in d["rows"] if not r["in_fit"]]
    m = d["meta"]
    print(f"open-loop action deviation, {m['cells']} of 300 cells cached, "
          f"weights median {m['shrink_median']:.3f}")
    print(f"held-out cycles: {len(rows)} (episodes never used to fit the weights)\n")

    def block(rs, label):
        if not rs: return
        h = sum(r["hold"] for r in rs) / len(rs)
        s = sum(r["shrink"] for r in rs) / len(rs)
        w, l, p = sign_test([(r["hold"], r["shrink"]) for r in rs])
        ps = f"{p:.1e}" if p > 0 else "<1e-300"
        print(f"  {label:34s} n={len(rs):4d}  hold {h:.4f}  shrink {s:.4f}  "
              f"{100*(1-s/h):+5.1f}%  wins {w}/{w+l}  p={ps}")

    block(rows, "all held-out")
    print()
    by = defaultdict(list)
    for r in rows: by[r["suite"][7:14]].append(r)
    for k in sorted(by): block(by[k], f"suite {k}")
    print()
    for lo, hi, lab in [(0, 5, "early cycles"), (5, 12, "mid cycles"), (12, 99, "late cycles")]:
        block([r for r in rows if lo <= r["cycle"] < hi], lab)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../traces/fw_shrink_open.json")

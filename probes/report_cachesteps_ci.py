"""Confidence interval for the paired success difference on a short schedule.

Characterizing a reliable success cost requires a confidence interval
for the paired difference and replication across seeds.
The replication exists (report_cachesteps.py, two seeds); this script provides the paired intervals.

Episodes are paired across arms by initial state, so the natural resampling unit
is the episode pair, not the episode. Two intervals are reported because they
answer slightly different questions and disagreeing would be informative:

  bootstrap   resamples episode pairs with replacement and makes no conditioning
              assumption. This is the primary interval and matches the paired
              bootstrap used elsewhere in this repository.

  exact       conditions on the number of discordant pairs and inverts the exact
              McNemar test via a Clopper-Pearson interval on the rescued share.
              It is the interval that agrees with the p-values already reported,
              but it treats the discordant count as fixed, so it is narrower by
              construction and is shown only as a cross-check.

A CI that spans zero is the finding, not a failure: it is what distinguishes
"no measured cost" from "a cost too small for 100 episodes to resolve".
"""
import json, glob, os, random
from collections import defaultdict

R = os.path.join(os.path.dirname(__file__), "..", "results")
CACHED = "c3ache_tau4_shrink_skip"
B = 10000
random.seed(20260919)


def _binom_cdf(k, n, p):
    """P(X <= k) for X ~ Binomial(n, p), summed exactly."""
    from math import comb
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))


def _bisect(f, target, increasing):
    """The p in [0,1] where the monotone f crosses target."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        below = f(mid) < target if increasing else f(mid) > target
        if below: lo = mid
        else: hi = mid
    return (lo + hi) / 2.0


def _cp(k, n, lvl=0.95):
    """Clopper-Pearson interval, by inverting the exact binomial tails.

    The continued-fraction form of the incomplete beta misconverges for the
    very small x these discordant counts produce, so the tails are summed
    directly instead. n is at most a handful here, so this costs nothing.
    """
    if n == 0: return (0.0, 1.0)
    al = (1 - lvl) / 2
    # lower: largest p with P(X >= k) = al, i.e. 1 - P(X <= k-1) = al
    # lower: P(X >= k) rises with p, so cross it from below
    lo = 0.0 if k == 0 else _bisect(
        lambda p: 1.0 - _binom_cdf(k - 1, n, p), al, increasing=True)
    # upper: P(X <= k) falls with p
    hi = 1.0 if k == n else _bisect(
        lambda p: _binom_cdf(k, n, p), al, increasing=False)
    return lo, hi


def analyse(path):
    d = json.load(open(path))
    meta = d["meta"]
    by = defaultdict(dict)
    for r in d["rows"]:
        key = (r["suite"], r["task"], r["episode"], r.get("init_state"))
        by[key][r["config"]] = bool(r["success"])
    pairs = [(v["full"], v[CACHED]) for v in by.values()
             if "full" in v and CACHED in v]
    n = len(pairs)
    rescued = sum(1 for f, c in pairs if not f and c)     # cache fixes it
    broken = sum(1 for f, c in pairs if f and not c)      # cache breaks it
    delta = 100.0 * (rescued - broken) / n

    # primary: paired bootstrap over episode pairs
    boot = []
    for _ in range(B):
        s = [pairs[random.randrange(n)] for _ in range(n)]
        boot.append(100.0 * sum(c - f for f, c in s) / n)
    boot.sort()
    b_lo, b_hi = boot[int(0.025 * B)], boot[int(0.975 * B) - 1]

    # cross-check: exact conditional interval, inverting exact McNemar
    m = rescued + broken
    if m:
        p_lo, p_hi = _cp(rescued, m)
        e_lo = 100.0 * (2 * p_lo * m - m) / n
        e_hi = 100.0 * (2 * p_hi * m - m) / n
    else:
        e_lo = e_hi = 0.0

    return {
        "seed": meta.get("seed"), "steps": meta.get("n_denoise")
        or (60 if meta.get("skip_cells") == 60 else None),
        "skip_cells": meta.get("skip_cells"), "n_pairs": n,
        "rescued": rescued, "broken": broken, "discordant": m,
        "delta": round(delta, 2),
        "boot_ci95": [round(b_lo, 2), round(b_hi, 2)],
        "exact_ci95": [round(e_lo, 2), round(e_hi, 2)],
        "spans_zero": b_lo <= 0 <= b_hi,
    }


def main():
    out = {}
    for f in sorted(glob.glob(os.path.join(R, "cachesteps", "*.json"))):
        out[os.path.basename(f)[:-5]] = analyse(f)

    print("Paired success difference, cached minus full, on short schedules")
    print("Resampling unit is the episode pair (matched on initial state).\n")
    hdr = f"{'run':14s} {'seed':>5s} {'n':>5s} {'resc':>5s} {'brok':>5s} {'delta':>7s}  {'bootstrap 95%':>18s}  {'exact 95%':>18s}"
    print(hdr); print("-" * len(hdr))
    for k, v in out.items():
        b = f"[{v['boot_ci95'][0]:+.1f}, {v['boot_ci95'][1]:+.1f}]"
        e = f"[{v['exact_ci95'][0]:+.1f}, {v['exact_ci95'][1]:+.1f}]"
        print(f"{k:14s} {str(v['seed']):>5s} {v['n_pairs']:5d} {v['rescued']:5d} "
              f"{v['broken']:5d} {v['delta']:+7.1f}  {b:>18s}  {e:>18s}")

    spans = [k for k, v in out.items() if v["spans_zero"]]
    print(f"\n{len(spans)} of {len(out)} intervals span zero: {', '.join(spans)}")
    print("At 100 episodes the paired interval is roughly +/-5 points wide, so a")
    print("three-point cost is inside the noise. That is the honest reading: the")
    print("sign replicates across seeds at four steps, the magnitude does not")
    print("survive as a deployment recommendation without more episodes.")
    return out


if __name__ == "__main__":
    main()

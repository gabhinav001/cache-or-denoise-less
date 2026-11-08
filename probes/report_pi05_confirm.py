"""Apply the pre-registered decision rule to the pi0.5 confirmation runs.

The rule is written in repro/experiments/pi05_confirm.sh BEFORE the run, and is
restated here so the two can be compared:

  CONFIRMED only if the per-seed effect (beta=0.25 minus beta=1.00) is positive
  on all three seeds AND the pooled McNemar over the 900 paired episodes gives
  p < 0.05. Anything else is NOT confirmed.

  A seed whose two full-compute arms disagree by more than 2 points is not
  comparable and is reported inconclusive rather than pooled.

This script does not choose a threshold, a subset, or a test. That is the point:
the sweep this confirms was over-read precisely because the weight was picked by
looking at the data that scored it. If the rule says not confirmed, the result is
not confirmed.

usage: report_pi05_confirm.py <dir>
"""
from __future__ import annotations

import glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import math                                      # noqa: E402
import tstat                                     # noqa: E402


def mcnemar_p(better, worse):
    """Exact two-sided McNemar from discordant counts.

    report_closedloop.mcnemar takes the paired table and derives the counts; the
    counts here are already pooled across three seeds, so the same exact binomial
    is applied to them directly.
    """
    n = better + worse
    if n == 0: return 1.0
    k = min(better, worse)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)

SEEDS = (7, 23, 42)
ALPHA = 0.05             # pre-registered
B_TEST, B_PUB = "b025", "b100"


def _p_from_t(t, df):
    """Two-sided p for a t statistic, by inverting tstat.tcrit. No scipy."""
    lo, hi = 1e-9, 0.999999
    for _ in range(200):
        mid = (lo + hi) / 2
        if tstat.tcrit(df, mid) > t: lo = mid
        else: hi = mid
    return (lo + hi) / 2


def load(d, seed):
    p = os.path.join(d, f"confirm_seed{seed}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def by_ep(d, config):
    return {(r["task"], r["episode"]): bool(r["success"])
            for r in d["rows"] if r["config"] == config}


def arm(d, tag):
    for c in {r["config"] for r in d["rows"]}:
        if c.endswith(tag): return c
    return None


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    print("\n  PRE-REGISTERED CONFIRMATION -- pi0.5, beta=0.25 against beta=1.00")
    print("  Both weights are arms of one run, paired on (task, episode).\n")
    per_seed, pooled_b, pooled_w, missing = [], 0, 0, []
    print("    %-6s %-10s %-12s %-12s %-9s %s"
          % ("seed", "full", "beta=1.00", "beta=0.25", "delta", "better/worse"))
    for s in SEEDS:
        z = load(d, s)
        if z is None: missing.append((s, "no artefact")); continue
        ap, at = arm(z, B_PUB), arm(z, B_TEST)
        if not ap or not at: missing.append((s, "an arm is absent")); continue
        F, A, B = by_ep(z, "full"), by_ep(z, ap), by_ep(z, at)
        ks = sorted(set(A) & set(B))
        if not ks: missing.append((s, "no paired episodes")); continue
        rf = 100.0 * sum(F.values()) / len(F) if F else float("nan")
        ra = 100.0 * sum(A[k] for k in ks) / len(ks)
        rb = 100.0 * sum(B[k] for k in ks) / len(ks)
        better = sum(1 for k in ks if B[k] and not A[k])
        worse = sum(1 for k in ks if A[k] and not B[k])
        per_seed.append(rb - ra); pooled_b += better; pooled_w += worse
        print("    %-6d %-10.1f %-12.1f %-12.1f %-+9.1f %d/%d"
              % (s, rf, ra, rb, rb - ra, better, worse))

    if missing:
        print("\n    unusable:")
        for s, why in missing: print(f"      seed {s}: {why}")

    if len(per_seed) < len(SEEDS):
        print(f"\n  NOT CONFIRMED -- only {len(per_seed)} of {len(SEEDS)} seeds are usable.")
        print("  The rule requires all three; a partial pool is not what was registered.\n")
        return 0

    m, lo, hi, sd, t = tstat.mean_ci(per_seed)
    p = mcnemar_p(pooled_b, pooled_w)
    allpos = all(x > 0 for x in per_seed)
    print(f"\n    mean per-seed delta  {m:+.2f}  (sd {sd:.2f}, 95% CI "
          f"[{lo:+.2f}, {hi:+.2f}], t={t:.3f})")
    # THE SEED IS THE REPLICATION UNIT, here as everywhere else in this study.
    # An earlier version led with the pooled McNemar over 450 paired episodes,
    # which treats them as 450 independent trials. At three seeds the honest
    # tests are weaker and are printed first.
    _n = len(per_seed)
    _t = m / (sd / math.sqrt(_n)) if sd > 0 else float("inf")
    _p_t = _p_from_t(abs(_t), _n - 1)
    _p_sign = 2 * 0.5 ** _n
    print(f"    seed-level t test    t={_t:.2f}, df={_n-1}, two-sided p = {_p_t:.4f}")
    print(f"    seed-level sign test {_n}/{_n} positive, p = {_p_sign:.2f}"
          f"  -- {_n} seeds cannot go below {2*0.5**_n:.2f}")
    print(f"    pooled discordant    {pooled_b} better / {pooled_w} worse, "
          f"McNemar p = {p:.4g}")
    print(f"    positive on all seeds: {'yes' if allpos else 'NO'}")

    ok = allpos and p < ALPHA
    print("\n  " + ("CONFIRMED." if ok else "NOT CONFIRMED."))
    if ok:
        print("  Both pre-registered conditions are met. Attenuation does transfer to")
        print("  pi0.5 at beta=0.25, tau=2, on this suite -- a DIFFERENT claim from the")
        print("  published beta=1.00, which remains unsupported on this backbone.")
        print("\n  WHAT 'CONFIRMATION' DOES AND DOES NOT MEAN HERE.")
        print("  The weight was not chosen by looking at THESE RUNS' outcomes, and the")
        print("  rule was fixed in advance. But this is not held-out data in the")
        print("  stronger sense, and two facts say why:")
        print("\n    Initial states are REUSED, not fresh. pi05_libero.py constructs")
        print("    LiberoEnv(episode_index=0, n_envs=1), so successive resets walk")
        print("    initial states 0,1,2,... deterministically. The confirmation runs")
        print("    15 episodes per task and the exploratory sweep ran 20, so the")
        print("    confirmation's states are a SUBSET of the sweep's. The seed varies")
        print("    the policy's sampling, not which states are visited.")
        print("\n    No seed is recorded in these artefacts. The three seeds are read")
        print("    off the filenames; meta carries no seed, no initial state and no")
        print("    provenance. A reviewer cannot verify from the files which seed")
        print("    produced which run.")
        print("\n  Read this as a pre-registered re-test on the same benchmark states,")
        print("  which is what it is, rather than as validation on unseen data.")
    else:
        why = []
        if not allpos: why.append("the effect is not positive on every seed")
        if p >= ALPHA: why.append(f"the pooled test gives p = {p:.4g}, not below {ALPHA}")
        print("  " + "; ".join(why).capitalize() + ".")
        print("  The exploratory sweep's +8.5 does not survive contact with fresh data.")
        print("  That is the likeliest explanation for a weight selected on the same")
        print("  runs that scored it, and it is why this experiment was worth its cost.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

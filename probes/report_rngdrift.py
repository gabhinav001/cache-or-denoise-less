"""Name the random stream that carried the arm-order drift.

Reads the four artefacts written by repro/experiments/rngdrift.sh and answers two
questions the repository has so far only been able to state as open:

  1. Does the per-episode reseed actually remove the arm-order dependence?
     With the reseed on, the same arm measured in a 3-arm and a 5-arm run must
     produce identical outcomes and identical step counts.

  2. Which stream was carrying it? With the reseed off, each episode's
     fingerprint of the four global streams is compared across the two arm
     lists. A stream that agrees under both conditions was never involved; a
     stream that agrees only when the reseed is on is the carrier.

usage: report_rngdrift.py <dir>
"""
from __future__ import annotations

import json, math, os, sys

STREAMS = ("torch_cpu", "torch_cuda", "numpy", "python")


def load(d, name):
    p = os.path.join(d, name + ".json")
    if not os.path.exists(p): return None
    return json.load(open(p))


def by_key(d):
    """(config, task, episode) -> row."""
    return {(r["config"], r["task"], r["episode"]): r for r in d["rows"]}


def shared_arms(a, b):
    return sorted({c for c, _, _ in a} & {c for c, _, _ in b})


def mcnemar_p(n01, n10):
    n = n01 + n10
    if n == 0: return 1.0
    k = min(n01, n10)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def compare(a, b, label):
    """Agreement for every arm both runs share.

    Reports exact agreement AND a systematic-difference test, because the two
    answer different questions. Exact step agreement is the strong claim: the
    original defect was found by noticing the first arm was bit-identical while
    later ones were not, so bit-identity is achievable here and its absence is
    informative. Absence of exact agreement alone does not isolate the cause.
    The latency experiment records divergence, but does not establish either
    compiler causality or distributional equivalence. Report a paired success
    test separately without interpreting non-rejection as equivalence.

    Returns (exact step agreement, worst systematic p-value).
    """
    print(f"\n  {label}")
    print("    %-22s %-10s %-12s %-10s %s"
          % ("arm", "outcomes", "step counts", "discordant", "position"))
    worst, worst_p = 1.0, 1.0
    for arm in shared_arms(a, b):
        ks = [k for k in a if k[0] == arm and k in b]
        if not ks: continue
        same_out = sum(1 for k in ks if a[k]["success"] == b[k]["success"])
        same_stp = sum(1 for k in ks if a[k]["steps"] == b[k]["steps"])
        n01 = sum(1 for k in ks if not a[k]["success"] and b[k]["success"])
        n10 = sum(1 for k in ks if a[k]["success"] and not b[k]["success"])
        p = mcnemar_p(n01, n10)
        # by_key preserves the episode rows' first-appearance order. Sorting
        # names reports alphabetical positions instead of execution positions.
        pa = list(dict.fromkeys(c for c, _, _ in a)).index(arm)
        pb = list(dict.fromkeys(c for c, _, _ in b)).index(arm)
        worst = min(worst, same_stp / len(ks)); worst_p = min(worst_p, p)
        print("    %-22s %3d/%-6d %3d/%-8d %d/%-7d %d vs %d%s"
              % (arm, same_out, len(ks), same_stp, len(ks), n01, n10, pa, pb,
                 "" if same_stp == len(ks) else
                 ("   <-- SYSTEMATIC" if p < 0.05 else "   <-- differs, not systematic")))
    return worst, worst_p


def stream_agreement(a, b):
    """Per-stream share of episodes whose fingerprint matches across arm lists."""
    out = {}
    for s in STREAMS:
        ks = [k for k in a if k in b and s in (a[k].get("rng") or {})
              and s in (b[k].get("rng") or {})]
        if not ks: continue
        out[s] = sum(1 for k in ks if a[k]["rng"][s] == b[k]["rng"][s]) / len(ks)
    return out


def rng_matters(a, b, label):
    """Does the global RNG affect these rollouts AT ALL?

    This is the test the first version of this script was missing, and without it
    the conclusion came out backwards. Comparing the same arm list with the
    reseed on against the same arm list with it off isolates one variable: the
    global random state. If outcomes and step counts are identical while the
    fingerprints differ, then nothing in the rollout consumes that state, and no
    amount of reseeding can be what fixes -- or fails to fix -- arm-order
    dependence.

    Returns True if the RNG demonstrably matters.
    """
    ks = sorted(set(a) & set(b))
    if not ks: return None
    so = sum(1 for k in ks if a[k]["success"] == b[k]["success"])
    ss = sum(1 for k in ks if a[k]["steps"] == b[k]["steps"])
    fp = sum(1 for k in ks
             if (a[k].get("rng") or {}).get("torch_cpu")
             == (b[k].get("rng") or {}).get("torch_cpu"))
    print("    %-10s outcomes %3d/%-4d  steps %3d/%-4d  torch fingerprint %3d/%-4d"
          % (label, so, len(ks), ss, len(ks), fp, len(ks)))
    return ss != len(ks)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    runs = {n: load(d, n) for n in
            ("arms3_reseed", "arms5_reseed", "arms3_drift", "arms5_drift")}
    missing = [n for n, v in runs.items() if v is None]
    if missing:
        print(f"  missing artefacts: {', '.join(missing)}"); return 1
    R = {n: by_key(v) for n, v in runs.items()}

    print("\n  DOES THE GLOBAL RNG AFFECT THESE ROLLOUTS AT ALL?")
    print("  Same arm list, reseed on against reseed off -- one variable changed.\n")
    m3 = rng_matters(R["arms3_reseed"], R["arms3_drift"], "3-arm")
    m5 = rng_matters(R["arms5_reseed"], R["arms5_drift"], "5-arm")
    rng_live = bool(m3) or bool(m5)
    if not rng_live:
        print("\n    Outcomes and step counts match while global RNG fingerprints differ.")
        print("    These controls detect no effect from global reseeding. They do not")
        print("    establish arm-order independence or identify the source of drift.")
    else:
        print("\n    Outcomes move when only the global random state moves, so the")
        print("    stream table below is meaningful.")

    print("\n\n  DOES ARM POSITION STILL CHANGE RESULTS?")
    print("  3-arm run against 5-arm run, for the arms they share.\n")
    ok, ok_p = compare(R["arms3_reseed"], R["arms5_reseed"], "reseed ON")
    bad, bad_p = compare(R["arms3_drift"], R["arms5_drift"], "reseed OFF")

    print("\n\n  READING\n")
    if not rng_live:
        print("    The recorded arm-list dependence persists with reseeding on and off.")
        print("    Global reseeding changes fingerprints but not these rollout results,")
        print("    so it does not explain or repair the observed dependence.")
        print("    Compiler state, numerical differences, or other order-dependent")
        print("    runtime state remain candidates; these controls do not isolate them.")
        print("    Compare identical-input outputs with compilation on/off and run")
        print("    counterbalanced arms or fresh processes before assigning a cause.")
        if ok_p < 0.05 or bad_p < 0.05:
            print()
            print("    NOTE: at least one arm shows a SYSTEMATIC outcome difference by")
            print("    arm position. This needs investigation; the test does not distinguish")
            print("    numerical effects from a semantic implementation error.")
    else:
        if ok >= 0.999:
            print("    With the reseed on every shared arm is identical across arm lists.")
        elif ok_p >= 0.05:
            print(f"    Step counts match on {ok:.1%} of episodes but no arm differs")
            print("    systematically.")
        else:
            print(f"    Arms still disagree systematically (p={ok_p:.3g}); the reseed does")
            print("    not remove the dependence.")

    print("\n\n  FINGERPRINT AGREEMENT ACROSS ARM LISTS (diagnostic only)\n")
    on = stream_agreement(R["arms3_reseed"], R["arms5_reseed"])
    off = stream_agreement(R["arms3_drift"], R["arms5_drift"])
    print("    %-12s %-14s %s" % ("stream", "reseed ON", "reseed OFF"))
    for s in STREAMS:
        if s in on and s in off:
            print("    %-12s %-14.3f %.3f" % (s, on[s], off[s]))
    print("\n    These say which streams the reseed controls, NOT which stream causes")
    print("    anything. Reading them as a cause is the error the first version of this")
    print("    script made.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

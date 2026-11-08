"""The pi0.5 substitution-weight sweep, tested the way everything else here is.

The sweep was read by comparing marginal success rates across its five runs --
76.5, 86.5, 78.0, 80.0, 78.0 -- and the 10-point spread was attributed to seed
noise, which made it independent evidence that the effect does not transfer to
this backbone. Two things are wrong with that reading.

The seed-noise figure it leans on comes from a different comparison. The full-
compute arms across these same five runs span 0.5 points, not 10, so whatever
moves the cached arm here is not the between-seed movement measured elsewhere.

And it is the only comparison in the study not paired. Every other one is paired
on (task, episode) and tested with McNemar, including the staleness result, which
pairs across separate artefact files exactly as these five runs allow. Applying
the study's own test to its own data reverses the conclusion: beta=0.25 beats
beta=1.00 by 8.5 points, 22 episodes better against 5 worse, p=0.0015 on 200
paired episodes, and p=0.0076 after correcting for the five weights tested.

What this does and does not establish matters. It does NOT reinstate the claim
that attenuation transfers to pi0.5 -- the closed-loop runs still disagree in
sign across seeds (-0.3 at one, +9.0 at another), and that disagreement is the
reason the transfer claim was withdrawn. It does mean the sweep cannot be cited
as a second, independent null. It is not a null.

usage: report_pi05_sweep.py <canonical_numbers.json>
"""
from __future__ import annotations

import json, sys


def main(path):
    try:
        d = json.load(open(path)).get("pi05_beta_sweep")
    except Exception as e:
        print(f"    could not read {path}: {e}")
        return 1
    if not d:
        print("    (pi05_beta_sweep absent -- run probes/canonical_numbers.py first)")
        return 1
    print("    %-6s %-7s %-8s %-9s %-13s %s"
          % ("beta", "full", "cached", "delta", "better/worse", "p"))
    for r in d["per_beta"]:
        print("    %-6.2f %-7.1f %-8.1f %-+9.1f %-13s %s"
              % (r["beta"], r["full"], r["cached"], r["delta_vs_beta1"],
                 "%d/%d" % (r["better"], r["worse"]), r["p"]))
    b = d["best"]
    print()
    print("    best: beta=%.2f, %+.1f points, %d better / %d worse, p=%s"
          % (b["beta"], b["delta_vs_beta1"], b["better"], b["worse"], b["p"]))
    print("    Bonferroni over the %d weights tested: p=%s"
          % (len(d["per_beta"]), d["bonferroni_p"]))
    print("    Full-compute arms span %.1f points across these runs, so the"
          % d["full_compute_spread"])
    print("    seed-to-seed movement cited to dismiss this is a different comparison.")
    print()
    print("    This does NOT reinstate the pi0.5 transfer claim: the closed-loop runs")
    print("    still disagree in sign across seeds. It means the sweep cannot be cited")
    print("    as independent evidence of a null -- it is not one.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "results/canonical_numbers.json"))

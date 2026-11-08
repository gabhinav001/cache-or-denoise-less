#!/usr/bin/env python3
"""Does a tag select the same inputs wherever it appears in --seeds?

An experiment that maps tags to checkpoints and episode lists has to do it with
pure functions of the tag. Assigning to a shared variable inside the per-tag
loop works when you run one tag and corrupts the next one when you run several:
`mechanism --seeds "idm7 primary"` collected fw_families.npz -- the primary
geometry artefact the paper reports -- from the 24-episode idm list, because
MANIFEST was set for idm7 and never cleared. The run looked normal and the
artefact was mislabelled.

This drives the real dispatcher with DRY_RUN and checks that each tag resolves
to the same checkpoint, manifest and output name in every ordering, including
orderings chosen to expose leakage. It needs no GPU, no checkpoint and no video.

usage: check_tag_purity.py
"""
from __future__ import annotations

import itertools, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RUN = os.path.join(ROOT, "repro", "run_experiment.sh")

# experiment -> tags whose inputs must not depend on ordering. mechanism is the
# one where a tag selects a checkpoint and an episode list; the rest take numeric
# seeds that legitimately change only the output suffix, and are included so a
# future tag-dependent input in any of them is covered the day it is added.
CASES = {"mechanism": ["primary", "repro", "idm", "idm7", "repeat16"],
         "statescale": ["42", "7"],
         "deployfrontier": ["42", "11", "7"],
         "matrix": ["42", "7"],
         "cachesteps": ["42", "7"],
         "baselines": ["42", "7"]}
FIELD = re.compile(r"(?:ckpt=|\+PROBE\.manifest=|\+PROBE\.out=)(\S+)")


def resolve(exp, seeds):
    """Run the dispatcher dry and return {tag: (ckpt, manifest, out)}."""
    env = dict(os.environ, DRY_RUN="1", CKPT_DIR="/nonexistent-ckpt-dir")
    r = subprocess.run(["bash", RUN, exp, "--seeds", " ".join(seeds)],
                       capture_output=True, text=True, env=env, cwd=ROOT)
    if r.returncode:
        raise SystemExit("  dry run failed for %s --seeds %r\n%s"
                         % (exp, " ".join(seeds), r.stderr[-800:]))
    rows = [tuple(os.path.basename(m) for m in FIELD.findall(line))
            for line in r.stdout.splitlines() if "+PROBE.out=" in line]
    if len(rows) != len(seeds):
        raise SystemExit("  expected %d invocations, parsed %d -- the dry-run "
                         "format changed; fix this check" % (len(seeds), len(rows)))
    return dict(zip(seeds, rows))


def main():
    bad = []
    for exp, tags in CASES.items():
        # every tag alone is the reference; then orderings that would expose a
        # value surviving from the previous iteration
        ref = {t: resolve(exp, [t])[t] for t in tags}
        orders = [tags, tags[::-1]]
        orders += [list(p) for p in itertools.permutations(tags, 2)]
        orders = [o for o in orders if len(o) > 1]
        seen = 0
        for order in orders:
            got = resolve(exp, order)
            seen += 1
            for t, row in got.items():
                if row != ref[t]:
                    bad.append((exp, t, " ".join(order), ref[t], row))
        print("  %s: %d tags, %d orderings" % (exp, len(tags), seen))
        for t in tags:
            print("    %-9s %s" % (t, "  ".join(ref[t])))

    if bad:
        print("\n  NOT PURE -- a tag's inputs depend on what ran before it:\n")
        for exp, t, order, want, got in bad[:10]:
            print("    %s --seeds %r" % (exp, order))
            print("      %s alone: %s" % (t, "  ".join(want)))
            print("      %s here : %s" % (t, "  ".join(got)))
        print("\n  Resolve each input as a function of the tag; do not assign to a")
        print("  shared variable inside the per-tag loop.")
        return 1
    print("\n  every tag resolves to the same inputs in every ordering\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

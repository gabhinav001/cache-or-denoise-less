"""Report direct pairwise errors from checkpoint-loaded stack diagnostics.

Legacy artifacts contain only distances to full compute and were collected
without loading the checkpoint. They are preserved but excluded from reporting.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

from stack_probe_common import validate_artifact


def main(path):
    if not os.path.exists(path):
        print(f"SKIP: no artifact at {path}")
        return 0
    with open(path) as fh:
        data = json.load(fh)
    try:
        validate_artifact(data, "stackdiff")
    except ValueError as exc:
        print(str(exc))
        return 2
    meta, rows, comparisons = data["meta"], data.get("rows", []), data.get("comparisons", [])
    if not comparisons:
        print("EXCLUDED: no direct pairwise tensor measurements were recorded; rerun stackdiff.")
        return 2
    expected_hits = len(meta["cached_steps"])
    bad = [r for r in rows if r["arm"].startswith("wholestack")
           and (r["stack_hits"] != expected_hits or r["stack_misses"])]
    if bad:
        print(f"EXCLUDED: {len(bad)} whole-stack observations have incomplete substitution.")
        return 2
    print(f"{os.path.relpath(path)}: {meta['steps']} denoising steps, "
          f"cached steps {meta['cached_steps']}")
    print(f"Loaded checkpoint: {meta['checkpoint_load']['name']} "
          f"(SHA256 {meta['checkpoint_load']['sha256']})")
    grouped = defaultdict(list)
    for row in comparisons:
        grouped[(bool(row["compiled"]), int(row["age"]), row["left"], row["right"])].append(row)
    print("Maximum across observations of DIRECT pairwise RMS/max errors:")
    for (compiled, age, left, right), group in sorted(grouped.items()):
        floor = [r["action_rms"] for r in rows if r["arm"] == "full_repeat"
                 and bool(r["compiled"]) == compiled and int(r["age"]) == age]
        if not floor:
            print("EXCLUDED: a matching repeated-full control is absent.")
            return 2
        print(f"  compiled={int(compiled)} age={age} {left} vs {right}: "
              f"action RMS={max(r['action_rms'] for r in group):.6g}, "
              f"action max={max(r['action_max'] for r in group):.6g}, "
              f"hidden RMS={max(r['hidden_rms'] for r in group):.6g}; "
              f"repeat-full RMS floor={max(floor):.6g} (n={len(group)})")
    print("These fixed-observation errors do not establish closed-loop equivalence "
          "or explain a task-success gap. wholestack32 changes only the final addition, "
          "using the same already-rounded cached residual.")
    return 0


if __name__ == "__main__":
    default = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "results", "stackdiff", "fw_stackdiff.json")
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else default))

"""Make the post-fix frontier collection the published one.

The frontier figures have been quoted from runs collected before the per-episode
seeding fix (A8), because until now there was no complete post-fix set to quote
instead. This performs the swap once all five seeds are in, and refuses to do it
early -- a partial pool is exactly what the completeness guard exists to prevent,
and a partial pool that happens to look favourable is worse than one that does
not.

What it does, in order, and only if every check passes:

  results/frontier/     -> results/frontier_v1_prefix/   (kept, not deleted:
                           the two generations are the evidence for what the
                           pairing defect cost, and A8 cites the comparison)
  results/frontier_v2/  -> results/frontier/             (becomes canonical, so
                           canonical_numbers.py picks it up with no change)

Checks before moving anything:
  - five complete seeds in frontier_v2, by the same grid test the analysis uses
  - every one of them differs from its pre-fix namesake, because a machine that
    never received the fix reproduces the old run exactly and two seeds already
    did that once here
  - the destination names are free

usage: promote_post_fix.py [--dry-run]
"""
from __future__ import annotations

import glob, json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
EXPECTED_SEEDS = {"42", "7", "11", "23", "3"}


def cells(path):
    d = json.load(open(path))
    return {(r["task"], r["episode"], r["config"]): (r["success"], r.get("steps"))
            for r in d["rows"]}


def complete(path):
    """Rectangular and fully covered -- same test canonical_numbers.py applies."""
    d = json.load(open(path)); rows = d.get("rows") or []
    if not rows: return False
    n_tasks = d.get("meta", {}).get("n_tasks")
    grid = {}
    for r in rows:
        grid.setdefault((r["task"], r["episode"]), set()).add(r["config"])
    tasks = {t for t, _ in grid}
    if n_tasks and len(tasks) < n_tasks: return False
    per_task = {t: sum(1 for tt, _ in grid if tt == t) for t in tasks}
    return len(set(per_task.values())) == 1 and len({frozenset(v) for v in grid.values()}) == 1


def main(dry):
    src = os.path.join(R, "frontier_v2")
    cur = os.path.join(R, "frontier")
    dst = os.path.join(R, "frontier_v1_prefix")
    problems = []

    files = sorted(glob.glob(os.path.join(src, "frontier_seed*.json")))
    seeds = {os.path.basename(f)[13:-5] for f in files}
    if seeds != EXPECTED_SEEDS:
        problems.append(f"have seeds {sorted(seeds)}, need {sorted(EXPECTED_SEEDS)}")

    for f in files:
        s = os.path.basename(f)[13:-5]
        if not complete(f):
            problems.append(f"seed {s} is not a complete grid")
            continue
        old = os.path.join(cur, os.path.basename(f))
        if not os.path.exists(old):
            problems.append(f"seed {s} has no pre-fix counterpart to compare")
            continue
        a, b = cells(f), cells(old)
        shared = set(a) & set(b)
        same = sum(1 for k in shared if a[k] == b[k])
        if shared and same == len(shared):
            problems.append(f"seed {s} is IDENTICAL to the pre-fix run -- that box "
                            f"never received the fix; do not publish it as post-fix")
        else:
            print(f"  seed {s:3s} complete, differs from pre-fix "
                  f"({100*same/max(len(shared),1):.1f}% of outcomes identical)")

    if os.path.exists(dst):
        problems.append(f"{dst} already exists")

    if problems:
        print("\n  NOT promoting:")
        for p in problems: print(f"    - {p}")
        return 1

    print(f"\n  all {len(EXPECTED_SEEDS)} seeds complete and genuinely post-fix")
    if dry:
        print("  --dry-run: no files moved")
        return 0
    shutil.move(cur, dst)
    shutil.move(src, cur)
    print(f"  moved {os.path.relpath(dst, R)} <- the pre-fix collection, kept for A8")
    print(f"  moved {os.path.relpath(cur, R)} <- now the post-fix collection")
    print("\n  Re-run probes/canonical_numbers.py and repro/verify.sh, and update any")
    print("  prose that describes the frontier rows as pre-fix.")
    return 0


if __name__ == "__main__":
    sys.exit(main("--dry-run" in sys.argv))

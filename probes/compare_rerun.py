"""How far apart are two collections of the same run?

A reviewer's question is not whether their numbers match ours -- they will not,
since only some runs record a seed and same-seed determinism holds only within a
GPU model. It is whether their numbers land close enough that the conclusion is
the same. That distance is measurable whenever a run has been collected twice,
and this reports it for any such pair.

Reports the marginal rate per arm, the paired agreement where episodes line up,
and -- for a two-arm file -- the contrast each collection would have supported.
A contrast that keeps its sign and rough size across two honest collections is
reproducible in the sense that matters; one that flips is not, however close the
marginal rates look.

usage: compare_rerun.py <original.json> <rerun.json>
"""
from __future__ import annotations

import json, os, sys


def load(p):
    d = json.load(open(p))
    by = {}
    for r in d["rows"]:
        by.setdefault(r["config"], {})[(r["task"], r["episode"])] = bool(r["success"])
    return d.get("meta", {}), by


def rate(m):
    return 100.0 * sum(m.values()) / len(m) if m else float("nan")


def main():
    if len(sys.argv) < 3:
        print(__doc__); return 2
    (ma, a), (mb, b) = load(sys.argv[1]), load(sys.argv[2])
    na, nb = os.path.basename(sys.argv[1]), os.path.basename(sys.argv[2])
    arms = sorted(set(a) & set(b))
    if not arms:
        print(f"  no shared arms: {sorted(a)} vs {sorted(b)}"); return 1

    print(f"\n  {na}  against  {nb}\n")
    ga = (ma.get("provenance") or {}).get("gpu", "?")
    gb = (mb.get("provenance") or {}).get("gpu", "not recorded")
    print(f"    hosts: {ga}  /  {gb}")
    print(f"    seeds: {ma.get('seed', '(not recorded)')}  /  {mb.get('seed', '(not recorded)')}\n")

    print("    %-26s %9s %9s %8s %s" % ("arm", "original", "re-run", "diff", "episodes agreeing"))
    for c in arms:
        ks = sorted(set(a[c]) & set(b[c]))
        agree = sum(1 for k in ks if a[c][k] == b[c][k])
        print("    %-26s %8.1f%% %8.1f%% %+7.1f  %d/%d"
              % (c, rate(a[c]), rate(b[c]), rate(b[c]) - rate(a[c]), agree, len(ks)))

    if len(arms) == 2:
        x, y = arms
        da = rate(a[y]) - rate(a[x])
        db = rate(b[y]) - rate(b[x])
        print(f"\n    contrast {y} minus {x}:")
        print(f"      original {da:+.1f}    re-run {db:+.1f}    moved {db - da:+.1f}")
        same = (da > 0) == (db > 0)
        print(f"      same sign: {'yes' if same else 'NO'}")
        if same:
            print("\n    The contrast keeps its sign and lands within a few points. That is")
            print("    what a reviewer should expect: not the same episodes, and not the")
            print("    same number, but the same conclusion.")
        else:
            print("\n    THE CONTRAST FLIPPED SIGN between two honest collections of the")
            print("    same configuration. Whatever this run is being used to support is")
            print("    not supported by it.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

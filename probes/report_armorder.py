"""Is one cached arm's rollout invariant to its POSITION and to compilation?

rngdrift's original four runs vary arm-list LENGTH with the target arm in a
fixed position. That cannot separate "the list is longer" from "the arm ran
later", and it says nothing about compilation. The audit needed both excluded:
a cached arm changed 3 of 40 outcomes and 36 of 40 episode lengths between
collections that already had the per-episode reseed on.

The six ao_* runs put the SAME arm alone, first, and last in a reversed list,
with compilation off and on, each in a fresh process. This compares the target
arm's rows across them, keyed on (task, episode).

  identical everywhere            position and compilation are both excluded
  differs only across compile     compilation is the carrier
  differs across position         arm isolation is required for every
                                  multi-arm collection in this repository

usage: report_armorder.py [results/rngdrift]
"""
from __future__ import annotations

import json, os, sys
from collections import OrderedDict

TARGET = "c3ache_tau8_shrink"
RUNS = OrderedDict([("ao_solo_nc", ("alone", 0)), ("ao_first_nc", ("first", 0)),
                    ("ao_last_nc", ("last", 0)), ("ao_solo_c", ("alone", 1)),
                    ("ao_first_c", ("first", 1)), ("ao_last_c", ("last", 1))])


def load(d, name):
    p = os.path.join(d, name + ".json")
    if not os.path.exists(p): return None
    rows = json.load(open(p))["rows"]
    out = {}
    for r in rows:
        if r.get("config") != TARGET: continue
        out[(r["task"], r["episode"])] = (bool(r["success"]), int(r["steps"]))
    return out


def main(d):
    got = {k: load(d, k) for k in RUNS}
    missing = [k for k, v in got.items() if v is None]
    if missing:
        print(f"  SKIP  {len(missing)} of {len(RUNS)} arm-order runs absent "
              f"({', '.join(missing)})")
        print("  Collect with: ./repro/run_experiment.sh rngdrift")
        return 0
    empty = [k for k, v in got.items() if not v]
    if empty:
        print(f"  {', '.join(empty)} contain no rows for {TARGET!r}; the target "
              "arm label changed and this check is comparing nothing.")
        return 1

    ref_name = "ao_solo_nc"
    ref = got[ref_name]
    print(f"\n  target arm {TARGET}, {len(ref)} episodes, reference {ref_name}\n")
    print(f"    {'run':<14} {'position':<9} {'compiled':>8} {'outcome match':>14} "
          f"{'steps match':>12}")
    diffs = {}
    for name, (pos, comp) in RUNS.items():
        g = got[name]
        keys = sorted(set(ref) & set(g))
        so = sum(1 for k in keys if ref[k][0] == g[k][0])
        st = sum(1 for k in keys if ref[k][1] == g[k][1])
        diffs[name] = (len(keys) - so, len(keys) - st)
        print(f"    {name:<14} {pos:<9} {comp:>8} {so:>7}/{len(keys):<6} "
              f"{st:>6}/{len(keys):<5}")

    pos_moved = any(diffs[n] != (0, 0) for n in ("ao_first_nc", "ao_last_nc"))
    cmp_moved = diffs["ao_solo_c"] != (0, 0)
    print()
    if not pos_moved and not cmp_moved:
        print("  The target arm is bit-identical alone, first and last, with and")
        print("  without compilation. Arm position and compilation are excluded as")
        print("  explanations for the cached-arm gap.")
        return 0
    if pos_moved:
        print("  POSITION MOVES THE RESULT. Every multi-arm collection in this")
        print("  repository is affected, including the headline: arms must be run")
        print("  in fresh single-arm processes before any of them is compared.")
    if cmp_moved:
        print("  COMPILATION MOVES THE RESULT. The compiled and eager paths do not")
        print("  agree, so timing runs and quality runs describe different policies.")
    return 1


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else os.path.join(here, "..", "results", "rngdrift")))

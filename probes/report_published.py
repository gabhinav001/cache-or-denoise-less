"""Describe local LIBERO Goal collections beside published reference values.

Each input file is a separate collection, never pooled across seeds, hosts,
revisions, or arm orders. A complete collection requires all three arms on the
same 10 tasks x 50 distinct initial states; smaller runs remain smoke/subset
measurements. Neither matching point estimates nor interval overlap establishes
reference checkpoint, protocol, implementation, or performance equivalence.

usage: report_published.py [results/published]
"""
from __future__ import annotations

import glob, json, math, os, sys
from collections import defaultdict

REF = {"FastWAM README, IDM": 98.6, "FastWAM README, first-frame": 97.8,
       "C3ache Table 1 baseline": 96.4, "C3ache [0,7] tau=8": 96.4}
ARMS = ["full", "whole_tau8", "c3ache_tau8"]
WHAT = {"full": "local full compute",
        "whole_tau8": "local whole-stack implementation",
        "c3ache_tau8": "local per-block implementation"}


def wilson(k, n, z=1.96):
    """Score interval -- the normal approximation misbehaves near 1.0, which is
    exactly where every arm here sits."""
    if n == 0: return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def collection(path):
    """Read one rollout file; unrelated gates/diagnostics return None."""
    with open(path) as f:
        j = json.load(f)
    if not isinstance(j, dict): return None
    meta, rows = j.get("meta") or {}, j.get("rows")
    if not isinstance(rows, list) or not rows or meta.get("suite") != "libero_goal":
        return None
    if not all(isinstance(r, dict) and r.get("config") in ARMS for r in rows):
        return None
    by = defaultdict(dict)
    for r in rows:
        if r.get("suite", meta["suite"]) != meta["suite"]:
            raise ValueError("row suite differs from collection metadata")
        if type(r.get("success")) is not bool:
            raise ValueError("success must be a recorded boolean")
        key = (r["task"], r["episode"])
        if key in by[r["config"]]:
            raise ValueError(f"duplicate episode {key} in {r['config']}")
        by[r["config"]][key] = r
    return meta, by


def report_collection(path, meta, by):
    expected = {(t, e) for t in range(10) for e in range(50)}
    complete = (meta.get("n_tasks") == 10 and meta.get("n_episodes") == 50
                and all(set(by.get(a, {})) == expected for a in ARMS))
    if complete:
        for arm in ARMS:
            hashes = [r.get("init_state") for r in by[arm].values()]
            complete = complete and all(hashes) and len(set(hashes)) == 500
        complete = complete and all(
            len({by[a][key].get("init_state") for a in ARMS}) == 1 for key in expected)

    pv = meta.get("provenance") or {}
    print(f"\n  Collection: {os.path.basename(path)}")
    print(f"  Seed {meta.get('seed', 'not recorded')}; "
          f"host {pv.get('host_tag') or 'not recorded'}; "
          f"revision {pv.get('commit') or 'not recorded'}")
    print(f"  Declared LIBERO Goal protocol: {meta.get('n_tasks')} tasks x "
          f"{meta.get('n_episodes')} episodes; actual rows {sum(map(len, by.values()))}")
    if complete:
        print("  COMPLETE LOCAL COLLECTION: all three arms on the same 500 distinct states.")
    else:
        print("  SMOKE / SUBSET / INCOMPLETE: this is not a complete 10-task x 50-state comparison.")
    print("  Completeness describes coverage only, not successful reproduction.")

    print(f"\n    {'arm':<14} {'what':<34} {'success':>9} {'Wilson 95%':>16}  n")
    for arm in ARMS:
        b = by.get(arm)
        if not b: continue
        k, n = sum(r["success"] for r in b.values()), len(b)
        pct = 100 * k / n
        lo, hi = wilson(k, n)
        print(f"    {arm:<14} {WHAT.get(arm,''):<34} {pct:8.1f}% "
              f"[{lo:5.1f},{hi:5.1f}]  {n}")

    print("\n  Published reference points (checkpoint and protocol matching unverified)")
    for name, v in REF.items():
        print(f"    {name:<30} {v:5.1f}")

    if "whole_tau8" in by and "c3ache_tau8" in by:
        keys = sorted(set(by["whole_tau8"]) & set(by["c3ache_tau8"]))
        matched = [x for x in keys if by["whole_tau8"][x].get("init_state")
                   and by["whole_tau8"][x]["init_state"] == by["c3ache_tau8"][x].get("init_state")]
        if matched and len(matched) == len(keys):
            diff = 100 * sum(by["whole_tau8"][x]["success"] - by["c3ache_tau8"][x]["success"]
                             for x in matched) / len(matched)
            print(f"\n  Local whole-stack minus per-block, paired on {len(matched)} "
                  f"verified initial states: {diff:+.1f} points")
        else:
            print("\n  Paired cache contrast omitted: missing or mismatched initial-state hashes.")
    print("  These descriptive intervals do not test equivalence to a published result.")
    print("  Agreement or disagreement cannot identify or exclude a checkpoint, protocol,")
    print("  compiler, or arm-order effect; reference equivalence remains unverified.")


def main(d):
    collections, errors, skipped = [], [], []
    for path in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            result = collection(path)
        except (ValueError, KeyError, TypeError) as e:
            errors.append((path, str(e))); continue
        if result is None:
            skipped.append(os.path.basename(path)); continue
        collections.append((path, *result))
    if skipped:
        print("  Skipped unrelated gates/diagnostics: " + ", ".join(skipped))
    if len(collections) > 1:
        print(f"  MULTIPLE COLLECTIONS: reporting {len(collections)} files separately; "
              "seeds, hosts, revisions, and arm orders are not pooled.")
    for path, meta, by in collections:
        report_collection(path, meta, by)
    for path, error in errors:
        print(f"  INVALID {os.path.basename(path)}: {error}")
    if errors: return 1
    if not collections:
        print(f"  SKIP  no eligible rollout collections under {d}")
        print("  Collect with: ./repro/run_experiment.sh published")
    return 0


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else os.path.join(here, "..", "results", "published")))

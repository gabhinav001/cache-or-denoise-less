"""Compare runs collected on a new host with the shipped runs, state by state.

    python3 probes/compare_regen.py DIR [DIR ...] [--json OUT]

Each run file under DIR (a regenerated results tree, or a smoke collection) is
matched to every run of the paper's collections under results/ (the directories
probes/report_budget_shift.py reads) with the same configuration --
checkpoint, suite, denoising steps, effective time shift, cache configuration and
prefix, fresh noise, camera jitter, CUDA graphs and seed -- whatever its name.
For every match the two are paired on (task, initial state):

  outcome changes   states where one run succeeded and the other failed. Rollouts
                    are deterministic on an RTX A6000, so this must be zero.
  length changes    states whose episode length differs. Seven A6000s changed 14
                    of 2,730 lengths without changing an outcome; small counts are
                    expected, and on a different GPU model both counts may grow.

Across the shipped collections themselves, 7 of 11,725 outcomes paired across
machines changed (report_budget_shift.py, determinism_exhaustive), mostly in
episodes that ended near the time limit on one machine. The exit status is 1 if
the new host changes more than --max-rate of its paired outcomes (default 0.5%,
about eight times that rate), which points at the host or its software rather than at
the runs.

Latency is reported as the ratio of the new host's mean call time to the shipped
run's, for runs regenerated under the same name; it describes the machine.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "probes"))
from report_budget_shift import config_key, EXCLUDED, DETERMINISM_DIRS   # noqa: E402  one definition of "same configuration"


def load(p):
    try:
        d = json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict) or "rows" not in d or "meta" not in d:
        return None
    m = d["meta"]
    if len(m.get("configs") or []) != 1:
        return None               # several configurations in one process: compared by name below
    return d


def key(d):
    return config_key(d["meta"])


def by_state(d):
    return {(r["task"], r["init_state"]): r for r in d["rows"]}


def ep_mean(d):
    ms = [r["ms_per_cycle"] for r in d["rows"] if not r.get("warmup") and "ms_per_cycle" in r]
    return sum(ms) / len(ms) if ms else None


def shipped_index():
    """The paper's collections only: other directories follow other protocols."""
    idx, names = {}, {}
    for p in sorted(q for d in DETERMINISM_DIRS for q in (RESULTS / d).rglob("*.json")):
        if p.name == "summary.json" or str(p.relative_to(RESULTS)) in EXCLUDED:
            continue
        d = load(p)
        if d is None:
            continue
        idx.setdefault(key(d), []).append((p, d))
        names.setdefault(p.name, []).append((p, d))
    return idx, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--verbose", action="store_true", help="one line per shipped counterpart")
    ap.add_argument("--max-rate", type=float, default=0.005,
                    help="largest share of paired outcomes that may change (default 0.005)")
    args = ap.parse_args()
    idx, names = shipped_index()
    report, flips_total, paired_total, steps_total, unmatched = [], 0, 0, 0, []
    same_path = {}
    for d0 in args.dirs:
        for p in sorted(d0.rglob("*.json")):
            if str(p).split("/results/", 1)[-1] in EXCLUDED:
                continue
            d = load(p)
            if d is None:
                continue
            new = by_state(d)
            matches = idx.get(key(d), [])
            if not matches:
                unmatched.append(str(p.relative_to(d0.parent))); continue
            for q, s in matches:
                if (s["meta"].get("provenance") or {}).get("utc") == (d["meta"].get("provenance") or {}).get("utc") \
                        and q.name == p.name:
                    continue          # the same run, copied
                old = by_state(s)
                common = sorted(set(new) & set(old))
                if not common:
                    continue
                flips = sum(new[k]["success"] != old[k]["success"] for k in common)
                steps = sum(new[k]["steps"] != old[k]["steps"] for k in common)
                lat = None
                if q.name == p.name.replace("smoke_", "") and ep_mean(d) and ep_mean(s):
                    lat = round(ep_mean(d) / ep_mean(s), 3)
                if str(q.relative_to(RESULTS)) == str(p).split("/results/", 1)[-1]:
                    # the shipped copy of this very run: the one the paper's numbers read
                    same_path[str(q.relative_to(RESULTS))] = {"outcome_changes": flips, "paired": len(common),
                        "states": [list(k) for k in common if new[k]["success"] != old[k]["success"]]}
                report.append({"run": str(p.relative_to(d0.parent)), "shipped": str(q.relative_to(ROOT)), "paired": len(common),
                               "outcome_changes": flips, "length_changes": steps, "latency_ratio": lat,
                               "shipped_host": (s["meta"].get("provenance") or {}).get("host_tag")})
                flips_total += flips; paired_total += len(common); steps_total += steps
    if args.verbose:
        for r in report:
            lat = f"  latency x{r['latency_ratio']}" if r["latency_ratio"] else ""
            print(f"  {Path(r['run']).name:34s} vs {r['shipped']:62s} {r['paired']:4d} states  "
                  f"outcomes changed {r['outcome_changes']}  lengths changed {r['length_changes']}{lat}")
    else:
        for run in dict.fromkeys(r["run"] for r in report):
            rs = [r for r in report if r["run"] == run]
            lat = [r["latency_ratio"] for r in rs if r["latency_ratio"]]
            print(f"  {Path(run).name:34s} {len(rs):2d} shipped runs on {len({r['shipped_host'] for r in rs})} hosts, "
                  f"{sum(r['paired'] for r in rs):5d} paired states, outcomes changed {sum(r['outcome_changes'] for r in rs)}, "
                  f"lengths changed {sum(r['length_changes'] for r in rs)}"
                  + (f", latency x{lat[0]} of the same-named shipped run" if lat else ""))
    for u in unmatched:
        print(f"  {u}: no shipped run with this configuration")
    runs = len({r["run"] for r in report})
    print(f"{runs} new runs against {len(report)} shipped runs: {paired_total} paired states, "
          f"{flips_total} outcome changes, {steps_total} length changes")
    if args.json:
        args.json.write_text(json.dumps({"comparisons": report, "unmatched": unmatched, "same_path": same_path,
                                         "paired": paired_total, "outcome_changes": flips_total,
                                         "length_changes": steps_total}, indent=1) + "\n")
    rate = flips_total / paired_total if paired_total else 0.0
    print(f"outcome change rate {100 * rate:.2f}% (shipped machines among themselves: 7 of 11725, 0.06%; "
          f"limit {100 * args.max_rate:.2f}%)")
    return 1 if rate > args.max_rate else 0


if __name__ == "__main__":
    sys.exit(main())

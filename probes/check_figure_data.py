"""Validate the workshop's two 500-state figure panels against raw artifacts.

Checks every point, success bound, sample count, metadata record, and source
checksum. Bootstrap bounds are independently recomputed from the raw state
outcomes using the documented report convention. Missing panels, unknown arms,
and unsupported snapshot schemas fail rather than silently checking zero points.

usage: check_figure_data.py [path/to/figure_data.json]
       WORKSHOP_FIGURE_DATA=<path> check_figure_data.py
The default paper snapshot is absent from an artifact-only checkout; that case
skips, while an explicitly supplied missing path fails.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DEFAULT = ROOT.parents[1] / "paper/workshop/figure_data.json"
ARMS = ("2st", "4st", "4st+cache", "10st+cache")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def marginal_intervals(arms):
    """Same percentile convention as report_statescale.py, from raw outcomes."""
    rng = random.Random(20260919)
    result = {}
    for key in ARMS:
        values = [arms[key][state] for state in sorted(arms[key])]
        n = len(values)
        samples = sorted(100 * sum(values[rng.randrange(n)] for _ in range(n)) / n
                         for _ in range(10000))
        result[key] = [round(samples[250], 1), round(samples[9749], 1)]
    return result


def validate(fd):
    bad = []
    def check(condition, message):
        if not condition:
            bad.append(message)

    check(fd.get("schema_version") == 2, "unsupported figure schema; regenerate the snapshot")
    panels = fd.get("panels", [])
    check([(p.get("panel"), p.get("seed")) for p in panels] == [("A", 42), ("B", 7)],
          "expected exactly panel A/seed42 and panel B/seed7")
    if bad:
        return bad, 0, 0, 0

    expected_summary = {"results/canonical_numbers.json", "probes/report_statescale.py"}
    declared_summary = fd.get("summary_sources", [])
    check(len(declared_summary) == 2 and {s.get("path") for s in declared_summary} == expected_summary,
          "summary source list is incomplete, duplicated, or unexpected")
    n_sources = 0
    for source in declared_summary:
        path = ROOT / source["path"]
        check(path.is_file(), f"missing summary source: {source['path']}")
        if path.is_file():
            check(source.get("sha256") == sha(path), f"changed summary source: {source['path']}")
        n_sources += 1

    canonical = json.loads((RESULTS / "canonical_numbers.json").read_text())
    shared_states, n_points = None, 0
    for panel in panels:
        seed = panel["seed"]
        suffix = "" if seed == 42 else f"_seed{seed}"
        expected_paths = [f"results/statescale/s{steps}{suffix}.json" for steps in (2, 4, 10)]
        sources = panel.get("sources", [])
        check([s.get("path") for s in sources] == expected_paths,
              f"seed {seed}: expected three schedule sources in order")
        sources_by_path = {s["path"]: s for s in sources}
        raw_arms, raw_metadata = {}, []
        for steps, relative in zip((2, 4, 10), expected_paths):
            path = ROOT / relative
            if not path.is_file():
                bad.append(f"missing raw source: {relative}")
                continue
            run = json.loads(path.read_text())
            meta = run["meta"]
            raw_metadata.append(meta)
            declared = sources_by_path.get(relative, {})
            check(declared.get("sha256") == sha(path), f"source hash changed: {relative}")
            check(declared.get("metadata") == meta, f"metadata mismatch: {relative}")
            check(meta["seed"] == seed and meta["num_inference_steps"] == steps,
                  f"seed/schedule mismatch: {relative}")
            expected_configs = (["stock"] if steps == 2 else
                                ["stock", "c3ache_tau4_shrink_skip"] if steps == 4 else
                                ["c3ache_tau4_shrink_skip"])
            check(meta["configs"] == expected_configs, f"arm-order mismatch: {relative}")
            check((meta["suite"], meta["replan"], meta["n_tasks"], meta["n_episodes"], meta["warmup_episodes"])
                  == ("libero_goal", 10, 10, 50, 1), f"protocol mismatch: {relative}")
            check({r["config"] for r in run["rows"]} == set(expected_configs),
                  f"unexpected or missing raw arm: {relative}")
            for config in expected_configs:
                rows = [r for r in run["rows"] if r["config"] == config]
                key = f"{steps}st" + ("" if config == "stock" else "+cache")
                states = {r["init_state"] for r in rows}
                check(len(rows) == len(states) == 500, f"seed {seed} {key}: not 500 unique states")
                if shared_states is None:
                    shared_states = states
                else:
                    check(states == shared_states, f"seed {seed} {key}: unmatched state population")
                check({r["task"] for r in rows} == set(range(10)), f"seed {seed} {key}: missing task")
                for task in range(10):
                    task_rows = [r for r in rows if r["task"] == task]
                    check(len(task_rows) == 50 and sum(bool(r.get("warmup")) for r in task_rows) == 1,
                          f"seed {seed} {key} task {task}: incorrect episode/warmup counts")
                raw_arms[key] = rows
            n_sources += 1
        if set(raw_arms) != set(ARMS):
            bad.append(f"seed {seed}: incomplete raw arms")
            continue
        check({m["provenance"]["host_tag"] for m in raw_metadata} == {panel.get("host_tag")},
              f"seed {seed}: host provenance mismatch")
        check({m["provenance"]["commit"] for m in raw_metadata} == {panel.get("commit")},
              f"seed {seed}: code-revision provenance mismatch")
        check({m["provenance"]["gpu"] for m in raw_metadata} == {"NVIDIA RTX A6000"},
              f"seed {seed}: unexpected hardware")
        points = panel.get("points", [])
        check([p.get("key") for p in points] == list(ARMS), f"seed {seed}: incorrect plotted arm set/order")
        outcomes = {key: {r["init_state"]: bool(r["success"]) for r in rows}
                    for key, rows in raw_arms.items()}
        bounds = marginal_intervals(outcomes)
        for point in points:
            key = point.get("key")
            if key not in raw_arms:
                bad.append(f"seed {seed}: unrecognized plotted arm {key}")
                continue
            n_points += 1
            rows = raw_arms[key]
            wins = sum(bool(r["success"]) for r in rows)
            timed = [r["ms_per_cycle"] for r in rows if not r.get("warmup")]
            expected = {"steps": int(key.split("st")[0]), "config": rows[0]["config"],
                        "cached": "+cache" in key, "successes": wins,
                        "success_episodes": len(rows), "success": 100 * wins / len(rows),
                        "timed_episodes": len(timed), "ms": round(statistics.mean(timed), 4),
                        "sd_ms": round(statistics.stdev(timed), 4), "ci95": bounds[key]}
            for field, value in expected.items():
                check(point.get(field) == value,
                      f"seed {seed} {key}: {field} is {point.get(field)!r}; raw data give {value!r}")
            if seed == 42:
                summary = canonical["statescale"]["arms"][key]
                check((summary["k"], summary["n"], summary["ms"], summary["ci95"])
                      == (wins, len(rows), round(statistics.mean(timed), 1), bounds[key]),
                      f"seed {seed} {key}: canonical summary differs from raw data")

    protocol = fd.get("protocol", {})
    expected_protocol = {"tasks": 10, "distinct_states_per_task": 50,
                         "same_states_across_all_arms_and_panels": True,
                         "success_episodes_per_arm": 500, "timed_episodes_per_arm": 490,
                         "warmup_episodes_per_task_per_arm": 1, "executed_actions_per_chunk": 10,
                         "cache_refresh_interval": 4, "hardware": "NVIDIA RTX A6000"}
    for field, value in expected_protocol.items():
        check(protocol.get(field) == value, f"incorrect declared protocol: {field}")
    if shared_states:
        digest = hashlib.sha256("\n".join(sorted(shared_states)).encode()).hexdigest()
        check(protocol.get("sorted_state_hashes_sha256") == digest, "shared state hash digest changed")
    pilot_states = {r["init_state"] for path in (RESULTS / "deployfrontier").glob("d*.json")
                    for r in json.loads(path.read_text())["rows"]}
    check(bool(pilot_states) and shared_states is not None and pilot_states <= shared_states,
          "pilot states are missing from the larger comparison")
    check(n_points == 8, f"checked {n_points} points, expected eight")
    return bad, n_points, n_sources, len(pilot_states)


def main(path, required=False):
    path = Path(path)
    if not path.exists():
        print(f"  {'FAIL' if required else 'SKIP'}: no figure data at {path}")
        return 1 if required else 0
    try:
        bad, points, sources, pilot_count = validate(json.loads(path.read_text()))
    except (KeyError, TypeError, ValueError, OSError) as error:
        print(f"  FAIL: invalid figure snapshot or raw artifact: {error}")
        return 1
    print(f"  figure data: {points} points and their bounds; {sources} source checksums")
    print(f"  state nesting: {pilot_count} pilot states within the shared 500-state population")
    if bad:
        print(f"  {len(bad)} PROBLEM(S):")
        for error in bad:
            print(f"    {error}")
        return 1
    print("  all plotted values, intervals, metadata, sample counts, and source hashes match")
    return 0


if __name__ == "__main__":
    explicit = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("WORKSHOP_FIGURE_DATA")
    sys.exit(main(explicit or DEFAULT, required=bool(explicit)))

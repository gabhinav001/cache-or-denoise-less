"""Merge the two halves of the libero_10 rerun into one file.

The suite was split across two machines by task range to keep wall-clock down.
Tasks are disjoint and every comparison is paired within a task, so the split
cannot bias the contrast; the assertion below is what guarantees the halves do
not overlap.
"""
from __future__ import annotations
import json, sys

def main(a_path, b_path, out):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    ta = {r["task"] for r in a["rows"]}; tb = {r["task"] for r in b["rows"]}
    assert not (ta & tb), f"task overlap between halves: {sorted(ta & tb)}"
    ca = {r["config"] for r in a["rows"]}; cb = {r["config"] for r in b["rows"]}
    assert ca == cb, f"halves ran different configs: {ca} vs {cb}"
    rows = a["rows"] + b["rows"]
    meta = {**a.get("meta", {}), "note": "libero_10 split across two machines by "
            "task range; tasks disjoint, comparisons paired within task",
            "tasks_a": sorted(ta), "tasks_b": sorted(tb)}
    json.dump({"meta": meta, "rows": rows}, open(out, "w"))
    print(f"merged {len(a['rows'])} + {len(b['rows'])} = {len(rows)} rows | "
          f"tasks {sorted(ta | tb)} | configs {sorted(ca)}")

if __name__ == "__main__":
    main(*sys.argv[1:4])

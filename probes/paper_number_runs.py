"""Which raw runs each number, table and figure of the paper depends on.

    python3 probes/paper_number_runs.py [--out repro/paper_number_runs.json]

Removes one run file at a time from a copy of the results, re-derives the
snapshot (probes/report_budget_shift.py), and records which printed numbers
(probes/paper_numbers.py), tables (probes/make_tables.py) and plotted points
(probes/make_figure.py) change. A number depends on a run if removing the run
changes the number as printed; if no single run does (a maximum, a rounded fit),
on the runs that change its unrounded value; and if none does that either, on
the runs that change any snapshot entry it reads. A tally over machines rests on
every run its sentence counts.

repro/regenerate_paper.sh and repro/PAPER_NUMBERS.md use the result to say which
tier of the regeneration plan reproduces what. CPU only; about ten minutes.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import paper_numbers as PN          # noqa: E402
import check_regeneration as CR     # noqa: E402

PY = sys.executable


def flat(o, pre="", out=None):
    out = {} if out is None else out
    if isinstance(o, dict):
        for k, v in o.items():
            flat(v, f"{pre}/{k}" if pre else str(k), out)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            flat(v, f"{pre}[{i}]", out)
    else:
        out[pre] = o
    return out


def derive(res, work):
    """Snapshot, printed numbers, raw values, tables and plotted points for one results tree."""
    s_path = work / "summary.json"
    subprocess.run([PY, str(HERE / "report_budget_shift.py"), "--results", str(res), "--out", str(s_path)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    S = json.loads(s_path.read_text())
    printed, raw, reads = {}, {}, {}
    for i, e in enumerate(PN.ENTRIES):
        for j, n in enumerate(e.nums):
            if n.get is None:
                continue
            key = f"{i}:{j}"
            PN._TRACE = []
            try:
                printed[key] = n.value(S)
                raw[key] = n.get(S) if callable(n.get) else PN.g(S, n.get)
            except Exception as ex:          # the number cannot be derived without this run
                printed[key] = raw[key] = f"<{type(ex).__name__}>"
            reads[key] = sorted(set(PN._TRACE))
            PN._TRACE = None
    for d in ("tables", "figures"):
        shutil.rmtree(work / d, ignore_errors=True)
    subprocess.run([PY, str(HERE / "make_tables.py"), "--summary", str(s_path), "--outdir", str(work)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([PY, str(HERE / "make_figure.py"), "--summary", str(s_path), "--outdir", str(work)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tables = {p.name: p.read_text() for p in sorted((work / "tables").glob("*.tex"))}
    points = json.loads((work / "figure_data.json").read_text())["points"]
    figs = {f"Fig. {k}": [p for p in points if p.get("figure") == k] for k in (1, 2)}
    return {"flat": flat(S), "printed": printed, "raw": raw, "reads": reads, "tables": tables, "points": figs}


def run_tiers(inputs):
    """Which tier of repro/regeneration_plan.txt rewrites each run."""
    tiers = {}
    with tempfile.TemporaryDirectory() as tmp:
        for row in CR.read_plan():
            calls = CR.dry_run(row, Path(tmp) / row["dir"])
            names = set(calls) | ({"gate.json"} if CR.gated(row["experiment"]) else set())
            for name in names:
                rel = f"{row['dir']}/{name}"
                if rel in inputs and row["tier"] not in tiers.setdefault(rel, []):
                    tiers[rel].append(row["tier"])
    return tiers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "repro" / "paper_number_runs.json")
    ap.add_argument("--tiers-only", action="store_true",
                    help="after editing repro/regeneration_plan.txt: refresh only which tier rewrites each run")
    args = ap.parse_args()
    if args.tiers_only:
        out = json.loads(args.out.read_text())
        out["run_tiers"] = run_tiers(out["inputs"])
        args.out.write_text(json.dumps(out, indent=1) + "\n")
        print(f"refreshed run_tiers in {args.out.name}")
        return
    inputs = CR.analysis_inputs()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        res = tmp / "results"
        for rel in inputs:
            (res / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "results" / rel, res / rel)
        # the analysis also counts harness revisions over whole directories
        for d in ("budget_shift", "matched100_a6000", "matched50_a6000", "composed2step"):
            for p in (ROOT / "results" / d).rglob("*.json"):
                q = res / p.relative_to(ROOT / "results")
                if not q.exists() and p.name != "summary.json":
                    q.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, q)
        work = tmp / "work"
        work.mkdir()
        base = derive(res, work)
        by_value, by_raw, by_read = {}, {}, {}
        tables, figures = {}, {}
        for n_, rel in enumerate(inputs, 1):
            f = res / rel
            off = f.with_name(f.name + ".off")
            f.rename(off)
            try:
                d = derive(res, work)
            finally:
                off.rename(f)
            changed = {k for k in base["flat"] if d["flat"].get(k, "<absent>") != base["flat"][k]} | \
                      {k for k in d["flat"] if k not in base["flat"]}
            for key in base["printed"]:
                if d["printed"].get(key) != base["printed"][key]:
                    by_value.setdefault(key, []).append(rel)
                if d["raw"].get(key) != base["raw"][key]:
                    by_raw.setdefault(key, []).append(rel)
                if any(c == r or c.startswith(r + "/") or c.startswith(r + "[") for r in base["reads"][key] for c in changed):
                    by_read.setdefault(key, []).append(rel)
            for name, text in base["tables"].items():
                if d["tables"].get(name) != text:
                    tables.setdefault(name, []).append(rel)
            for fig, pts in base["points"].items():
                if d["points"].get(fig) != pts:
                    figures.setdefault(fig, []).append(rel)
            print(f"  {n_:3d}/{len(inputs)} {rel}", flush=True)
    numbers = {}
    for i, e in enumerate(PN.ENTRIES):
        for j, n in enumerate(e.nums):
            key = f"{i}:{j}"
            if n.get is None:
                continue
            runs, basis = by_value.get(key), "printed value"
            if not runs:
                runs, basis = by_raw.get(key), "unrounded value"
            if not runs:
                runs, basis = by_read.get(key, []), "snapshot entries read"
            numbers[key] = {"where": e.where, "printed": n.printed, "kind": n.kind, "basis": basis, "runs": sorted(runs)}
        # A tally over machines ("seven machines", "none of 2,730 outcomes") is
        # moved by no single run; it rests on every run its sentence counts.
        keys = [f"{i}:{j}" for j, n in enumerate(e.nums) if n.get is not None]
        pooled = sorted({r for k in keys for r in numbers[k]["runs"]})
        for k in keys:
            if numbers[k]["kind"] == "hosts" or not numbers[k]["runs"]:
                if set(pooled) != set(numbers[k]["runs"]):
                    numbers[k]["runs"], numbers[k]["basis"] = pooled, "runs of the whole sentence"
    out = {"inputs": inputs, "run_tiers": run_tiers(inputs), "numbers": numbers,
           "tables": {k: sorted(v) for k, v in tables.items()},
           "figures": {k: sorted(v) for k, v in figures.items()}}
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    unused = sorted(set(inputs) - {r for v in numbers.values() for r in v["runs"]}
                    - {r for v in tables.values() for r in v} - {r for v in figures.values() for r in v})
    print(f"wrote {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}: "
          f"{len(numbers)} numbers, {len(tables)} tables, figure data")
    print(f"  {len(inputs) - len(unused)} of {len(inputs)} runs the analysis reads reach something printed")
    for r in unused:
        print(f"    reaches nothing printed: {r}")


if __name__ == "__main__":
    main()

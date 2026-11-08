#!/usr/bin/env bash
# Focused workshop verification. CPU only; does not install, provision, or mutate results.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python3}"
for test in test_cache_semantics test_path_fraction test_stack_probe_audit test_report_published test_baseline_audit test_schedule_sensitivity test_checkpoint_coverage test_cache_integrity test_report_tau8_fresh test_screen_engine test_report_screen test_latent_delay test_report_latent_delay; do
  "$PY" "$ROOT/probes/$test.py"
done
PYTHONPATH="$ROOT/probes${PYTHONPATH:+:$PYTHONPATH}" "$PY" - "$ROOT" <<'PYVERIFY'
import json,sys
from pathlib import Path
from report_schedule_sensitivity import analyze
root=Path(sys.argv[1])
frozen=json.loads((root/"results/schedule_sensitivity.json").read_text())
current=analyze(root/"results")
if "numpy" in current.get("method", {}):
    current["method"]["numpy"] = frozen.get("method", {}).get("numpy")
if frozen!=current:
    raise SystemExit("Schedule snapshot differs from raw-data analysis; inspect before regenerating")
print("Sensitivity snapshot, 24 source hashes, coverage, pairing, and intervals verified.")
PYVERIFY
echo "Historical figure data (500-state collection cited by earlier drafts):"
"$PY" "$ROOT/probes/check_figure_data.py" "$ROOT/probes/workshop_figure_data.json"
"$PY" "$ROOT/probes/report_tau8_fresh.py" "$ROOT/results/tau8_fresh"
[ -d "$ROOT/results/matched_frontier" ] && "$PY" "$ROOT/probes/report_matched_frontier.py" "$ROOT/results/matched_frontier"
[ -d "$ROOT/results/matched50_a6000" ] && "$PY" "$ROOT/probes/report_matched50.py" "$ROOT/results/matched50_a6000"
[ -d "$ROOT/results/matched100_a6000" ] && "$PY" "$ROOT/probes/report_matched100.py" "$ROOT/results/matched100_a6000"
[ -d "$ROOT/results/matched_composed" ] && "$PY" "$ROOT/probes/report_matched_composed.py" "$ROOT/results/matched_composed"
# The paper's numbers: re-derive the snapshot from the raw run files and require
# it to match the committed one exactly (the bootstrap uses a fixed generator).
if [ -s "$ROOT/results/budget_shift/summary.json" ]; then
  tmp="$(mktemp -t budget_shift_summary.XXXXXX)"
  "$PY" "$ROOT/probes/report_budget_shift.py" --out "$tmp" >/dev/null
  cmp -s "$tmp" "$ROOT/results/budget_shift/summary.json" \
    || { echo "results/budget_shift/summary.json differs from a fresh derivation"; rm -f "$tmp"; exit 1; }
  rm -f "$tmp"
  echo "Paper snapshot re-derived from raw runs and matches results/budget_shift/summary.json."
  # Every number printed in the paper, from that snapshot; the paper's tables,
  # rebuilt from it byte for byte; and every raw run behind them, against the
  # scripts that would regenerate it on a GPU host.
  "$PY" "$ROOT/probes/paper_numbers.py"
  tdir="$(mktemp -d -t paper_tables.XXXXXX)"
  "$PY" "$ROOT/probes/make_tables.py" --summary "$ROOT/results/budget_shift/summary.json" --outdir "$tdir" >/dev/null
  for t in "$ROOT"/repro/expected/tables/*.tex; do
    cmp -s "$t" "$tdir/tables/$(basename "$t")" \
      || { echo "tables/$(basename "$t") differs from repro/expected/"; rm -rf "$tdir"; exit 1; }
  done
  rm -rf "$tdir"
  echo "The paper's $(ls "$ROOT"/repro/expected/tables/*.tex | wc -l | tr -d ' ') tables rebuild from the snapshot exactly (repro/expected/tables/)."
  "$PY" "$ROOT/probes/check_regeneration.py"
  doc="$(mktemp -t paper_numbers.XXXXXX)"
  "$PY" "$ROOT/probes/paper_numbers.py" --markdown "$doc" >/dev/null
  cmp -s "$doc" "$ROOT/repro/PAPER_NUMBERS.md" \
    || { echo "repro/PAPER_NUMBERS.md is stale: python3 probes/paper_numbers.py --markdown repro/PAPER_NUMBERS.md"; rm -f "$doc"; exit 1; }
  rm -f "$doc"
  echo "repro/PAPER_NUMBERS.md matches the number map."
fi
bash -n "$ROOT/repro/experiments/budget_shift.sh"
bash -n "$ROOT/repro/experiments/baseline_audit.sh"
for variant in original optional; do
  for phase in smoke pilot full; do
    DRY_RUN=1 BASELINE_VARIANT="$variant" BASELINE_PHASE="$phase" \
      bash "$ROOT/repro/run_experiment.sh" baseline_audit --seeds 42 --out /tmp/baseline-audit-dryrun >/dev/null
  done
done
echo "Workshop CPU checks passed. Pending GPU collections are not validated by these checks."

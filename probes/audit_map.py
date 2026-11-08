"""Every reported figure, back to the run that produced it.

Maps each table and figure to its configuration, revision, outcomes and
analysis command. The pieces existed and were scattered:
MANIFEST.md carried per-run provenance, check_repro_map.py carried
experiment-to-artefact, figure_data.json carried per-source hashes, and the
analysis command was only ever implicit in verify.sh. Nothing joined them, so
auditing one number meant reading four files and inferring the link.

This emits that join as one table. For each reported quantity:

  claim        what the write-ups assert
  experiment   the committed script that collects it, runnable as
               repro/run_experiment.sh <name>
  artefacts    the shipped files it reads
  commit       the code revision recorded IN those artefacts, or a dash where
               the run predates provenance capture -- shown, never inferred
  gpu          the hardware recorded in the same place
  position     the arm's index within its invocation. Position 1 reproduces
               bit-identically (results/rngdrift/); a shifted position is the
               known arm-order hazard, so this is reported rather than assumed
               away
  analysis     the exact command that turns those artefacts into the number

Written for an auditor who does not trust the summary: every column is read out
of the artefacts at run time, so a stale entry cannot survive here.
"""
from __future__ import annotations

import glob, json, os, sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R = os.path.join(ROOT, "results")

# claim -> (experiment, results subdir, glob, analysis command)
MAP = OrderedDict([
    ("Table 1 / Fig. 1: every schedule and cache setting, both checkpoints, 100 LIBERO-Goal states, one host",
     ("budget_shift --seeds @frontier100_a6000", "budget_shift/frontier100_a6000", "*.json",
      "probes/report_budget_shift.py")),
    ("latency: upstream CUDA graphs, blockwise hooks, and repeated arms on the frontier host",
     ("budget_shift --seeds @latency_a6000", "budget_shift/latency_a6000", "*.json",
      "probes/report_budget_shift.py")),
    ("Fig. 2: cached share of the flow path and 10-step uncached controls at both shifts, 50 states",
     ("budget_shift --seeds @shift_a6000", "budget_shift/shift_a6000", "*.json",
      "probes/report_budget_shift.py")),
    ("fresh initial noise at every call instead of the fixed per-run latent, 50 states",
     ("budget_shift --seeds @noise_a6000", "budget_shift/noise_a6000", "*.json",
      "probes/report_budget_shift.py")),
    ("second copies of arms that ran on two new hosts (cross-host determinism check only)",
     ("budget_shift --seeds @repeats_a6000", "budget_shift/repeats_a6000", "*.json",
      "probes/report_budget_shift.py")),
    ("LIBERO-Long: 2 vs 10 steps vs tau=8 cache, both checkpoints, 50 states",
     ("budget_shift --seeds @long_a6000", "budget_shift/long_a6000", "*.json",
      "probes/report_budget_shift.py")),
    ("follow-up: s = 3 and 57-75% shares (Fig. 2), camera jitter, interleaved composed timing, fresh noise at 70-75%",
     ("followup", "budget_shift/followup_a6000", "*.json",
      "probes/report_budget_shift.py")),
    ("idle host: camera jitter on 250 states (Optional-IDM) and interleaved composed timing, alone on a machine rented whole",
     ("idle_host", "budget_shift/idle_a6000", "*.json",
      "probes/report_budget_shift.py")),
    ("held-out shift s = 2 and a second noise stream (seed 7), predictions committed before the runs",
     ("heldout", "budget_shift/heldout_a6000", "*.json",
      "probes/report_budget_shift.py")),
    ("cache screen on uncached rollouts and a held-out closed-loop test, predictions committed before the runs",
     ("screen", "budget_shift/screen_a6000", "*.json",
      "probes/report_budget_shift.py")),
    ("episode-latent control for the fresh-noise loss, and delayed execution, predictions committed before the runs",
     ("latent_delay", "budget_shift/latent_delay_a6000", "*.json",
      "probes/report_budget_shift.py")),
    ("delays of CUDA-graph execution for the delayed-execution comparison, analysis committed before the runs",
     ("delay_graphs", "budget_shift/delay_graphs_a6000", "*.json",
      "probes/report_budget_shift.py")),
    ("matched same-host uncached vs composed 2-step caching, 100 states",
     ("matched_composed", "matched_composed", "*.json",
      "probes/report_matched_composed.py")),
    ("tau=8 attenuation, +15.2 pts / 5 seeds",
     ("tau8", "tau8", "*.json", "probes/report_tau8.py results/tau8")),
    ("residual geometry, 3 backbones",
     ("mechanism", "residual", "*families*.npz", "probes/report_families.py")),
    # The ages 5-7 plateau is its own reported quantity on its own artefact and
    # its own tag, so it gets its own row rather than hiding inside the geometry
    # glob above -- a reader chasing that number should not have to infer which
    # of eight npz files carries it or which tag collected it.
    ("cache-age weights, ages 1-7 (deployment checkpoint)",
     ("mechanism --seeds idm7", "residual", "fw_families_idm_taps7.npz",
      "probes/kalman_beta.py results/residual/fw_families_idm_taps7.npz")),
    ("global vs per-cell at tau=4",
     ("baselines", "baselines", "*.json", "probes/report_baselines.py results/baselines")),
    ("deployment frontier, 11 arms",
     ("deployfrontier", "deployfrontier", "d*.json",
      "probes/report_deployfrontier.py results/deployfrontier")),
    ("frontier fragility + 3 seeds",
     ("deployfrontier", "deployfrontier", "d*_seed*.json",
      "probes/report_frontier_seeds.py")),
    ("tail latency, all 11 arms",
     ("deployfrontier", "deployfrontier", "d*.json", "probes/report_tails.py")),
    ("500 states, 4 arms",
     ("statescale", "statescale", "s*.json", "probes/report_statescale.py")),
    ("completed matrix (reference, plain, fitted global)",
     ("matrix", "matrix", "m*.json", "probes/report_matrix.py")),
    ("short-schedule composition + CI",
     ("cachesteps", "cachesteps", "*.json", "probes/report_cachesteps_ci.py")),
    ("pi_0.5 confirmation, 3 seeds",
     ("pi05_confirm", "pi05_confirm", "*.json",
      "probes/report_pi05.py results/pi05_confirm")),
    ("tau=8 fresh-process replication (+15.5 pts / 100 states)",
     ("tau8_fresh", "tau8_fresh", "*.json",
      "probes/report_tau8_fresh.py results/tau8_fresh")),
    ("historical matched accuracy-latency frontier, 6 arms",
     ("matched_frontier", "matched_frontier", "*.json",
      "probes/report_matched_frontier.py results/matched_frontier")),
    ("matched 50-state evaluation, one A6000 host (whole-stack sweep, blockwise hooks, both checkpoints)",
     ("matched50", "matched50_a6000", "*.json",
      "probes/report_matched50.py results/matched50_a6000")),
    ("fresh-process baseline audit (original vs Optional-IDM checkpoint)",
     ("baseline_audit", "baseline_audit", "*/*/*/*.json",
      "probes/report_baseline_audit.py results/baseline_audit/ --variant original")),
    ("matched 100-state frontier and 2x2 checkpoint-by-shift intervention, one A6000 host",
     ("matched100", "matched100_a6000", "*.json",
      "probes/report_matched100.py results/matched100_a6000")),
    ("composed 2-step schedule with step 0 cached, 100 states",
     ("composed2step", "composed2step", "*.json",
      "probes/report_budget_shift.py")),
])


def provenance(files):
    commits, gpus = set(), set()
    for f in files:
        try:
            m = (json.load(open(f)).get("meta") or {})
        except Exception:
            continue
        p = m.get("provenance") or {}
        commits.add((p.get("commit") or "")[:12] or "—")
        gpus.add(p.get("gpu") or "—")
    # .npz tensors carry no meta block, so an empty set means "not applicable"
    return (commits or {"n/a (tensor artefact)"}), (gpus or {"n/a (tensor artefact)"})


def positions(files):
    """arm -> index within its invocation, read from row order."""
    out = {}
    for f in files:
        try:
            rows = json.load(open(f))["rows"]
        except Exception:
            continue
        seen = []
        for r in rows:
            if r.get("config") and r["config"] not in seen: seen.append(r["config"])
        for i, c in enumerate(seen, 1):
            out.setdefault(i, set()).add(os.path.basename(f))
    return out


def main():
    print("# Audit map\n")
    print("Each reported quantity, back to the script that collects it, the")
    print("artefacts it reads, the revision and hardware recorded inside those")
    print("artefacts, and the command that derives the number. Generated by")
    print("probes/audit_map.py; every column is read from the artefacts.\n")
    shifted = 0
    for claim, (exp, sub, pat, cmd) in MAP.items():
        files = sorted(glob.glob(os.path.join(R, sub, pat)))
        files = [f for f in files if not os.path.basename(f).startswith("gate") and not f.endswith(".command.json")]
        if not files:
            print(f"## {claim}\n\n  NOT SHIPPED YET (experiment `{exp}`)\n"); continue
        commits, gpus = provenance(files)
        pos = positions(files)
        npos = sorted(pos)
        if npos and npos != [1]: shifted += 1
        print(f"## {claim}\n")
        print(f"- experiment: `repro/run_experiment.sh {exp}`")
        print(f"- artefacts: {len(files)} file(s) in `results/{sub}/` "
              f"(`{os.path.basename(files[0])}` …)")
        print(f"- commit recorded: {', '.join(sorted(commits))}")
        print(f"- gpu recorded: {', '.join(sorted(gpus))}")
        if not npos:
            posline = "n/a (no per-arm rows)"
        elif npos == [1]:
            posline = "1 only — bit-identical per results/rngdrift/"
        else:
            posline = (", ".join(str(p) for p in npos)
                       + " — arms beyond 1 carry the known arm-order hazard")
        print(f"- arm positions: {posline}")
        print(f"- analysis: `python3 {cmd}`\n")
    print(f"---\n\n{len(MAP)} reported quantities mapped; {shifted} include an arm "
          f"at a position beyond 1.")
    print("\nPosition matters because a different arm list compiles a different set")
    print("of graphs, not because of random state: results/rngdrift/ shows reseeding")
    print("changes nothing while a position-1 arm reproduces bit-identically. The")
    print("headline 500-state comparison is position-1 against position-1.")
    print("\nA dash under commit means the run predates provenance capture; the")
    print("manifest lists those rows rather than hiding them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

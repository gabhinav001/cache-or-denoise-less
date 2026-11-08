#!/usr/bin/env bash
# Re-derive every published number from the committed artefacts. No GPU, no
# rented machine, no cost, about a minute.
#
# This is the check to run first. It does not re-collect the data -- it re-runs
# the analysis over the data that is in this repository, and prints each result
# next to the value the paper states. If something here disagrees, the paper is
# wrong about its own artefacts, which is a different and more serious problem
# than a fresh run landing somewhere else.
#
#   ./repro/verify.sh            everything
#   ./repro/verify.sh residual   one section

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
R="$ROOT/results"
# A section that disagrees with a stated figure must fail the whole run. This
# script has no `set -e` -- deliberately, since every section should run even if
# an earlier one fails -- so failures are accumulated and reported at the end.
# Without this, report_variance.py exited 1 on a live disagreement and verify.sh
# exited 0 over the top of it.
FAILED=0
V() { $PY "$ROOT/probes/report_variance.py" "$@" || FAILED=1; }
WHICH="${1:-all}"

hdr() { printf '\n\033[1m%s\033[0m\n%s\n' "$1" "$(printf '=%.0s' $(seq 1 ${#1}))"; }
want() { [ "$WHICH" = all ] || [ "$WHICH" = "$1" ]; }

command -v "$PY" >/dev/null || { echo "python3 not found" >&2; exit 1; }
$PY -c "import numpy" 2>/dev/null || { echo "need numpy: pip install numpy" >&2; exit 1; }

if want tests; then
  hdr "Analysis math and cache semantics (unit tests)"
  $PY "$ROOT/probes/test_famlib.py"        || exit 1
  $PY "$ROOT/probes/test_cache_semantics.py" || exit 1
  $PY "$ROOT/probes/test_perturb.py"         || exit 1
  echo "-- canonical numbers (every figure a write-up may quote) --"
  $PY "$ROOT/probes/canonical_numbers.py" || exit 1
  echo "-- no run pooled twice, and no re-run that silently reproduced old data --"
  $PY "$ROOT/probes/check_duplicates.py"     || exit 1
  echo "-- nothing in the repository identifies its authors --"
  $PY "$ROOT/probes/check_anonymity.py"      || FAILED=1
  # Regenerate the experiment manifest and fail if it is stale: it is the one
  # place to read what produced each run, INCLUDING the runs whose
  # provenance is incomplete, and a manifest that drifts from the artefacts
  # answers the opposite of the question it was asked.
  $PY "$ROOT/probes/make_manifest.py" > "$ROOT/repro/.MANIFEST.new" 2>/dev/null
  if ! diff -q "$ROOT/repro/MANIFEST.md" "$ROOT/repro/.MANIFEST.new" >/dev/null 2>&1; then
    echo "  MANIFEST.md is stale -- regenerate with probes/make_manifest.py"; FAILED=1
  fi
  rm -f "$ROOT/repro/.MANIFEST.new"
  echo "-- every cited artefact can be re-collected by an experiment --"
  $PY "$ROOT/probes/check_repro_map.py"      || FAILED=1
  echo "-- every argument an experiment passes is one its probe reads --"
  $PY "$ROOT/probes/check_probe_args.py"     || FAILED=1
  echo "-- a tag selects the same inputs wherever it appears in --seeds --"
  $PY "$ROOT/probes/check_tag_purity.py"     || FAILED=1
  echo "-- the t quantile used for every small-sample interval --"
  $PY "$ROOT/probes/tstat.py"                || FAILED=1
  echo "-- which random stream the denoiser latent comes from (the A8 mechanism) --"
  $PY "$ROOT/probes/check_seed_path.py"      || FAILED=1
  echo "-- the write-up still quotes the artefacts (skips if no write-up present) --"
  $PY "$ROOT/probes/check_paper_numbers.py"  || FAILED=1
  # The workshop paper is the document that gets submitted, and it was not
  # guarded at all until now: the checker only knew the full paper and the
  # results page. It carries an explicit expected set (probes/workshop_figures
  # .json) because a 4-page condensation legitimately omits most figures, but
  # everything it states must still match the artefacts.
  _WS="$ROOT/../../paper/workshop/main.tex"
  [ -f "$_WS" ] && { $PY "$ROOT/probes/check_paper_numbers.py" "$_WS" || FAILED=1; }
  # The figure ships inside the PDF as a picture, so no number check reaches it.
  # This re-derives every plotted point from results/ and skips if absent.
  $PY "$ROOT/probes/check_figure_data.py" || FAILED=1
  # Seed variants share initial states, so a blind glob merges them silently.
  # This has happened twice; the check is structural rather than numeric.
  $PY "$ROOT/probes/check_seed_merge.py" || FAILED=1
  # The audit map joins claim -> script -> artefacts -> revision -> command.
  # Regenerated here so a stale copy cannot ship.
  $PY "$ROOT/probes/audit_map.py" > "$ROOT/repro/.AUDIT_MAP.new" 2>/dev/null
  if ! diff -q "$ROOT/repro/AUDIT_MAP.md" "$ROOT/repro/.AUDIT_MAP.new" >/dev/null 2>&1; then
    echo "  AUDIT_MAP.md is stale -- regenerate with probes/audit_map.py"; FAILED=1
  fi
fi

if want tau8; then
  hdr "§01b  tau=8 re-collected as a five-seed replication"
  $PY "$ROOT/probes/report_tau8.py" "$R/tau8"           || FAILED=1
  # The same two collections, compared on the arm that caches nothing.
  $PY "$ROOT/probes/report_armshift.py" "$R/frontier" "$R/tau8" || FAILED=1
fi

if want refimpl; then
  hdr "§02b  Reference-protocol reconciliation: the executed horizon"
  $PY "$ROOT/probes/report_refimpl.py" "$R/refimpl"          || FAILED=1
fi

if want stocktime; then
  hdr "§09b  An un-instrumented baseline, and why it is not measurable"
  $PY "$ROOT/probes/report_stocktime.py" "$R/stocktime"      || FAILED=1
fi

if want deployfrontier; then
  hdr "§10d  Deployable frontier: clean baselines, refresh/reuse tails"
  $PY "$ROOT/probes/report_deployfrontier.py" "$R/deployfrontier" || FAILED=1
fi

if want stockquiet; then
  hdr "§09c  Stock-policy timing on a quiet host"
  $PY "$ROOT/probes/report_stocktime.py" "$R/stockquiet"     || FAILED=1
fi

if want cachesteps; then
  hdr "§10c  Caching on top of a reduced-step schedule"
  $PY "$ROOT/probes/report_cachesteps.py" "$R/cachesteps"    || FAILED=1
  $PY "$ROOT/probes/report_cachesteps_ci.py"                  || FAILED=1
  $PY "$ROOT/probes/report_frontier_fragility.py"             || FAILED=1
  $PY "$ROOT/probes/report_frontier_seeds.py"                 || FAILED=1
  $PY "$ROOT/probes/report_statescale.py"                     || FAILED=1
  $PY "$ROOT/probes/report_tails.py"                          || FAILED=1
fi

if want frontier_timing; then
  hdr "§10b  Frontier latency, one quiet host"
  $PY "$ROOT/probes/report_frontier_timing.py" "$R/frontier_timing" || FAILED=1
fi

if want latency; then
  hdr "§09  Skip-equivalence, dead-code elimination, and measured latency"
  $PY "$ROOT/probes/report_latency.py" "$R/latency"       || FAILED=1
fi

if want baselines; then
  hdr "§10  The latency-success frontier against simple baselines"
  $PY "$ROOT/probes/report_baselines.py" "$R/baselines" "$R/latency" \
      --timing "$R/frontier_timing" || FAILED=1
fi

if want rngdrift; then
  hdr "§A8  Does the global RNG affect these rollouts, and does arm order?"
  $PY "$ROOT/probes/report_rngdrift.py" "$R/rngdrift"     || FAILED=1
fi

if want pi05_confirm; then
  hdr "§07b  pi0.5 pre-registered confirmation at beta=0.25"
  $PY "$ROOT/probes/report_pi05_confirm.py" "$R/pi05_confirm" || FAILED=1
fi

if want alpha; then
  hdr "§03  Is it the tap count that fails, or the sum-to-one constraint?"
  $PY "$ROOT/probes/alpha_constraint.py"     || FAILED=1
fi

if want closedloop; then
  hdr "§01/§02  Attenuation in closed loop, and the refresh-interval gradient"
  V closedloop "$R/closedloop"
fi

if want frontier; then
  hdr "§04  Does attenuation move the compute/quality frontier?"
  V closedloop "$R/frontier"
  echo
  V frontier "$R/frontier"
fi

if want applicability; then
  hdr "§02  Where the correction helps, and where it hurts"
  V applicability "$R/closedloop"
fi

if want ceiling_beta; then
  hdr "§04c  Can a better weight close the gap? (oracle-weight ceiling)"
  V closedloop "$R/ceiling"
fi

if want staleness; then
  hdr "§04b  Staleness-aware weights in closed loop"
  V closedloop "$R/staleness"
fi

if want generalize; then
  hdr "§05b  Does the correction survive an observation shift?"
  V generalize "$R/generalize"
fi

if want residual; then
  hdr "§03/§04  Residual geometry, three backbones, two runs each"
  V residual "$R/residual"
fi

if want kalman; then
  hdr "Staleness-conditioned attenuation (the Kalman question)"
  # fw_families_idm.npz is the DEPLOYMENT checkpoint -- the one the closed-loop
  # runs use and the one the shipped weights are fit from. It was missing here,
  # so the cache-age behaviour was only ever reported for checkpoints the policy
  # does not run, which was a real gap: the diagnostic
  # and the deployed configuration were different models.
  # Every file listed here lands in kalman_beta.json, and --write OVERWRITES
  # that file with exactly this set -- omitting one silently drops it. Without
  # --write the probe only prints, so inspecting a single artefact is safe.
  $PY "$ROOT/probes/kalman_beta.py" --write "$R/residual/fw_families.npz" \
      "$R/residual/fw_families_idm.npz" "$R/residual/fw_families_idm_taps7.npz" \
      "$R/residual/pi05_families.npz" "$R/residual/dz_families.npz"
fi

if want online; then
  hdr "Estimating the weight with no extra compute (the practitioner's question)"
  $PY "$ROOT/probes/online_beta.py" "$R/residual/fw_families.npz" \
      "$R/residual/pi05_families.npz" "$R/residual/dz_families.npz"
fi

if want families; then
  hdr "§A1  Predictor families over the committed Gram tensors"
  $PY "$ROOT/probes/report_families.py" \
      "$R/residual/fw_families.npz"   Fast-WAM \
      "$R/residual/pi05_families.npz" pi0.5 \
      "$R/residual/dz_families.npz"   DreamZero || FAILED=1
fi

if want beta_ceiling; then
  hdr "How much headroom does a learned scalar weight have?"
  $PY "$ROOT/probes/beta_ceiling.py" "$R/residual/fw_families.npz" \
      "$R/residual/pi05_families.npz" "$R/residual/dz_families.npz"
fi

if want fidelity; then
  hdr "§05  Open-loop action fidelity"
  V fidelity "$R/audit/fw_shrink_open.json" "$R/audit/fw_shrink_open_repro.json"
fi

if want curve; then
  hdr "§06  Compute-success curve, two seeds"
  V curve "$R/audit/libero_tau_mj332.json" "$R/audit/curve_seed7.json"
fi

if want placement; then
  hdr "§06  Placement at a matched compute budget, two machines"
  for f in "$R/audit/fw_scale_a100.json" "$R/audit/place_repro.json"; do
    [ -e "$f" ] || continue
    echo "--- $(basename "$f")"
    $PY "$ROOT/probes/report_placement.py" "$f"
    echo
  done
  echo "  The ordering is identical on both machines and every placement agrees"
  echo "  to within 0.005. Predictability is a weak criterion: the oracle beats"
  echo "  random but recovers only 35% of the distance to the published placement,"
  echo "  using ground truth that is not available at deployment."
fi

if want ceiling; then
  hdr "§06  Compile ceiling, two machines"
  V ceiling "$R/audit/vast_fw_step_sweep.json" "$R/killchecks/stepsweep_a100_uncond_v1.json"
fi

if want mujoco; then
  hdr "§08  mujoco version hazard, same box"
  V mujoco "$R/audit/q4_mujoco_332.json" "$R/audit/q4_mujoco_237.json"
fi

if want pi05; then
  hdr "§07  pi0.5 closed loop -- unresolved, and why"
  V closedloop "$R/pi05" 2>/dev/null
  cat <<'CAVEAT'

  Do not read the pooled row as the result. The runs disagree in SIGN on the
  same configuration (-0.3 at seed 42, +9.0 at seed 7), so the between-seed
  variance dominates the effect and pooling discordant pairs across seeds
  summarises nothing. The per-run rows are the evidence, and they do not agree.

CAVEAT
  echo "  The beta sweep is the decisive part: if the effect were real, success"
  echo "  would vary systematically with the substitution weight."
  echo
  echo "  Read by marginal rates the sweep looks flat. That is not the test used"
  echo "  anywhere else here: every other comparison is paired on (task, episode),"
  echo "  including staleness, which pairs across separate files exactly as these"
  echo "  five runs allow. Paired against beta=1.00, the published method:"
  echo
  $PY "$ROOT/probes/report_pi05_sweep.py" "$R/canonical_numbers.json"
fi

if want replication; then
  hdr "What agreement an independent re-run should expect"
  $PY "$ROOT/probes/replication_margin.py" || FAILED=1
fi

if [ "$WHICH" = all ]; then
  hdr "Withdrawn claims"
  cat <<'NOTE'
  Three claims from an earlier draft did not survive repetition and are not in
  the results above. They are listed because the pattern -- each was written
  from a single run -- is the reason this repository repeats everything.

    "no benefit on long-horizon tasks"     1 run, 100 eps: +7.0, p=0.25 --
                                           the run behind that figure was NOT
                                           kept, so it is the one number in this
                                           section a reviewer cannot check. The
                                           100-episode long-horizon run that IS
                                           shipped gives +12.0, p=0.058, and the
                                           300-episode one +9.3, p=6e-4. Both
                                           contradict the withdrawn claim.
    "does not transfer to pi0.5"           1 seed: -0.3, p=1.00
                                           -> seed 7: +9.0, p=0.007. Still open:
                                              paired, the sweep puts beta=0.25 at
                                              +8.5 (22/5, p=0.0015, Bonferroni
                                              0.0076) over the published beta=1.
                                              One exploratory sweep, unconfirmed.
    "L2-optimal != success-optimal"        1 control run: +7, p=0.19
                                           -> 5-point sweep is flat within noise
NOTE
  printf '\n  Done. To re-collect any of this on fresh GPUs: ./repro/run_experiment.sh list\n\n'
fi

exit "$FAILED"

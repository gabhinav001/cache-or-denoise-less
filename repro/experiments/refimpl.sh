# Executed-horizon sensitivity in the local cache implementation.
#
# C3ache Table 1 reports LIBERO Goal 96.4% for full compute and for its
# steps [0,7], tau8 cache. Our full success is similar and cache success lower,
# but that does not isolate the cause: checkpoint and implementation can both
# affect robustness to a stale residual.
#
# The reference predicts 32 actions and explicitly allows executing only part
# of a chunk. It does not specify that all 32 are executed. These runs compare
# our replan10 and replan32 settings; neither is established as the reference
# protocol. Both use 20 episodes/task, whereas the reference uses 50 on LIBERO.
#
# Per-block residuals telescope in exact arithmetic, but the local BF16 sequence
# of 30 subtract/add operations need not match one whole-stack subtraction and
# addition. The same-input gate does not test this stale-input equivalence.
# A direct whole-stack comparison remains necessary before excluding granularity.
#
# This experiment changes only executed length within our implementation.
# Whatever the outcome, it does not establish reference-system equivalence.
TITLE="Reference-protocol reconciliation: executed horizon"
SECTION="02b"
GATE=yes
ANALYSIS=none
CKPT=libero_optional_idm_2cam224.pt

# name:replan
RUNS="${RUNS:-replan10:10 replan32:32}"
DEFAULT_SEEDS="$RUNS"

run_one() {   # <name:replan> <outdir>
  local spec="$1" out="$2" name replan
  name=${spec%%:*}; replan=${spec#*:}
  case "$replan" in
    ''|*[!0-9]*) echo "refimpl: replan must be numeric, got '$replan'" >&2; return 2 ;;
  esac

  echo "  -- $name (execute $replan of each predicted chunk, then re-plan)" >&2
  cd "$WAM_ROOT" && python probes/fw_libero.py \
    ckpt="$CKPT_DIR/libero_optional_idm_2cam224.pt" \
    model.redirect_common_files=false \
    EVALUATION.dataset_stats_path="$CKPT_DIR/libero_optional_idm_2cam224_dataset_stats.json" \
    EVALUATION.replan_steps="$replan" \
    seed=42 \
    +PROBE.suite=libero_goal +PROBE.n_tasks=10 +PROBE.n_episodes=20 \
    +PROBE.configs=\'"full,c3ache:8,c3ache:8:s"\' \
    +PROBE.shrink="$REPO/inputs/weights/shrink_idm.json" \
    +PROBE.out="$out/${name}.json" || return 1
}

post_collect() {
  local out="$1"
  [ -n "${DRY_RUN:-}" ] && { echo "ANALYSIS_ONLY=$out"; return 0; }
  echo
  python3 "$REPO/probes/report_refimpl.py" "$out" || return 1
}

# Compare local cache implementations on 500 LIBERO Goal initial states.
#
# The five-seed tau8 headline uses 10 tasks x 10 episodes per seed. The separate
# statescale and matrix experiments already contain 500-state evaluations, but
# neither compares the plain per-block and local whole-stack tau8 caches. This
# experiment supplies that comparison on 10 tasks x 50 initial states.
#
# The local optional-IDM checkpoint is public. Its identity relative to C3ache's
# evaluated checkpoint, and the reference inference mode and executed horizon,
# remain unverified. Our sim_libero config instantiates base FastWAM: these runs
# use first-frame action inference, execute 10 of 32 predicted actions, and wait
# 30 simulator steps. Matching an episode count or a reported success rate does
# not establish that the reference checkpoint or remaining protocol is matched.
#
# WHAT EACH ARM ANSWERS
#   full       local full-compute success on these 500 states.
#   whole:8    local whole-stack implementation: one residual h_L - h_0 per
#              denoising step, cached steps [0,7], refreshed every eight chunks.
#   c3ache:8   local per-block implementation under the same cache schedule.
#
# Arms share initial states and a policy seed within each collection. Their
# execution order is fixed, and prior controls showed arm-list dependence, so
# even a difference between these arms does not by itself isolate granularity.
# Published success values are descriptive references, not equivalence tests.
# A smoke/subset run cannot establish full-collection coverage or reproduction.
#
# BETA=1 THROUGHOUT. The published method is training-free, so no arm here needs
# calibration weights and this experiment has no dependency on stackbeta.
PENDING=yes
TITLE="Local whole-stack and per-block caches on 500 LIBERO Goal states"
SECTION="01d"
GATE=yes
ANALYSIS=none
CKPT=libero_optional_idm_2cam224.pt

ARMS="${ARMS:-full,whole:8,c3ache:8}"
# The target collection: 10 tasks x 50 initial states. NOT the harness's EPISODES,
# which run_experiment.sh exports as 10 for every experiment and would silently
# put this back on the 100-episode subset the whole experiment exists to leave.
# EPISODES_PER_TASK=2 with TASKS=1 is the smoke test that proves the whole-stack
# arm runs at all before five hours are committed to it.
EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
TASKS="${TASKS:-10}"

RUNS="${RUNS:-goal42:libero_goal:42}"
DEFAULT_SEEDS="$RUNS"

run_one() {   # <name:suite:seed> <outdir>
  local spec="$1" out="$2" name suite seed eps
  name=${spec%%:*};  spec=${spec#*:}
  suite=${spec%%:*}; seed=${spec#*:}
  eps="$EPISODES_PER_TASK"
  case "$seed" in
    ''|*[!0-9]*) echo "  !! $name has a non-numeric seed ($seed); refusing" >&2; return 1 ;;
  esac
  echo "  -- $name ($suite, seed $seed, $TASKS tasks x $eps episodes, arms: $ARMS)" >&2
  cd "$WAM_ROOT" && python probes/fw_libero.py \
    ckpt="$CKPT_DIR/libero_optional_idm_2cam224.pt" \
    model.redirect_common_files=false \
    EVALUATION.dataset_stats_path="$CKPT_DIR/libero_optional_idm_2cam224_dataset_stats.json" \
    seed="$seed" \
    +PROBE.suite="$suite" +PROBE.n_tasks="$TASKS" +PROBE.n_episodes="$eps" \
    +PROBE.configs=\'"$ARMS"\' \
    +PROBE.shrink="$REPO/inputs/weights/shrink_idm.json" \
    +PROBE.out="$out/${name}.json" || return 1
}

post_collect() {
  local out="$1"
  [ -n "${DRY_RUN:-}" ] && { echo "ANALYSIS_ONLY=$out"; return 0; }
  echo
  python3 "$REPO/probes/report_published.py" "$out" || return 1
}

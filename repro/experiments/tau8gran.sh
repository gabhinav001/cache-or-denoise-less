# The tau=8 headline again, with the cache granularity as an ARM.
#
# WHY THIS IS SEPARATE FROM tau8.sh. That experiment's five artefacts are
# shipped, and they are reproducible only by the arm list that produced them.
# Changing ARMS there would leave results/tau8/seed*.json claimed by a command
# that no longer regenerates them -- the exact defect check_repro_map exists to
# catch. So the granularity comparison runs beside it, same suite, same seeds,
# same 10 tasks x 10 episodes, writing its own files.
#
# WHAT IT ANSWERS. The paper's plain cache reaches 67.6% where the published
# number is 96.4%, and the repository has been attributing the gap to the method
# on the strength of a telescoping identity. That identity is exact in exact
# arithmetic and the model is BF16; probes/test_cache_semantics.py shows the two
# forms differing while self-replay passes. These five arms separate the two
# explanations directly:
#
#   full            the reference
#   c3ache:8        our per-block plain cache -- the arm that reads 67.6%
#   whole:8         ONE residual for the stack, added once: the published form
#   whole:8:s       that same cache, attenuated with a training-split weight
#   c3ache:8:s      our per-cell attenuated cache -- the proposed method
#
# If whole:8 lands near 96.4 and c3ache:8 does not, the weak baseline is ours and
# the paper's framing has to change. If they agree, granularity is excluded and
# the gap is somewhere else. Either outcome is worth more than another sweep of
# the same two arms.
#
# RUN stackdiff AND stackbeta FIRST. stackdiff says whether the granularities
# differ at all on fixed inputs -- if they agree there, this sweep is unnecessary
# -- and stackbeta produces the training-split weight whole:8:s needs.
PENDING=yes
TITLE="tau=8 with cache granularity as an arm"
SECTION="01c"
GATE=yes
ANALYSIS=none
CKPT=libero_optional_idm_2cam224.pt

ARMS="full,c3ache:8,whole:8,whole:8:s,c3ache:8:s"

RUNS="${RUNS:-\
gran3:libero_goal:3 \
gran7:libero_goal:7 \
gran11:libero_goal:11 \
gran23:libero_goal:23 \
gran42:libero_goal:42}"
DEFAULT_SEEDS="$RUNS"

run_one() {   # <name:suite:seed> <outdir>
  local spec="$1" out="$2" name suite seed
  name=${spec%%:*};  spec=${spec#*:}
  suite=${spec%%:*}; seed=${spec#*:}
  case "$seed" in
    ''|*[!0-9]*) echo "  !! $name has a non-numeric seed ($seed); refusing" >&2; return 1 ;;
  esac
  # The whole-stack weight must exist before an arm asks for it; fw_libero.py
  # refuses the arm rather than silently running it at beta=1, which would file
  # an unattenuated run under an attenuated name.
  if [ -z "${DRY_RUN:-}" ] && [ ! -s "$REPO/inputs/weights/stack_shrink_idm.json" ]; then
    echo "  !! inputs/weights/stack_shrink_idm.json missing." >&2
    echo "     Run ./repro/run_experiment.sh stackbeta and copy its output there." >&2
    return 1
  fi
  if [ -z "${DRY_RUN:-}" ]; then
    # The original shipped fit named a checkpoint without loading its weights.
    # Reject that legacy artifact before starting an expensive model process.
    python3 "$REPO/probes/report_stackbeta.py" \
      "$REPO/inputs/weights/stack_shrink_idm.json" || return 1
  fi
  echo "  -- $name ($suite, seed $seed, 10 tasks x 10 episodes, arms: $ARMS)" >&2
  cd "$WAM_ROOT" && python probes/fw_libero.py \
    ckpt="$CKPT_DIR/libero_optional_idm_2cam224.pt" \
    model.redirect_common_files=false \
    EVALUATION.dataset_stats_path="$CKPT_DIR/libero_optional_idm_2cam224_dataset_stats.json" \
    seed="$seed" \
    +PROBE.suite="$suite" +PROBE.n_tasks=10 +PROBE.n_episodes=10 \
    +PROBE.configs=\'"$ARMS"\' \
    +PROBE.shrink="$REPO/inputs/weights/shrink_idm.json" \
    +PROBE.stack_shrink="$REPO/inputs/weights/stack_shrink_idm.json" \
    +PROBE.out="$out/${name}.json" || return 1
}

post_collect() {
  local out="$1"
  [ -n "${DRY_RUN:-}" ] && { echo "ANALYSIS_ONLY=$out"; return 0; }
  echo
  python3 "$REPO/probes/report_tau8gran.py" "$out" || return 1
}

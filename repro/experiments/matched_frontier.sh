# Matched accuracy-latency frontier across caching granularities and short inference.
#
# Addresses Revision Priority 1 from strict workshop reviews:
# Evaluates short uncached inference, full uncached inference, whole-stack plain caching,
# whole-stack preaggregated attenuation, blockwise plain caching, and blockwise
# attenuated caching under identical experimental conditions:
#   - Checkpoint: libero_optional_idm_2cam224.pt
#   - Task suite: libero_goal, 10 tasks x 5 episodes = 50 matched initial states
#   - Seed: 42
#   - Compilation: compile_action_infer=True (default inductor mode, without reduce-overhead)
#   - Timing: cycle-level end-to-end latency with CUDA synchronization
#   - Process isolation: each arm is executed in an independent, fresh Python process.

TITLE="Matched accuracy-latency frontier (short stock, whole-stack, preaggregated attenuation, blockwise)"
SECTION="10e"
GATE=yes
ANALYSIS=none
CKPT=libero_optional_idm_2cam224.pt

# name:steps:config
RUNS="${RUNS:-\
stock_2step:2:stock \
stock_10step:10:stock \
whole_plain_tau8:10:whole:8 \
whole_agg_tau8:10:whole:8:a \
block_plain_tau8:10:c3ache:8 \
block_shrink_tau8:10:c3ache:8:s}"
DEFAULT_SEEDS="$RUNS"

run_one() {   # <name:steps:config> <outdir>
  local spec="$1" out="$2" name steps cfg
  name=${spec%%:*}; spec=${spec#*:}
  steps=${spec%%:*}; cfg=${spec#*:}

  case "$steps" in
    ''|*[!0-9]*) echo "matched_frontier: steps must be numeric" >&2; return 2 ;;
  esac

  echo "  -- $name ($steps denoising steps, arm: $cfg, 10 tasks x 5 episodes, fresh process)" >&2
  cd "$WAM_ROOT" && python probes/fw_libero.py \
    ckpt="$CKPT_DIR/libero_optional_idm_2cam224.pt" \
    model.redirect_common_files=false \
    EVALUATION.dataset_stats_path="$CKPT_DIR/libero_optional_idm_2cam224_dataset_stats.json" \
    EVALUATION.num_inference_steps="$steps" \
    seed=42 \
    +PROBE.suite=libero_goal +PROBE.n_tasks=10 +PROBE.n_episodes=5 \
    +PROBE.configs=\'"$cfg"\' \
    +PROBE.timing=1 +PROBE.warmup_episodes=1 +PROBE.cycle_times=1 \
    +PROBE.shrink="$REPO/inputs/weights/shrink_idm.json" \
    +PROBE.out="$out/${name}.json" || return 1
}

post_collect() {
  local out="$1"
  [ -n "${DRY_RUN:-}" ] && { echo "ANALYSIS_ONLY=$out"; return 0; }
  echo
  python3 "$REPO/probes/report_matched_frontier.py" "$out" || return 1
}

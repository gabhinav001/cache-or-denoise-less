# Matched evaluation on one A6000 host, each arm in a fresh process:
# Part A: High-powered matched frontier evaluation on 100 states (10 tasks x 10 episodes)
# Part B: Controlled causal schedule-shift intervention on 50 states (10 tasks x 5 episodes)
#
# Spec format: name:ckpt_type:steps:shift:episodes:config
# All arms evaluated with fresh Python process isolation per arm and cycle-level synchronized timing.

TITLE="Matched 100-state frontier and 2x2 checkpoint-by-shift intervention"
SECTION="4-5"
GATE="${GATE:-yes}"
ANALYSIS=none

# Part A: High-Powered Frontier (100 states, 10 tasks x 10 episodes)
PART_A="\
idm_stock_1step:idm:1:1.0:10:stock \
idm_stock_2step:idm:2:1.0:10:stock \
idm_stock_4step:idm:4:1.0:10:stock \
idm_stock_10step:idm:10:1.0:10:stock \
idm_whole_agg_tau8:idm:10:1.0:10:whole:8:a \
idm_whole_plain_tau2:idm:10:1.0:10:whole:2 \
uncond_stock_1step:uncond:1:5.0:10:stock \
uncond_stock_2step:uncond:2:5.0:10:stock \
uncond_stock_4step:uncond:4:5.0:10:stock \
uncond_stock_10step:uncond:10:5.0:10:stock \
uncond_whole_4step_tau4:uncond:4:5.0:10:whole:4 \
uncond_whole_tau8:uncond:10:5.0:10:whole:8"

# Part B: Causal Schedule Shift Intervention (50 states, 10 tasks x 5 episodes)
# 2x2 Factorial: Checkpoint (IDM vs Uncond) x Shift (1.0 vs 5.0) for Stock-2step and Whole-Tau8
PART_B="\
causal_idm_s1_stock2:idm:2:1.0:5:stock \
causal_idm_s5_stock2:idm:2:5.0:5:stock \
causal_idm_s1_whole_tau8:idm:10:1.0:5:whole:8 \
causal_idm_s5_whole_tau8:idm:10:5.0:5:whole:8 \
causal_uncond_s5_stock2:uncond:2:5.0:5:stock \
causal_uncond_s1_stock2:uncond:2:1.0:5:stock \
causal_uncond_s5_whole_tau8:uncond:10:5.0:5:whole:8 \
causal_uncond_s1_whole_tau8:uncond:10:1.0:5:whole:8"

RUNS="${RUNS:-$PART_A $PART_B}"
DEFAULT_SEEDS="$RUNS"

run_one() {   # <name:ckpt_type:steps:shift:episodes:config> <outdir>
  local spec="$1" out="$2" name ckpt_type steps shift n_episodes cfg
  name=${spec%%:*}; spec=${spec#*:}
  ckpt_type=${spec%%:*}; spec=${spec#*:}
  steps=${spec%%:*}; spec=${spec#*:}
  shift=${spec%%:*}; spec=${spec#*:}
  n_episodes=${spec%%:*}; cfg=${spec#*:}

  case "$steps" in
    ''|*[!0-9]*) echo "matched100: steps must be numeric" >&2; return 2 ;;
  esac
  case "$n_episodes" in
    ''|*[!0-9]*) echo "matched100: episodes must be numeric" >&2; return 2 ;;
  esac

  local ckpt_file stats_file task_name shrink_file extra_args=()
  if [ "$ckpt_type" = "idm" ]; then
    ckpt_file="$CKPT_DIR/libero_optional_idm_2cam224.pt"
    stats_file="$CKPT_DIR/libero_optional_idm_2cam224_dataset_stats.json"
    task_name="libero_optional_idm_2cam224_1e-4"
    extra_args+=("+EVALUATION.action_infer_mode=first_frame")
    shrink_file="$REPO/inputs/weights/shrink_idm.json"
  elif [ "$ckpt_type" = "uncond" ]; then
    ckpt_file="$CKPT_DIR/libero_uncond_2cam224.pt"
    stats_file="$CKPT_DIR/libero_uncond_2cam224_dataset_stats.json"
    task_name="libero_uncond_2cam224_1e-4"
    shrink_file="$REPO/inputs/weights/shrink_fw.json"
  else
    echo "Unknown ckpt_type: $ckpt_type" >&2; return 2
  fi

  echo "  -- $name ($ckpt_type ckpt, $steps steps, shift: $shift, $n_episodes ep/task, arm: $cfg, fresh process)" >&2
  cd "$WAM_ROOT" && python probes/fw_libero.py \
    task="$task_name" \
    ckpt="$ckpt_file" \
    model.redirect_common_files=false \
    EVALUATION.dataset_stats_path="$stats_file" \
    EVALUATION.sigma_shift="$shift" \
    EVALUATION.num_inference_steps="$steps" \
    seed=42 \
    ${extra_args[@]+"${extra_args[@]}"} \
    +PROBE.suite=libero_goal +PROBE.n_tasks=10 +PROBE.n_episodes="$n_episodes" \
    +PROBE.configs=\'"$cfg"\' \
    +PROBE.timing=1 +PROBE.warmup_episodes=1 +PROBE.cycle_times=1 \
    +PROBE.shrink="$shrink_file" \
    +PROBE.out="$out/${name}.json" || return 1
}

post_collect() {
  local out="$1"
  [ -n "${DRY_RUN:-}" ] && { echo "ANALYSIS_ONLY=$out"; return 0; }
  echo
  python3 "$REPO/probes/report_matched100.py" "$out" || return 1
}

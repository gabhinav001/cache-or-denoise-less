# Matched 50-state evaluation on one A6000 host:
# 1. Re-timed whole-stack preaggregated attenuation (cache aggregation inside timed cycle)
# 2. Explicit block skipping (c3ache:8:sx) on conditional IDM
# 3. Short uncached inference (2 steps) under native unconditional Fast-WAM (shift 5.0)
# 4. Full uncached inference (10 steps) under native unconditional Fast-WAM (shift 5.0)
# 5. Native whole-stack caching (10 steps, tau=8) under native unconditional Fast-WAM (shift 5.0)
#
# All arms evaluated on 50 matched initial states (libero_goal, 10 tasks x 5 episodes)
# with fresh Python process isolation per arm and cycle-level synchronized timing.

TITLE="Matched 50-state evaluation (whole-stack sweep, blockwise hooks, both checkpoints)"
SECTION="4-5"
GATE="${GATE:-yes}"
ANALYSIS=none

# spec: name:ckpt_type:steps:config
RUNS="${RUNS:-\
idm_stock_1step:idm:1:stock \
idm_stock_2step:idm:2:stock \
idm_stock_4step:idm:4:stock \
idm_stock_10step:idm:10:stock \
idm_whole_plain_tau0:idm:10:whole:0 \
idm_whole_plain_tau2:idm:10:whole:2 \
idm_whole_plain_tau4:idm:10:whole:4 \
idm_whole_plain_tau8:idm:10:whole:8 \
idm_whole_agg_tau4:idm:10:whole:4:a \
idm_whole_agg_tau8:idm:10:whole:8:a \
idm_whole_4step_tau4:idm:4:whole:4 \
idm_block_plain_tau8:idm:10:c3ache:8 \
idm_block_skip_tau8:idm:10:c3ache:8:sx \
uncond_stock_1step:uncond:1:stock \
uncond_stock_2step:uncond:2:stock \
uncond_stock_4step:uncond:4:stock \
uncond_stock_10step:uncond:10:stock \
uncond_whole_tau0:uncond:10:whole:0 \
uncond_whole_tau2:uncond:10:whole:2 \
uncond_whole_tau4:uncond:10:whole:4 \
uncond_whole_tau8:uncond:10:whole:8 \
uncond_whole_4step_tau4:uncond:4:whole:4}"
DEFAULT_SEEDS="$RUNS"

run_one() {   # <name:ckpt_type:steps:config> <outdir>
  local spec="$1" out="$2" name ckpt_type steps cfg
  name=${spec%%:*}; spec=${spec#*:}
  ckpt_type=${spec%%:*}; spec=${spec#*:}
  steps=${spec%%:*}; cfg=${spec#*:}

  case "$steps" in
    ''|*[!0-9]*) echo "matched50: steps must be numeric" >&2; return 2 ;;
  esac

  local ckpt_file stats_file task_name shift shrink_file extra_args=()
  if [ "$ckpt_type" = "idm" ]; then
    ckpt_file="$CKPT_DIR/libero_optional_idm_2cam224.pt"
    stats_file="$CKPT_DIR/libero_optional_idm_2cam224_dataset_stats.json"
    task_name="libero_optional_idm_2cam224_1e-4"
    shift="1.0"
    extra_args+=("+EVALUATION.action_infer_mode=first_frame")
    shrink_file="$REPO/inputs/weights/shrink_idm.json"
  elif [ "$ckpt_type" = "uncond" ]; then
    ckpt_file="$CKPT_DIR/libero_uncond_2cam224.pt"
    stats_file="$CKPT_DIR/libero_uncond_2cam224_dataset_stats.json"
    task_name="libero_uncond_2cam224_1e-4"
    shift="5.0"
    shrink_file="$REPO/inputs/weights/shrink_fw.json"
  else
    echo "Unknown ckpt_type: $ckpt_type" >&2; return 2
  fi

  echo "  -- $name ($ckpt_type checkpoint, $steps denoising steps, shift: $shift, arm: $cfg, 10 tasks x 5 episodes, fresh process)" >&2
  cd "$WAM_ROOT" && python probes/fw_libero.py \
    task="$task_name" \
    ckpt="$ckpt_file" \
    model.redirect_common_files=false \
    EVALUATION.dataset_stats_path="$stats_file" \
    EVALUATION.sigma_shift="$shift" \
    EVALUATION.num_inference_steps="$steps" \
    seed=42 \
    ${extra_args[@]+"${extra_args[@]}"} \
    +PROBE.suite=libero_goal +PROBE.n_tasks=10 +PROBE.n_episodes=5 \
    +PROBE.configs=\'"$cfg"\' \
    +PROBE.timing=1 +PROBE.warmup_episodes=1 +PROBE.cycle_times=1 \
    +PROBE.shrink="$shrink_file" \
    +PROBE.out="$out/${name}.json" || return 1
}

post_collect() {
  local out="$1"
  [ -n "${DRY_RUN:-}" ] && { echo "ANALYSIS_ONLY=$out"; return 0; }
  echo
  python3 "$REPO/probes/report_matched50.py" "$out" || return 1
}

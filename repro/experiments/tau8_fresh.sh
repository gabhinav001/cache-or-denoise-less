# tau=8 replication with each arm executed in an independent, fresh process.
#
# Isolates the tau=8 attenuation effect from
# intra-process arm-order dependence and execution-context bias.

TITLE="tau=8 plain vs attenuated caching in fresh processes"
SECTION="01b"
GATE=yes
ANALYSIS=none
CKPT=libero_optional_idm_2cam224.pt

# name:suite:seed:config
RUNS="${RUNS:-\
seed42_plain:libero_goal:42:c3ache:8 \
seed42_attenuated:libero_goal:42:c3ache:8:s \
seed7_plain:libero_goal:7:c3ache:8 \
seed7_attenuated:libero_goal:7:c3ache:8:s}"
DEFAULT_SEEDS="$RUNS"

run_one() {   # <name:suite:seed:config> <outdir>
  local spec="$1" out="$2" name suite seed cfg
  name=${spec%%:*};  spec=${spec#*:}
  suite=${spec%%:*}; spec=${spec#*:}
  seed=${spec%%:*};  cfg=${spec#*:}
  case "$seed" in
    ''|*[!0-9]*) echo "  !! $name has a non-numeric seed ($seed); refusing" >&2; return 1 ;;
  esac
  echo "  -- $name ($suite, seed $seed, arm: $cfg, 10 tasks x 10 episodes, fresh process)" >&2
  cd "$WAM_ROOT" && python probes/fw_libero.py \
    ckpt="$CKPT_DIR/libero_optional_idm_2cam224.pt" \
    model.redirect_common_files=false \
    EVALUATION.dataset_stats_path="$CKPT_DIR/libero_optional_idm_2cam224_dataset_stats.json" \
    seed="$seed" \
    +PROBE.suite="$suite" +PROBE.n_tasks=10 +PROBE.n_episodes=10 \
    +PROBE.configs=\'"$cfg"\' \
    +PROBE.shrink="$REPO/inputs/weights/shrink_idm.json" \
    +PROBE.out="$out/${name}.json" || return 1
}

post_collect() {
  local out="$1"
  [ -n "${DRY_RUN:-}" ] && { echo "ANALYSIS_ONLY=$out"; return 0; }
  echo
  python3 "$REPO/probes/report_tau8_fresh.py" "$out" || return 1
}

# The physics-version hazard. BOTH arms must run on the SAME machine: an earlier
# estimate compared two machines and so confounded the mujoco version with
# cross-machine variance, which is itself ~15 points.
# "Seeds" here are mujoco versions; the script reinstalls between arms.
TITLE="mujoco version hazard (same host, both arms)"
SECTION="08"
DEFAULT_SEEDS="3.3.2 2.3.7"
EPISODES=8
GATE=no
ANALYSIS=mujoco

# The checkpoint this experiment loads. Preflight verifies THIS file:
# four experiments use the unconditional checkpoint, and the check was
# hardcoded to the other one, so they failed inside model loading on a
# host that preflight had just called ready.
CKPT=libero_optional_idm_2cam224.pt

run_one() {
  local ver="$1" out="$2"
  echo "  -- installing mujoco==$ver" >&2
  pip install -q "mujoco==$ver" || return 1
  cd "$WAM_ROOT" && python probes/fw_libero.py \
    ckpt="$CKPT_DIR/libero_optional_idm_2cam224.pt" \
    model.redirect_common_files=false \
    EVALUATION.dataset_stats_path="$CKPT_DIR/libero_optional_idm_2cam224_dataset_stats.json" \
    seed=42 \
    +PROBE.suite=libero_goal +PROBE.n_tasks=10 +PROBE.n_episodes="$EPISODES" \
    +PROBE.configs=\'full\' \
    +PROBE.shrink="$REPO/inputs/weights/shrink_idm.json" \
    +PROBE.out="$out/q4_mujoco_$(echo "$ver" | tr -d .).json"
}

# This experiment deliberately downgrades the physics engine, and a downgrade
# outlives it: the next simulator experiment on the same host inherits 2.3.7 and
# collects data rendered against the wrong mujoco. That happened here -- curve ran
# next and its preflight refused the host, which is the check working, but only
# after the box had been paid for and the queue had stalled.
#
# run_experiment.sh calls this after the last seed if it exists.
restore_env() {
  echo "  -- restoring mujoco 3.3.2 (this experiment downgrades it)" >&2
  pip install -q mujoco==3.3.2 2>&1 | tail -2 >&2 || true
}

# Does correcting the weight for staleness recover the error left on the table?
#
# The shipped coefficients are fit against the residual ONE cycle back and then
# applied unchanged however stale the cache is. At tau=2 that is exactly right --
# the cache is never more than one cycle old. At tau=4 it is not: the cache is
# reused at staleness 1, 2 and 3, where the fitted optima are 0.717 / 0.492 /
# 0.372. So on two of every three cached cycles the earlier runs applied roughly
# twice the correct weight and under-attenuated.
#
# Offline this is worth about 3 points of residual error (+13.4% -> +16.4%). This
# asks whether it reaches task success. Every published closed-loop number for
# this method therefore rests on a deliberately suboptimal weight, which makes
# them a LOWER BOUND on what the correction can do.
#
# Three arms, all at tau=4, paired within each run:
#   c3ache:4      no correction at all      (the published method)
#   c3ache:4:s    one weight per cell       (what we reported)
#   c3ache:4:k    weight per cell AND per staleness   (the question)
TITLE="Staleness-corrected attenuation weights"
SECTION="04b"
DEFAULT_SEEDS="42 7 11"
EPISODES=10
GATE=yes
ANALYSIS=closedloop

# The checkpoint this experiment loads. Preflight verifies THIS file:
# four experiments use the unconditional checkpoint, and the check was
# hardcoded to the other one, so they failed inside model loading on a
# host that preflight had just called ready.
CKPT=libero_optional_idm_2cam224.pt

run_one() {   # <seed> <outdir>
  local seed="$1" out="$2"
  # the :s arm reads shrink_idm.json, the :k arm reads shrink_idm_k.json. The
  # harness takes one --shrink file, so the two arms run as separate passes.
  cd "$WAM_ROOT" && python probes/fw_libero.py \
    ckpt="$CKPT_DIR/libero_optional_idm_2cam224.pt" \
    model.redirect_common_files=false \
    EVALUATION.dataset_stats_path="$CKPT_DIR/libero_optional_idm_2cam224_dataset_stats.json" \
    seed="$seed" \
    +PROBE.suite=libero_goal +PROBE.n_tasks=10 +PROBE.n_episodes="$EPISODES" \
    +PROBE.configs=\'full,c3ache:4,c3ache:4:s\' \
    +PROBE.shrink="$REPO/inputs/weights/shrink_idm.json" \
    +PROBE.out="$out/stale_fixed_seed${seed}.json" || return 1

  cd "$WAM_ROOT" && python probes/fw_libero.py \
    ckpt="$CKPT_DIR/libero_optional_idm_2cam224.pt" \
    model.redirect_common_files=false \
    EVALUATION.dataset_stats_path="$CKPT_DIR/libero_optional_idm_2cam224_dataset_stats.json" \
    seed="$seed" \
    +PROBE.suite=libero_goal +PROBE.n_tasks=10 +PROBE.n_episodes="$EPISODES" \
    +PROBE.configs=\'full,c3ache:4,c3ache:4:s\' \
    +PROBE.shrink="$REPO/inputs/weights/shrink_idm_k.json" \
    +PROBE.out="$out/stale_aware_seed${seed}.json"
}

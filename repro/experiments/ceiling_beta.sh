# How much task success can ANY scalar attenuation weight recover?
#
# Separates two failures that the closed-loop numbers cannot tell apart:
#   (a) the weight is badly estimated -> better estimation would help
#   (b) a scalar weight cannot do better -> estimation is not the bottleneck
#
# The `:o` arm picks, for every cached cell on every cycle, the exact
# least-squares weight for THAT cycle, using the true residual. It is NOT
# deployable and saves NO compute -- every block still runs -- which is the
# point: it bounds what the whole family of per-cell scalar weights could ever
# achieve on task success.
#
# Reading it:
#   oracle ~= full compute  -> attenuation can close the gap; chase the estimator
#   oracle still well below -> a scalar weight is the wrong instrument, and the
#                              remaining error is angular, exactly as the
#                              geometry predicts
TITLE="Ceiling: the best any scalar weight could do"
SECTION="04c"
DEFAULT_SEEDS="42 7"
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
  cd "$WAM_ROOT" && python probes/fw_libero.py \
    ckpt="$CKPT_DIR/libero_optional_idm_2cam224.pt" \
    model.redirect_common_files=false \
    EVALUATION.dataset_stats_path="$CKPT_DIR/libero_optional_idm_2cam224_dataset_stats.json" \
    seed="$seed" \
    +PROBE.suite=libero_goal +PROBE.n_tasks=10 +PROBE.n_episodes="$EPISODES" \
    +PROBE.configs=\'full,c3ache:4,c3ache:4:s,c3ache:4:o\' \
    +PROBE.shrink="$REPO/inputs/weights/shrink_idm.json" \
    +PROBE.out="$out/ceilbeta_seed${seed}.json"
}

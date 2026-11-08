# Does attenuation move the compute/quality frontier, or only help at operating
# points nobody should use? tau=8 (70% of block evaluations saved) costs ~28
# points of success, which no practitioner would accept. tau=2 (40% saved) costs
# ~3 points and is usable -- and is where attenuation was worth -1.2 points in
# the single run that measured it. tau=4 sits between them and had exactly one
# measurement, on a run whose plain baseline drew unusually high.
# This settles both at three seeds, with full compute in every run as a control.
TITLE="Compute/quality frontier at tau=2 and tau=4"
SECTION="04"
# All five seeds the published frontier figures pool, not the three this started
# at. The pooled p-value is computed over whatever is in the directory, so a
# default that collects three would reproduce the table's shape with a different
# sample behind it -- the most confusing way for a reproduction to disagree.
DEFAULT_SEEDS="42 7 11 23 3"
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
    +PROBE.configs=\'full,c3ache:2,c3ache:2:s,c3ache:4,c3ache:4:s\' \
    +PROBE.shrink="$REPO/inputs/weights/shrink_idm.json" \
    +PROBE.out="$out/frontier_seed${seed}.json"
}

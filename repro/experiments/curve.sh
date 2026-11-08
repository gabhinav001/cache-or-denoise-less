# Success against compute across refresh intervals, including tau=0 (never
# THE ATTENUATED tau=0 ARM IS HERE FOR A REPRODUCIBILITY REASON. The staleness
# table reports a tau=0 row with both a plain arm (12.5%) and an attenuated one
# (18.8%), and until now only the plain arm was regenerable: this experiment
# produced c3ache:0 but not c3ache:0:s, so the +6.3 delta came solely from
# results/audit/libero_shrink.json -- an early run that predates the scripted
# experiments and that nothing reproduces. A reviewer could regenerate half the
# row. Adding c3ache:0:s makes the comparison reproducible. The shipped numbers
# still come from that early run and are reported as a single collection, which
# is why the table marks tau=0 as not established.
# refresh) -- the setting behind the published speedup.
TITLE="Compute-success curve across refresh intervals"
SECTION="06"
DEFAULT_SEEDS="42 7"
EPISODES=8
GATE=yes
ANALYSIS=curve

# The checkpoint this experiment loads. Preflight verifies THIS file:
# four experiments use the unconditional checkpoint, and the check was
# hardcoded to the other one, so they failed inside model loading on a
# host that preflight had just called ready.
CKPT=libero_optional_idm_2cam224.pt

run_one() {
  local seed="$1" out="$2" name
  # The two shipped artefacts for this experiment have unrelated names: seed 42
  # is libero_tau_mj332.json and seed 7 is curve_seed7.json. verify.sh is given
  # both paths explicitly rather than a directory to glob, so a regenerated file
  # under any other name is simply never read -- the reviewer gets a clean run
  # and verify.sh goes on reporting the shipped numbers.
  if [ "$seed" = 42 ]; then name=libero_tau_mj332; else name="curve_seed$seed"; fi
  cd "$WAM_ROOT" && python probes/fw_libero.py \
    ckpt="$CKPT_DIR/libero_optional_idm_2cam224.pt" \
    model.redirect_common_files=false \
    EVALUATION.dataset_stats_path="$CKPT_DIR/libero_optional_idm_2cam224_dataset_stats.json" \
    seed="$seed" \
    +PROBE.suite=libero_goal +PROBE.n_tasks=10 +PROBE.n_episodes="$EPISODES" \
    +PROBE.configs=\'full,c3ache:1,c3ache:2,c3ache:4,c3ache:8,c3ache:0,c3ache:0:s\' \
    +PROBE.shrink="$REPO/inputs/weights/shrink_idm.json" \
    +PROBE.out="$out/$name.json"
}

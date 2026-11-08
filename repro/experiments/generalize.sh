# Does the correction survive a shift in the observations?
#
# Every other experiment runs the policy on exactly the distribution its weights
# were fit on. The attenuation coefficients are fit offline on clean replay
# episodes, so the question that decides whether any of this is deployable is
# whether they still help when the world does not look like the fitting data.
#
# Two of these act on the rendered image and two act on the simulator, so shading
# and parallax change rather than only pixel statistics. Magnitudes come from
# probes/calibrate_perturb.py, which measures the pixel change each one actually
# produces -- a perturbation the policy cannot see tests nothing, and one that
# floors full compute tests nothing either. Run it before changing them.
#
# The control is in every arm: `full` runs under the SAME perturbation, so the
# run reports both whether the shift is real (full compute degrades) and whether
# the correction survives it (the attenuated-minus-cached delta holds up). A
# perturbation that leaves full compute untouched was not a shift at all, and its
# row should be read as such.
#
# "Seeds" here are perturbations; the environment seed is fixed so that what
# varies between rows is the shift, not the draw.
TITLE="Robustness of the correction under observation shift"
SECTION="05b"
# Magnitudes are CALIBRATED, not guessed: probes/calibrate_perturb.py renders the
# same state with and without each one and reports the pixel change. The first
# attempt used light:0.10 (inert -- 100% full compute, tests nothing), cam:0.02
# and bright:0.15 (both severe enough to flatten the policy, which tests nothing
# either). These land in the 6-13 mean |dpix| band: visible, not destructive.
# These are the settings the shipped results/generalize/ artefacts were collected
# at, so the default run reproduces that set exactly rather than a similar one.
#
# light:0.10 is kept deliberately even though it is inert. It is the control for
# the calibration argument: it shows what a perturbation that fails to move full
# compute looks like, which is the failure the other magnitudes were chosen to
# avoid. The analysis reads it as such -- an arm whose full-compute column does
# not drop is reported and excluded from the pooled test.
#
# noise:0.05 is implemented and accepted here but was never collected, so it is
# not in this list; adding it produces a result with nothing shipped to compare.
DEFAULT_SEEDS="none light:0.10 light:0.50 cam:0.005 bright:0.05"
EPISODES=6
GATE=yes
ANALYSIS=generalize

# The checkpoint this experiment loads. Preflight verifies THIS file:
# four experiments use the unconditional checkpoint, and the check was
# hardcoded to the other one, so they failed inside model loading on a
# host that preflight had just called ready.
CKPT=libero_optional_idm_2cam224.pt

run_one() {   # <perturbation> <outdir>
  local pert="$1" out="$2" tag
  tag=$(echo "$pert" | tr ':.' '__')
  cd "$WAM_ROOT" && python probes/fw_libero.py \
    ckpt="$CKPT_DIR/libero_optional_idm_2cam224.pt" \
    model.redirect_common_files=false \
    EVALUATION.dataset_stats_path="$CKPT_DIR/libero_optional_idm_2cam224_dataset_stats.json" \
    seed=42 \
    +PROBE.suite=libero_goal +PROBE.n_tasks=10 +PROBE.n_episodes="$EPISODES" \
    +PROBE.configs=\'full,c3ache:4,c3ache:4:s\' \
    +PROBE.perturb="$pert" \
    +PROBE.shrink="$REPO/inputs/weights/shrink_idm.json" \
    +PROBE.out="$out/gen_${tag}.json"
}

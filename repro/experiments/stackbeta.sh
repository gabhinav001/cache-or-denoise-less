# Fit the whole-stack attenuation weight, on the training split only.
#
# Needed before the tau8 rerun can compare granularities honestly: the per-cell
# coefficients are fit on 18 of 24 replay episodes, so a whole-stack weight fit
# on all 24 would win on calibration budget rather than on granularity. This
# holds out every fourth episode exactly as the per-cell fit does and reports the
# held-out weights beside the training ones.
#
# No evaluation rollouts are read. The probe takes a replay manifest and recorded
# video; there is no code path in it that opens a results file.
#
# Deterministic given the episode set, so a meaningful repeat is a DIFFERENT
# MACHINE, not another run here.
TITLE="Whole-stack attenuation weight (training split)"
SECTION="A9b"
DEFAULT_SEEDS="primary"
NEEDS_SIM=no
GATE=no
ANALYSIS=none
CKPT=libero_optional_idm_2cam224.pt
MANIFEST="${MANIFEST:-episodes24_idm.json}"
VIDEO_ROOT="${VIDEO_ROOT:-/workspace/libero}"

ckpt_for_seed() { echo "$CKPT"; }
manifest_for_seed() { echo "$MANIFEST"; }

run_one() {   # <tag> <outdir>
  local out="$2"
  cd "$WAM_ROOT" && python probes/fw_stackbeta.py \
    ckpt="$CKPT_DIR/$CKPT" model.redirect_common_files=false \
    EVALUATION.dataset_stats_path="$CKPT_DIR/libero_optional_idm_2cam224_dataset_stats.json" \
    EVALUATION.num_inference_steps=10 \
    +PROBE.manifest="$REPO/inputs/manifests/$MANIFEST" \
    +PROBE.video_root="$VIDEO_ROOT" \
    +PROBE.max_age=7 +PROBE.holdout_every=4 \
    +PROBE.out="$out/stack_shrink_idm.json" || return 1
}

post_collect() {
  local out="$1"
  [ -n "${DRY_RUN:-}" ] && { echo "ANALYSIS_ONLY=$out"; return 0; }
  echo "  Copy $out/stack_shrink_idm.json to inputs/weights/ before running tau8."
}

# Is our per-block cache the published whole-stack cache? (diagnostic, no sweep)
#
# The repository has been treating the telescoping identity as proof that our
# beta=1 substitution IS C3ache's. It is not proof: the identity is exact in
# exact arithmetic, the model runs in BF16, and probes/test_cache_semantics.py
# shows the two forms differing by ~2e-3 on tensors of the right shape while
# self-replay still passes. Until this runs, a 67.6% cached arm against their
# 96.4% cannot be attributed cleanly to the method rather than to our implementation of
# it.
#
# THIS IS A DIAGNOSTIC, NOT A SUCCESS SWEEP. No simulator, no rollouts, no seeds
# to average: fixed recorded observations, a cache filled at one cycle and reused
# at ages 1, 4 and 7, and a direct comparison of the hidden states and actions
# each substitution produces. Full compute is run twice so every claimed
# difference has a noise floor to clear. Run this BEFORE any further success
# collection -- another large sweep first would reproduce the same ambiguity at
# the same cost.
#
# Deterministic given the episode set, so a meaningful repeat is a DIFFERENT
# MACHINE, not another run here.
# Written and validated on CPU (probes/test_cache_semantics.py exercises both
# harnesses), not yet collected on a GPU host. Remove this line when the
# artefact ships.
# COMPILE defaults to 0. The granularity question is answerable eager, and the
# compiled path puts Python state mutation inside a fullgraph region where a
# graph break would change what is being measured rather than just slow it.
# COMPILE=0,1 adds the compiled variant once the eager answer is in hand;
# armorder tests compilation directly and is the better instrument for it.
PENDING=yes
TITLE="Per-block against whole-stack substitution"
SECTION="A9"
DEFAULT_SEEDS="primary"
NEEDS_SIM=no
GATE=no
ANALYSIS=none
# The deployment checkpoint: the question is about the policy that runs closed
# loop, not the unconditional one the geometry study uses.
CKPT=libero_optional_idm_2cam224.pt
MANIFEST="${MANIFEST:-episodes24_idm.json}"
VIDEO_ROOT="${VIDEO_ROOT:-/workspace/libero}"

ckpt_for_seed() { echo "$CKPT"; }
manifest_for_seed() { echo "$MANIFEST"; }

run_one() {   # <tag> <outdir>
  local tag="$1" out="$2"
  cd "$WAM_ROOT" && python probes/fw_stackdiff.py \
    ckpt="$CKPT_DIR/$CKPT" model.redirect_common_files=false \
    EVALUATION.dataset_stats_path="$CKPT_DIR/libero_optional_idm_2cam224_dataset_stats.json" \
    EVALUATION.num_inference_steps=10 \
    +PROBE.manifest="$REPO/inputs/manifests/$MANIFEST" \
    +PROBE.video_root="$VIDEO_ROOT" \
    +PROBE.n_episodes="${EPISODES:-6}" \
    +PROBE.ages=\'1,4,7\' +PROBE.anchors=0 \
    +PROBE.compile=\'"${COMPILE:-0}"\' \
    +PROBE.out="$out/fw_stackdiff.json" || return 1
}

post_collect() {
  local out="$1"
  [ -n "${DRY_RUN:-}" ] && { echo "ANALYSIS_ONLY=$out"; return 0; }
  echo
  python3 "$REPO/probes/report_stackdiff.py" "$out/fw_stackdiff.json" || return 1
}

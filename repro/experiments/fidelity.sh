# Does the residual-level gain reach the action? Open loop, held-out cycles only.
# Deterministic, so a meaningful repeat is a different machine.
TITLE="Open-loop action fidelity"
SECTION="05"
# Writes the filename the shipped artefact carries. The analysis for this
# section is given explicit paths rather than a directory to glob, so a
# regenerated file under any other name is simply not read -- the reviewer
# gets a clean run and verify.sh still reports the shipped numbers.
DEFAULT_SEEDS="fw_shrink_open fw_shrink_open_repro"
NEEDS_SIM=no
GATE=no
ANALYSIS=fidelity

# The checkpoint this experiment loads. Preflight verifies THIS file:
# four experiments use the unconditional checkpoint, and the check was
# hardcoded to the other one, so they failed inside model loading on a
# host that preflight had just called ready.
CKPT=libero_uncond_2cam224.pt

# Recorded episodes this experiment replays. Overridable: the path was
# hardcoded to the machine it was first run on and documented nowhere, so a
# reviewer holding the videos anywhere else could not run it and got no
# warning until the probe failed.
VIDEO_ROOT="${VIDEO_ROOT:-/workspace/libero}"

run_one() {
  local tag="$1" out="$2"
  cd "$WAM_ROOT" && python probes/fw_shrink_open.py \
    ckpt="$CKPT_DIR/libero_uncond_2cam224.pt" model.redirect_common_files=false \
    +PROBE.manifest="$REPO/inputs/manifests/episodes50.json" \
    +PROBE.video_root="$VIDEO_ROOT" \
    +PROBE.shrink="$REPO/inputs/weights/shrink_fw.json" \
    +PROBE.stride=10 +PROBE.skip_steps=8 \
    +PROBE.out="$out/${tag}.json"
}

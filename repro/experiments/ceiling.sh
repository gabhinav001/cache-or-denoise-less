# The compiled-vs-eager speedup ceiling. A reported 2.51x exceeds what caching can
# reach over a compiled baseline; the eager ceiling accommodates it. Hardware
# dependent, so run it on two machines.
TITLE="Compile ceiling (denoise-step sweep)"
SECTION="06"
# Writes the filename the shipped artefact carries. The analysis for this
# section is given explicit paths rather than a directory to glob, so a
# regenerated file under any other name is simply not read -- the reviewer
# gets a clean run and verify.sh still reports the shipped numbers.
# Two artefacts: the measurement and its repeat on a SECOND MACHINE. This is a
# deterministic timing measurement, so a repeat on the same host proves nothing --
# verify.sh is given both paths and reports the spread between them.
DEFAULT_SEEDS="vast_fw_step_sweep stepsweep_a100_uncond_v1"
NEEDS_SIM=no
GATE=no
ANALYSIS=ceiling

# The checkpoint this experiment loads. Preflight verifies THIS file:
# four experiments use the unconditional checkpoint, and the check was
# hardcoded to the other one, so they failed inside model loading on a
# host that preflight had just called ready.
CKPT=libero_uncond_2cam224.pt

run_one() {
  local tag="$1" out="$2"
  cd "$WAM_ROOT" && python probes/step_sweep.py \
    ckpt="$CKPT_DIR/libero_uncond_2cam224.pt" model.redirect_common_files=false \
    +PROBE.out="$out/${tag}.json"
}

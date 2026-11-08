# Five ways to choose WHICH 240 of 300 cells to cache, at an identical compute
# budget. Answers whether "cache the most predictable cells" -- the intuitive
# criterion -- is any good. It is not: an oracle placement using ground truth
# unavailable at deployment still loses badly to the fixed published one.
# Deterministic given the episode set, so a meaningful repeat is a different
# machine, not another run here.
# THE ORACLE ARM IS NOT REPRODUCIBLE FROM THIS REPOSITORY.
# The oracle placement arm cannot be reproduced here. probes/fw_scale.py reads a
# cell set from +PROBE.oracle (default /workspace/oracle_cells.json); that file
# was produced once, off-line, and was never committed -- nothing in this
# repository generates it. A fresh collection therefore has four placements where
# results/audit/fw_scale_a100.json has five, and the oracle-vs-random comparison
# quoted in the write-up cannot be re-derived.
#
# What still reproduces: the four fixed placements (c3ache, depth, late, random)
# at a matched budget of 240 skipped cells, which is the comparison the
# "predictability is a weak criterion" claim mostly rests on. fw_scale.py now
# warns loudly rather than dropping the arm in silence, and report_placement.py
# reports its absence instead of raising.
TITLE="Cache placement at a matched compute budget"
SECTION="A5"
# Writes the filename the shipped artefact carries. The analysis for this
# section is given explicit paths rather than a directory to glob, so a
# regenerated file under any other name is simply not read -- the reviewer
# gets a clean run and verify.sh still reports the shipped numbers.
DEFAULT_SEEDS="fw_scale_a100 place_repro"
NEEDS_SIM=no
GATE=no
ANALYSIS=placement

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
  cd "$WAM_ROOT" && python probes/fw_scale.py \
    ckpt="$CKPT_DIR/libero_uncond_2cam224.pt" model.redirect_common_files=false \
    +PROBE.manifest="$REPO/inputs/manifests/episodes50.json" \
    +PROBE.video_root="$VIDEO_ROOT" \
    +PROBE.stride=10 +PROBE.repeats=3 \
    +PROBE.out="$out/${tag}.json"
}

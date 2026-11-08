# Is one cached arm's rollout invariant to its POSITION and to compilation?
#
# SEPARATE FROM rngdrift.sh for the same reason tau8gran.sh is separate from
# tau8.sh: that experiment's four artefacts are shipped and are reproducible only
# by the run list that produced them. These six runs are new, so they live here.
#
# WHAT rngdrift LEFT OPEN. Its four runs vary arm-list LENGTH with the target arm
# in a fixed position, so "reseed on makes the lists agree" cannot separate "the
# list is longer" from "the arm ran later", and it says nothing about
# compilation. The audit needed both excluded: a cached arm changed 3 of 40
# outcomes and 36 of 40 episode lengths between collections that already had the
# per-episode reseed on, and compiler behaviour was left as a live candidate.
#
# THE DESIGN. One target arm -- the tau=8 attenuated cache, the deployment
# configuration -- placed alone, first, and last in a reversed list, each with
# compilation off and on. Six fresh processes, one arm list each, same seed, same
# 2 tasks x 10 episodes. If the target's rollouts are bit-identical across all
# six, position and compilation are both excluded. If they are not, whichever
# dimension moves is named, and every multi-arm collection in this repository
# has to be re-run arm-isolated before its arms may be compared.
#
# Small on purpose: the question is whether the rollouts are IDENTICAL, which a
# single differing episode settles. It is not a success measurement.
PENDING=yes
TITLE="Arm position and compilation as carriers (A8b)"
SECTION="A8b"
GATE=yes
ANALYSIS=none
CKPT=libero_optional_idm_2cam224.pt

TARGET="c3ache:8:s"
AO_FIRST="$TARGET,full,c3ache:2,c3ache:4"
AO_LAST="c3ache:4,c3ache:2,full,$TARGET"

# name:configs:compile
RUNS="${RUNS:-\
ao_solo_nc:$TARGET:0 \
ao_first_nc:$AO_FIRST:0 \
ao_last_nc:$AO_LAST:0 \
ao_solo_c:$TARGET:1 \
ao_first_c:$AO_FIRST:1 \
ao_last_c:$AO_LAST:1}"
DEFAULT_SEEDS="$RUNS"

run_one() {   # <name:configs:compile> <outdir>
  local spec="$1" out="$2" name cfgs cmp
  name=${spec%%:*};  spec=${spec#*:}
  cmp=${spec##*:};   cfgs=${spec%:*}
  echo "  -- $name (compile $cmp, arms: $cfgs)" >&2
  cd "$WAM_ROOT" && python probes/fw_libero.py \
    ckpt="$CKPT_DIR/libero_optional_idm_2cam224.pt" \
    model.redirect_common_files=false \
    EVALUATION.dataset_stats_path="$CKPT_DIR/libero_optional_idm_2cam224_dataset_stats.json" \
    seed=42 \
    EVALUATION.compile_action_infer="$([ "$cmp" = 1 ] && echo true || echo false)" \
    +PROBE.suite=libero_goal +PROBE.n_tasks=2 +PROBE.n_episodes=10 \
    +PROBE.configs=\'"$cfgs"\' \
    +PROBE.rng_probe=1 \
    +PROBE.shrink="$REPO/inputs/weights/shrink_idm.json" \
    +PROBE.out="$out/${name}.json" || return 1
}

post_collect() {
  local out="$1"
  [ -n "${DRY_RUN:-}" ] && { echo "ANALYSIS_ONLY=$out"; return 0; }
  echo
  python3 "$REPO/probes/report_armorder.py" "$out" || return 1
}

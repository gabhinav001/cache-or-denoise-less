# Does skipping the block actually work, and what does it buy?
#
# Every closed-loop arm in this study computes each block and then substitutes
# the cached residual for its output. That is deliberate -- it makes the plain
# and attenuated arms differ in exactly one scalar multiply, which is what makes
# the quality comparison clean -- but it also means no arm has ever been timed,
# and the reported compute savings are a schedule-based count rather than a
# measurement. A cache nobody timed is not a deployment result.
#
# Two runs settle it, in this order, because the second is meaningless if the
# first fails:
#
#   equiv   full, tau=4 substituted and skipped, plain and attenuated.
#           Skipping must give BIT-IDENTICAL episode outcomes and step counts to
#           substituting. If it does not, calling fn() has a side effect the
#           cache was silently depending on, and every quality number in this
#           study describes a schedule no deployment runs. That would be the
#           finding, and it is why this arm exists.
#
#   curve   full against tau=2/4/8, all genuinely skipping and attenuated.
#           Latency and success from the same episodes, so the two never have to
#           be collected separately and assumed comparable.
#
# Timing is CUDA events around the denoise call only, not the rollout: the
# rollout is dominated by simulator stepping, which no cache changes. The first
# episode of each arm pays torch.compile and is excluded as warmup.
#
# The baseline here is COMPILED -- the probe leaves torch.compile on and only
# disables CUDA-graph capture, which the cache needs. That is the comparison
# that matters: probes/step_sweep.py already showed compilation alone is worth
# more than the cache, so a speedup measured against an eager baseline would
# mostly be measuring the compiler.
# SEEDING REGIME. These artefacts were collected while fw_libero.py reseeded only
# torch per (task, episode). rngdrift subsequently measured that this left numpy
# and python drifting -- they matched on 20/60 and 0/60 episodes across runs with
# different arm lists, against 60/60 for both torch streams -- and the harness now
# reseeds all three. Experiments collected after that change therefore sit on a
# different seeding regime from these.
#
# It does not confound anything here: every arm in a given run shares one arm list
# and one process, so the comparison within equiv.json and within curve.json is
# unaffected. It does mean these artefacts will not reproduce episode-for-episode
# under the current harness, and a re-run should be read as a fresh measurement of
# the same quantity rather than a replication.
TITLE="Skip-equivalence and measured latency"
SECTION="09"
GATE=yes
ANALYSIS=none
CKPT=libero_optional_idm_2cam224.pt

# name:configs
RUNS="${RUNS:-\
equiv:full,c3ache:4,c3ache:4:x,c3ache:4:s,c3ache:4:sx \
curve:full,c3ache:2:sx,c3ache:4:sx,c3ache:8:sx}"
DEFAULT_SEEDS="$RUNS"

run_one() {   # <name:configs> <outdir>
  local spec="$1" out="$2" name cfgs
  name=${spec%%:*}; cfgs=${spec#*:}
  echo "  -- $name ($cfgs)" >&2
  cd "$WAM_ROOT" && python probes/fw_libero.py \
    ckpt="$CKPT_DIR/libero_optional_idm_2cam224.pt" \
    model.redirect_common_files=false \
    EVALUATION.dataset_stats_path="$CKPT_DIR/libero_optional_idm_2cam224_dataset_stats.json" \
    seed=42 \
    +PROBE.suite=libero_goal +PROBE.n_tasks=3 +PROBE.n_episodes=10 \
    +PROBE.configs=\'"$cfgs"\' \
    +PROBE.timing=1 +PROBE.warmup_episodes=1 \
    +PROBE.shrink="$REPO/inputs/weights/shrink_idm.json" \
    +PROBE.out="$out/${name}.json" || return 1
}

post_collect() {
  local out="$1"
  [ -n "${DRY_RUN:-}" ] && { echo "ANALYSIS_ONLY=$out"; return 0; }
  echo
  python3 "$REPO/probes/report_latency.py" "$out" || return 1
}

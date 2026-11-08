# The deployable frontier: clean uncached baselines against caches, one host.
#
# WHAT WAS WRONG WITH THE OLD FRONTIER. Every arm in it, baseline included, ran
# the residual-recording wrapper. A deployment that is not building a cache
# records nothing, so those ratios were speedups against our own instrumentation.
# stockquiet.sh measured that wrapper at 10.2 ms/cycle but on a different host
# from the frontier, so the correction could only be extrapolated. This collects
# both on ONE host in ONE run, which removes the extrapolation entirely.
#
# THE UNCACHED ARMS USE `stock` -- full compute with recording OFF. That is the
# honest baseline. The CACHED arms keep recording on refresh cycles, because
# that is not instrumentation for them: it is how the cache gets built, and a
# deployed cache pays it too. So this compares a clean uncached policy against a
# cache carrying only the cost it genuinely has.
#
# SHORTER SCHEDULES, which the old frontier never tested. Once the claim becomes
# "a cache adds a point beyond step reduction", two and three steps are the
# natural neighbours of the fast operating region -- if two steps is as good as
# four, the cached four-step point stops being interesting. Not testing them let
# the comparison flatter the cache.
#
# REFRESH AND REUSE ARE SEPARATED. PROBE.cycle_times=1 records one latency per
# replanning cycle and whether that cycle was a refresh. A periodic cache has two
# populations and an episode mean hides both; a control loop misses its deadline
# on the slow cycle, not the average one. This is what makes a tail-latency
# statement possible at all.
#
# SIZED FOR TIMING. 3 tasks x 10 episodes, the shape latency.sh uses to reach
# sd 5-9 ms. Success at 100-200 episodes per arm already exists for the ten-step
# arms at two seeds (baselines.sh); this run is about latency, and its success
# column is reported only as a consistency check.
TITLE="Deployable frontier: clean baselines, one host"
SECTION="10d"
GATE=yes
ANALYSIS=none
CKPT=libero_optional_idm_2cam224.pt

# name:steps:configs
RUNS="${RUNS:-\
d2:2:stock \
d3:3:stock \
d4:4:stock,c3ache:4:sx \
d6:6:stock,c3ache:4:sx \
d8:8:stock \
d10:10:stock,c3ache:2:sx,c3ache:4:sx,c3ache:8:sx}"
DEFAULT_SEEDS="$RUNS"

# The seed repeats shipped alongside the seed-42 collection. check_repro_map
# dry-runs each declared set, so those artefacts are CLAIMED by this script
# rather than reading as "cited but nothing would regenerate it". They were
# first collected by an uncommitted driver into a separate directory, which the
# map could not see at all because it keys artefacts by basename: d2.json in
# three directories collapsed to one entry that the seed-42 run already claimed.
# Add a seed here when its artefacts are committed, not before.
PARAM_SETS="WAM_SEED=7 WAM_SEED=11"

run_one() {   # <name:steps:configs> <outdir>
  local spec="$1" out="$2" name steps cfgs SUF
  SUF=""; [ "${WAM_SEED:-42}" = 42 ] || SUF="_seed${WAM_SEED}"
  name=${spec%%:*};  spec=${spec#*:}
  steps=${spec%%:*}; cfgs=${spec#*:}
  case "$steps" in ''|*[!0-9]*) echo "deployfrontier: steps must be numeric" >&2; return 2 ;; esac

  echo "  -- $name ($steps denoising steps, $cfgs)" >&2
  cd "$WAM_ROOT" && python probes/fw_libero.py \
    ckpt="$CKPT_DIR/libero_optional_idm_2cam224.pt" \
    model.redirect_common_files=false \
    EVALUATION.dataset_stats_path="$CKPT_DIR/libero_optional_idm_2cam224_dataset_stats.json" \
    EVALUATION.num_inference_steps="$steps" \
    seed="${WAM_SEED:-42}" \
    +PROBE.suite=libero_goal +PROBE.n_tasks=3 +PROBE.n_episodes=10 \
    +PROBE.configs=\'"$cfgs"\' \
    +PROBE.timing=1 +PROBE.warmup_episodes=1 +PROBE.cycle_times=1 \
    +PROBE.shrink="$REPO/inputs/weights/shrink_idm.json" \
    +PROBE.out="$out/${name}${SUF}.json" || return 1
}

post_collect() {
  local out="$1"
  [ -n "${DRY_RUN:-}" ] && { echo "ANALYSIS_ONLY=$out"; return 0; }
  echo
  python3 "$REPO/probes/report_deployfrontier.py" "$out" || return 1
}

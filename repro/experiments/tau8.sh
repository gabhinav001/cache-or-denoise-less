# The tau=8 headline, re-collected as an actual replication.
#
# WHAT WAS WRONG WITH THE ORIGINAL NINE RUNS
#
# They were reported as "nine runs, all positive, sign test p=0.004", which reads
# as nine replications. They are not. They mix four LIBERO suites at 10, 20 and
# 30 episodes per arm, three arms or five, and FOUR OF THE NINE RECORD NO SEED --
# the field did not exist when they were written, so the reproduction script runs
# those at 42 and gets a valid measurement of the same quantity at a different
# draw. A reviewer cannot reproduce them and the uncertainty calculation does not
# match the claim.
#
# THE FIX IS TO SEPARATE THE TWO AXES THAT WERE POOLED
#
#   seeds   ONE configuration -- libero_goal, 10 tasks, 10 episodes -- at five
#           seeds. This is the replication, and the only part a seed-level
#           interval or sign test may be computed over.
#   suites  coverage across LIBERO suites, which is a different claim and stays
#           where it was -- the original runs in results/closedloop/. It says the
#           effect appears wherever it was looked for and says nothing about seed
#           variance, so it must never enter a seed-level interval.
#
# report_tau8.py computes the interval over the seed axis ONLY and reports the
# suite axis beside it as a separate claim. Pooling the two is the error being
# corrected; the analysis refuses to do it rather than trusting a reader not to.
#
# WHAT IS RECORDED THAT WAS NOT
#
# Every run here carries its seed. Beyond that, each artefact now records the
# checkpoint's hash (a run against a different checkpoint of the same name is a
# different experiment), the denoiser's initial latent hash, and every episode's
# initial-state hash. Two runs that disagree can now be told apart from two runs
# that were never given the same starting point -- the ambiguity that made the
# arm-order defect take three attempts to explain.
TITLE="tau=8 re-collected as a five-seed replication"
SECTION="01b"
GATE=yes
ANALYSIS=none
CKPT=libero_optional_idm_2cam224.pt

ARMS="full,c3ache:8,c3ache:8:s"

# name:suite:seed   -- every seed explicit; no "?" is permitted here
#
# THE SEED AXIS ONLY. This began with three suite runs appended, and they are
# removed rather than left undone. Two reasons. They would have written
# suite_object.json and suite_spatial.json, which are the names of artefacts the
# original collection already shipped in results/closedloop/ -- two experiments
# writing one filename, which is how a directory comes to pool the same run
# twice. To isolate the seed variation without confounding, five seeds of one
# configuration are run, while suite coverage is reported separately.
#
# Suite coverage remains what it was: the original runs in results/closedloop/,
# reported separately and never pooled into a seed-level interval.
RUNS="${RUNS:-\
seed3:libero_goal:3 \
seed7:libero_goal:7 \
seed11:libero_goal:11 \
seed23:libero_goal:23 \
seed42:libero_goal:42}"
DEFAULT_SEEDS="$RUNS"

run_one() {   # <name:suite:seed> <outdir>
  local spec="$1" out="$2" name suite seed
  name=${spec%%:*};  spec=${spec#*:}
  suite=${spec%%:*}; seed=${spec#*:}
  case "$seed" in
    ''|*[!0-9]*) echo "  !! $name has a non-numeric seed ($seed); refusing" >&2; return 1 ;;
  esac
  echo "  -- $name ($suite, seed $seed, 10 tasks x 10 episodes, arms: $ARMS)" >&2
  cd "$WAM_ROOT" && python probes/fw_libero.py \
    ckpt="$CKPT_DIR/libero_optional_idm_2cam224.pt" \
    model.redirect_common_files=false \
    EVALUATION.dataset_stats_path="$CKPT_DIR/libero_optional_idm_2cam224_dataset_stats.json" \
    seed="$seed" \
    +PROBE.suite="$suite" +PROBE.n_tasks=10 +PROBE.n_episodes=10 \
    +PROBE.configs=\'"$ARMS"\' \
    +PROBE.shrink="$REPO/inputs/weights/shrink_idm.json" \
    +PROBE.out="$out/${name}.json" || return 1
}

post_collect() {
  local out="$1"
  [ -n "${DRY_RUN:-}" ] && { echo "ANALYSIS_ONLY=$out"; return 0; }
  echo
  python3 "$REPO/probes/report_tau8.py" "$out" || return 1
}

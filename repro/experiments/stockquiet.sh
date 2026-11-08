# The stock-policy baseline, on a quiet host. Third attempt, and the reason why.
#
# Every speedup this repository reports is a ratio against the `full` arm, which
# runs the recording wrapper: it subtracts and clones a residual for every block
# on every cycle because that is how a cache gets built. A deployment records
# nothing. So the published ratios are speedups against an INSTRUMENTED harness,
# not against the policy anyone would ship, and the size of that gap has never
# been established.
#
# WHY THE FIRST TWO ATTEMPTS FAILED, since the failures are the design here:
#
#   stocktime.sh    collected stock against full, but on a box whose load average
#                   reached 63 on 36 cores. Paired se was ~48 ms against an effect
#                   of tens of ms. Unusable, and the sign followed the arm order.
#   frontier_timing collected on a quiet box (load 3-8) but timed `full`, not
#                   `stock` -- so the clean numbers are the instrumented ones.
#
# Neither run had both properties at once. This one does: quiet host, stock arm.
#
# ARM ORDER IS A CONTROL, NOT A FOOTNOTE. The first arm in a run absorbs process
# warmup, and that is exactly the stock-versus-full comparison, so running one
# order lets the design pick the answer. Both orders run; the analysis averages
# them to cancel the position term and reports the sign disagreement if there is
# one.
#
# A NEGATIVE RESULT IS NOT IMPOSSIBLE HERE, and an earlier version of the
# analysis wrongly said it was. Deleting a detached clone cannot slow a forward
# pass at source level, but removing a tensor's only consumer changes fusion
# boundaries, liveness and kernel scheduling under torch.compile, any of which
# can cost more than the clone saved. If stock comes back slower, that is a
# result about compilation, not proof of noise.
#
# WHAT IT LICENSES. If recording is measurable, every ratio in the frontier table
# can be restated against a stock baseline and the headline stops being "1.75x
# against our own instrumentation". If it is not measurable even here, that is
# the honest ceiling on what this hardware can say, and the ratios stay labelled
# as instrumented.
TITLE="Stock-policy timing on a quiet host"
SECTION="09c"
GATE=yes
ANALYSIS=none
CKPT=libero_optional_idm_2cam224.pt

# name:steps:configs   -- both orders, and a cache arm to re-express the ratio
RUNS="${RUNS:-\
q_sfirst:10:stock,full,c3ache:8:sx \
q_ffirst:10:full,stock,c3ache:8:sx}"
DEFAULT_SEEDS="$RUNS"

run_one() {   # <name:steps:configs> <outdir>
  local spec="$1" out="$2" name steps cfgs
  name=${spec%%:*};  spec=${spec#*:}
  steps=${spec%%:*}; cfgs=${spec#*:}

  echo "  -- $name ($steps denoising steps, $cfgs)" >&2
  cd "$WAM_ROOT" && python probes/fw_libero.py \
    ckpt="$CKPT_DIR/libero_optional_idm_2cam224.pt" \
    model.redirect_common_files=false \
    EVALUATION.dataset_stats_path="$CKPT_DIR/libero_optional_idm_2cam224_dataset_stats.json" \
    EVALUATION.num_inference_steps="$steps" \
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
  python3 "$REPO/probes/report_stocktime.py" "$out" || return 1
}

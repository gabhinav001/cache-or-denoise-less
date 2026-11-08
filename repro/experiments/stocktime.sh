# An honest denominator for every speedup in this repository.
#
# Each closed-loop arm here runs under a wrapper that can record the residual of
# every block. The FULL-COMPUTE arm records on every cycle -- 300 subtract-and-
# clone operations per replanning cycle, one per (denoise step, block) -- because
# that is how the cache gets built. A cached arm records only on refresh cycles,
# so at tau=8 it pays that cost once in eight.
#
# A deployment that is not building a cache records nothing at all.
#
# So the baseline every speedup on this page is quoted against carries
# instrumentation the arms it is compared with carry less of, and the ratios are
# inflated by an unknown amount. That includes the headline: "4 denoising steps
# is 1.80x" is 1.80x against an instrumented ten-step baseline.
#
# THE ARM. `stock` is full compute with recording switched off -- the same
# forward pass, the same schedule, nothing retained. It is quality-identical to
# `full` by construction: recording only reads a block's output and stores a
# detached clone, so removing it cannot change an action. That is checkable
# rather than assumed, and the analysis checks it: if `stock` and `full` disagree
# on any episode, the arm is wrong and the timing is meaningless.
#
# WHAT THIS MEASURES. One run, all arms timed on the same episodes:
#
#   stock                full compute, no recording        the honest baseline
#   full                 full compute, recording           what we have been using
#   c3ache:2/4/8:sx      attenuated, genuinely skipping    the cache arms
#
# Then every speedup in the study can be recomputed against `stock` instead of
# `full`, and the difference between the two denominators is the size of the
# error. If stock and full are within timing noise, the instrumentation was free
# and the published ratios stand. If stock is materially faster, every ratio in
# this repository shrinks and the 4-step result shrinks with them -- it is a
# full-compute arm too, so it is inflated by the same mechanism, though by less
# since it runs 4 steps x 30 blocks rather than 10 x 30.
#
# SIZED FOR TIMING, NOT SUCCESS. 3 tasks x 10 episodes, which is exactly the
# shape latency.sh used to reach sd 5-9 ms on 27 timed cycles. The first attempt
# here inverted it to 10 tasks x 3 episodes and the sd came back at 126-217 ms,
# fifteen times worse and useless for a 5% effect: warmup_episodes drops one
# episode per run, so spreading 3 episodes over 10 tasks pays a settle cost on
# almost every episode instead of amortising it over ten. Concentrate the
# episodes. The success columns exist only to verify stock == full.
#
# ARM ORDER IS A CONFOUND HERE, so it is a control rather than a footnote. The
# first arm in a run absorbs process warmup, and this repository already
# documents (A8) that an arm's position changes its numbers. Whichever of stock
# and full runs first is handicapped, and that is precisely the comparison being
# made -- so both orders are run. If the sign of the difference follows the
# order, position dominates and the overhead is not measurable this way; that is
# a real answer and the analysis reports it rather than picking the flattering
# run. Only if both orders agree is the overhead established.
TITLE="An un-instrumented baseline for every speedup"
SECTION="09b"
GATE=yes
ANALYSIS=none
CKPT=libero_optional_idm_2cam224.pt

# name:steps:configs
RUNS="${RUNS:-\
sfirst10:10:stock,full,c3ache:8:sx \
ffirst10:10:full,stock,c3ache:8:sx \
sfirst4:4:stock,full \
ffirst4:4:full,stock}"
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

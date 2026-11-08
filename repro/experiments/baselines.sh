# Does attenuated caching beat the obvious alternatives at equal wall-clock?
#
# Every comparison in this repository is between cache settings. None of them
# asks the question a practitioner actually asks, which is: given a latency
# budget, what is the best thing to do with it? A cache that raises success at a
# fixed refresh interval has not established anything if simply running fewer
# denoising steps reaches the same latency at higher success.
#
# Two baselines, chosen because they are the ones that could plausibly win and
# both are nearly free to test:
#
#   fewer steps   full compute at 4, 6 and 8 denoising steps instead of 10. No
#                 cache at all. This is the strongest competitor: it is trivial
#                 to deploy, has no calibration, and trades exactly the same
#                 thing the cache trades.
#   global beta   tau=4 with ONE scalar weight shared by every cell, rather than
#                 the per-cell fit. The per-cell weights are worth +7.7% on
#                 held-out residual error against +3.1% for a single global
#                 scalar, so if the closed-loop gain survives collapsing them,
#                 the per-cell fit -- and its calibration pass -- is not what is
#                 doing the work.
#
# Both report latency and success from the same episodes, so the comparison is
# against measured wall-clock rather than a modelled block count.
#
# SIZED TO THE QUESTION. The first draft ran 3 tasks x 10 episodes, matching the
# latency experiment. Exercised against synthetic data with a known answer, that
# turned out to settle nothing: at 30 episodes the 95% interval on a success rate
# near 93% is +-9.9 points, so an arm that is genuinely faster AND better is not
# distinguishable from one that is neither. The frontier came back with every arm
# on it, which is not a result -- it is a sample size.
#
#   episodes/arm    95% half-width at p=0.93
#            30                        9.9
#           100                        5.2
#           200                        3.6
#           400                        2.5
#
# 200 per arm (10 tasks x 20 episodes) resolves a 10-point gap cleanly and a
# 5-point gap marginally, which is the range that decides this question. Ranking
# finer than that would need 400+ and is not worth its cost here; the analysis
# says so rather than implying a precision it does not have.
#
# The cache arms are re-run HERE at the same episode count rather than borrowed
# from latency.sh, which used 30. Comparing a 200-episode baseline against a
# 30-episode cache arm would put most of the frontier's uncertainty in one half
# of it.
#
# NOT TESTED HERE: an adaptive cache that refreshes on a content criterion
# instead of a fixed schedule. An earlier version of this comment called every
# cheap version a straw man, on the grounds that an informative trigger needs the
# block output -- the thing being skipped. That is false. X-Cache gates on a
# subsampled fingerprint of the block's INPUT, computed before the block runs and
# therefore available exactly when the decision has to be made. An adaptive cache
# is a legitimate comparator we have not implemented, not one ruled out by
# construction, and the limitations say so.
# THE SEED IS A PARAMETER. Everything below was collected once at seed 42, and a
# single seed cannot support a frontier: seed-to-seed movement in this study
# reaches 10 points, which is larger than most of the gaps this table ranks. Set
# WAM_SEED to collect the same comparison at another seed; artefacts are named
# with it so two seeds never overwrite each other.
SEED="${WAM_SEED:-42}"

# Parameter sets this experiment has SHIPPED artefacts for. check_repro_map.py
# dry-runs each, so the seed-7 collection is claimed by this script rather than
# reading as "cited but nothing would regenerate it". Add a seed here when its
# artefacts are committed, not before.
PARAM_SETS="WAM_SEED=7"

TITLE="Simple baselines at matched latency"
SECTION="10"
GATE=yes
ANALYSIS=none
CKPT=libero_optional_idm_2cam224.pt

# name:steps:configs
RUNS="${RUNS:-\
steps4:4:full \
steps6:6:full \
steps8:8:full \
cache:10:full,c3ache:2:sx,c3ache:4:sx,c3ache:8:sx \
globalbeta:10:full,c3ache:4:sx}"
DEFAULT_SEEDS="$RUNS"

run_one() {   # <name:steps:configs> <outdir>
  local spec="$1" out="$2" name steps cfgs shrink SUF
  SUF=""; [ "$SEED" = 42 ] || SUF="_seed$SEED"
  name=${spec%%:*};  spec=${spec#*:}
  steps=${spec%%:*}; cfgs=${spec#*:}

  # The global-beta arm needs a shrink file whose every coefficient is the same
  # scalar. That scalar is the median of the 300 per-cell weights themselves
  # (0.7165 on Fast-WAM; strictly vals[n//2], the upper of the two middle values
  # at even n), so the global arm is the best single weight derived from the same
  # fit rather than an arbitrary one -- the baseline should be the strongest
  # version of itself. Both arms therefore use calibration, so this comparison
  # tests per-cell against global, NOT calibrated against uncalibrated.
  shrink="$REPO/inputs/weights/shrink_idm.json"
  if [ "$name" = globalbeta ]; then
    shrink="$out/_shrink_global.json"
    python3 - "$REPO/inputs/weights/shrink_idm.json" "$shrink" <<'PYFLAT'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(src))
vals = sorted(v[0] for v in d["coef"].values())
b = vals[len(vals) // 2]
d["coef"] = {k: [b] for k in d["coef"]}
d["note"] = f"one global weight {b:.4f} (median of the per-cell fit)"
json.dump(d, open(dst, "w"))
print(f"  -- global weight {b:.4f} over {len(vals)} cells", flush=True)
PYFLAT
  fi

  echo "  -- $name ($steps denoising steps, $cfgs)" >&2
  cd "$WAM_ROOT" && python probes/fw_libero.py \
    ckpt="$CKPT_DIR/libero_optional_idm_2cam224.pt" \
    model.redirect_common_files=false \
    EVALUATION.dataset_stats_path="$CKPT_DIR/libero_optional_idm_2cam224_dataset_stats.json" \
    EVALUATION.num_inference_steps="$steps" \
    seed="$SEED" \
    +PROBE.suite=libero_goal +PROBE.n_tasks=10 +PROBE.n_episodes=20 \
    +PROBE.configs=\'"$cfgs"\' \
    +PROBE.timing=1 +PROBE.warmup_episodes=1 \
    +PROBE.shrink="$shrink" \
    +PROBE.out="$out/${name}${SUF}.json" || return 1
}

post_collect() {
  local out="$1"
  [ -n "${DRY_RUN:-}" ] && { echo "ANALYSIS_ONLY=$out"; return 0; }
  echo
  # The cache arms come from the latency experiment. Its artefacts may be in the
  # committed results/ or still in a driver's output directory, whose name
  # carries a run label -- so hand the analysis every candidate and let it use
  # what exists rather than assuming one layout.
  # Self-contained: the cache arms are in this directory at the same episode
  # count. The latency directories are still passed so a reviewer who has them
  # sees those arms too, clearly labelled by their source file.
  python3 "$REPO/probes/report_baselines.py" "$out" \
      "$REPO/results/latency" "$REPO/repro/out"/latency* || return 1
}

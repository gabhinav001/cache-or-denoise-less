# Pre-registered confirmation: does attenuation transfer to pi0.5 after all?
#
# THE SITUATION THIS RESOLVES
#
# The closed-loop seeds disagree in sign (-0.3 at one, +9.0 at another), so the
# transfer claim is established in neither direction. Separately, a five-point
# weight sweep -- paired on (task, episode) against the published beta=1.00 --
# put beta=0.25 at +8.5 points, 22 episodes better against 5 worse, p=0.0015, and
# p=0.0076 after correcting for the five weights tested. That sweep was
# EXPLORATORY: the weight was chosen by looking at the same data that scored it,
# and the runs were 200 paired episodes at one seed apiece.
#
# An exploratory result selected post hoc cannot confirm itself. This experiment
# exists to test it once, properly, on data it has never seen.
#
# THE PRE-REGISTRATION, WRITTEN BEFORE THE RUN
#
#   Hypothesis      Substituting the cached residual at beta=0.25 yields higher
#                   LIBERO-goal success than the published beta=1.00, on pi0.5,
#                   at tau=2.
#   Design          3 seeds (7, 23, 42) x 10 tasks x 15 episodes = 150 paired
#                   episodes per arm per seed, 450 pooled -- against the 200 the
#                   exploratory sweep used to find +8.5. Amended from 30 episodes
#                   on cost grounds; see the amendment note below. Both weights are ARMS OF ONE FILE,
#                   so they pair on (task, episode) within the run against a
#                   single full-compute control. The first draft of this ran them
#                   as two files and compared across; that needs the two full
#                   arms to agree, which is an extra assumption, an extra way for
#                   a good seed to be discarded, and it pays for full compute
#                   twice.
#   Primary test    Mean per-seed difference in success (beta=0.25 minus
#                   beta=1.00), with a Student-t interval over the three seeds.
#   Decision rule   CONFIRMED only if the effect is positive on all three seeds
#                   AND the pooled McNemar over the 900 paired episodes gives
#                   p < 0.05. Anything else is reported as NOT confirmed. The
#                   rule is fixed here so it cannot be adjusted once the numbers
#                   are in, which is how the sweep came to be over-read.
#   Control         Each file runs full compute once, as the reference the two
#                   weights are read against. It is not part of the test.
#
# WHAT A NULL WOULD MEAN
#
# Not that attenuation fails on pi0.5 -- three seeds cannot establish that
# either. It would mean the sweep's +8.5 does not survive contact with fresh
# data, which is the single most likely explanation for it and the reason this
# run is worth its cost.
# AMENDMENT RULE, recorded 2026-09-14 while seed 7 task 0 was still running and
# BEFORE any beta=0.25 outcome existed in the log (only `full` and `b100` had
# reported). Stating it in advance is what keeps it from being an outcome-
# dependent choice.
#
# Observed: task 0 cost ~22 min per arm, all three arms paying compile. If tasks
# 1-9 reuse the compiled graphs they should fall to ~6 min per arm and the whole
# experiment costs ~12 GPU-hours. If they do not, it is ~33 hours, which this
# budget cannot carry alongside the other experiments.
#
#   if task 1 comes in at <= 10 min/arm   keep 10 tasks x 30 episodes
#   if task 1 comes in at  > 10 min/arm   halve the episodes per arm for ALL
#                                         THREE seeds and restart, so the design
#                                         stays uniform
#
# TRIGGERED, and settled without waiting for task 1. The rule named task 1 on the
# assumption that task 0 was contaminated by compilation. It is not: by arm 3 the
# autotune count had stopped moving at 46 while the GPU kept bursting and CPU kept
# accumulating, so that arm's 25 minutes is pure stepping. That is exactly the
# compile-free steady-state number the rule was waiting for, an hour earlier and
# a dollar cheaper. (The cached arms are slower than full compute because they
# fail more often, and a failed episode runs to max_steps.)
#
# Measured ~55 min of stepping per task across the three arms: 10 tasks x 30
# episodes is 9.6 h per seed, 28.8 h and $12.62 for the experiment. Halving to
# 150 episodes per arm per seed brings that to 15 h and $6.58.
#
# The recorded wording said "5 tasks x 30 episodes". This uses 10 tasks x 15
# instead: identical episode count, identical cost, twice the task coverage, and
# task diversity is what a transfer claim needs. Deviating from one's own
# amendment deserves saying out loud -- it is defensible here only because the
# change is strictly better at the same size and is unrelated to any outcome.
# What has been seen at the time of writing: seed 7 task 0 for `full` (SR=1.00)
# and for beta=1.00 (SR=0.87). NO beta=0.25 result exists yet, which is the arm
# the hypothesis is about.
#
# 5 tasks gives 150 paired episodes per arm per seed and 450 pooled, against the
# 200 the exploratory sweep used to find +8.5. That is ample for the pooled
# McNemar in the decision rule and for the per-seed sign, so the reduction costs
# power this test does not need. The DECISION RULE ITSELF is unchanged.
TITLE="pi0.5: pre-registered confirmation at beta=0.25"
SECTION="07b"
GATE=no
STACK=lerobot
MUJOCO_PIN=
ANALYSIS=none

RUNS="${RUNS:-7 23 42}"
DEFAULT_SEEDS="$RUNS"

run_one() {   # <seed> <outdir>
  local seed="$1" out="$2" b100 b025
  # pi05_libero.py applies one coefficient per cell from a shrink file, with no
  # flat-weight option, so a constant weight is a file whose every coefficient is
  # that constant. That is the route the shipped sweep used, so this confirms the
  # same quantity it measured. PI_SHRINK takes label=path pairs, which is what
  # lets both weights be arms of one run.
  b100="$out/_shrink_b100.json"; b025="$out/_shrink_b025.json"
  python3 - "$REPO/inputs/weights/shrink_pi05.json" "$b100" 1.00 <<'PYFLAT'
import json, sys
src, dst, b = sys.argv[1], sys.argv[2], float(sys.argv[3])
d = json.load(open(src))
d["coef"] = {k: [b] for k in d["coef"]}
d["note"] = f"flat weight {b}, pre-registered confirmation"
json.dump(d, open(dst, "w"))
PYFLAT
  python3 - "$REPO/inputs/weights/shrink_pi05.json" "$b025" 0.25 <<'PYFLAT'
import json, sys
src, dst, b = sys.argv[1], sys.argv[2], float(sys.argv[3])
d = json.load(open(src))
d["coef"] = {k: [b] for k in d["coef"]}
d["note"] = f"flat weight {b}, pre-registered confirmation"
json.dump(d, open(dst, "w"))
PYFLAT
  echo "  -- seed $seed: full, beta=1.00, beta=0.25 -- 150 episodes per arm" >&2
  cd "$WAM_ROOT" && \
    PI_SUITE=libero_goal PI_TASKS=10 PI_EPISODES=15 PI_SEED="$seed" \
    PI_EXEC=10 PI_SKIP_STEPS=8 \
    PI_CONFIGS="full,c3ache:2:s=b100,c3ache:2:s=b025" \
    PI_SHRINK="b100=$b100,b025=$b025" \
    PI_OUT="$out/confirm_seed${seed}.json" python probes/pi05_libero.py || return 1
}

post_collect() {
  local out="$1"
  [ -n "${DRY_RUN:-}" ] && { echo "ANALYSIS_ONLY=$out"; return 0; }
  echo
  python3 "$REPO/probes/report_pi05_confirm.py" "$out" || return 1
}

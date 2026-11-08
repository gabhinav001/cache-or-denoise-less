# The arms the 500-state collection was missing, on the same 500 states.
#
# WHY. The state-scale collection compared
# schedules but carried no full-compute reference and no plain (beta=1) cache, so
# it could not show what attenuation CONTRIBUTES at that scale -- only how a
# cached arm compares to a shorter uncached one. And the global-weight ablation
# used the MEDIAN of the fitted per-cell weights where the least-squares
# objective defines a POOLED scalar; the two differ, and the median is the weaker
# of them, so that comparison could not establish per-cell superiority over a
# correctly fitted single scalar.
#
# Three arms close both, all at ten denoising steps on the same 500 LIBERO Goal
# initial states the state-scale run used:
#   full        full compute, recording off -- the reference that was missing
#   plain       beta=1 cache at tau=4 -- what attenuation is measured against
#   globalfit   one scalar for every cell, the POOLED least-squares value
#   percell     the shipped 300-weight fit
#
# THE PER-CELL ARM IS RE-RUN HERE RATHER THAN REUSED. It already exists in
# results/statescale/s10.json, but that collection is on a different host, and
# every comparison this matrix exists to make involves it: per-cell against the
# fitted global is the ablation, per-cell against plain is attenuation's
# contribution. Reusing it would put a host boundary through the middle of both.
# Runs of one seed on different hardware agree on most outcomes and not all, and
# the effects here are a few points, so the boundary is not affordable. Re-running
# costs one arm and makes every comparison within-collection.
#
# THE GLOBAL SCALAR IS DERIVED, NOT TYPED. Equation (1) is
# sum_i <T_i,C_i> / sum_i ||C_i||^2 over the calibration observations, which is
# recomputed here from the shipped Gram tensors. Hardcoding 0.7384 would let the
# arm drift silently from the objective it claims to implement.
#
# SUCCESS IS THE QUESTION; LATENCY IS NOT. These arms run on a different host
# from the earlier state-scale arms, so their timings must not be pooled with
# them. Success is paired on the initial state and is unaffected by the host.
TITLE="Completing the 500-state matrix: reference, plain cache, fitted global"
SECTION="10f"
GATE=yes
ANALYSIS=none
CKPT=libero_optional_idm_2cam224.pt

EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
N_TASKS="${N_TASKS:-10}"

# name:steps:configs
RUNS="${RUNS:-\
m10full:10:stock \
m10plain:10:c3ache:4:x \
m10global:10:c3ache:4:sx \
m10percell:10:c3ache:4:sx}"
DEFAULT_SEEDS="$RUNS"

pooled_beta_file() {   # <dst> -> writes a shrink file whose every coef is the pooled LS scalar
  local dst="$1"
  python3 - "$REPO" "$dst" <<'PY'
import json, os, sys
sys.path.insert(0, os.path.join(sys.argv[1], "probes"))
import numpy as np, report_families as RF
repo, dst = sys.argv[1], sys.argv[2]
# the deployment checkpoint's own calibration tensors -- the same ones
# inputs/weights/shrink_idm.json was fitted from
z, G, cell, ep = RF.load(os.path.join(repo, "results", "residual", "fw_families_idm.npz"))
# SPLIT MISMATCH, MADE EXPLICIT RATHER THAN SILENTLY FIXED. shrink_idm.json is
# fitted on the training episodes only -- every fourth held out -- and this
# scalar was fitted on all 24. The ablation therefore compared 300 weights fitted
# on 18 episodes against 1 weight fitted on 24, so part of any difference was
# calibration budget rather than granularity. Training-only moves the scalar
# 0.738362 -> 0.738959.
#
# The shipped results/matrix/m10global*.json were collected at 0.738362, and the
# write-ups quote 0.7384 as a description of THOSE runs. Changing the default
# here would leave a committed artefact that this script no longer regenerates,
# which is the one defect this repository is least entitled to introduce. So the
# split is a parameter: FIT_SPLIT=all reproduces what shipped, FIT_SPLIT=train
# is the matched fit any NEW comparison should use.
split = os.environ.get("FIT_SPLIT", "all")
if split not in ("all", "train"):
    raise SystemExit(f"FIT_SPLIT must be 'all' or 'train', not {split!r}")
ok = G[:, 0, 0] > 0
if split == "train":
    ok = ok & ((ep % RF.TEST_EVERY) != 0)
G = G[ok]
ct, cc = G[:, 0, 1], G[:, 1, 1]
beta = float(ct.sum() / cc.sum())          # Equation (1), pooled over TRAIN observations
src = json.load(open(os.path.join(repo, "inputs", "weights", "shrink_idm.json")))
src["coef"] = {k: [beta] for k in src["coef"]}
src["note"] = (f"one global weight {beta:.6f}, the POOLED least-squares scalar of "
               f"Eq (1), fitted on {'the TRAINING episodes only' if split == 'train' else 'ALL episodes'}"
               f" (every {RF.TEST_EVERY}th is held out from the per-cell fit)")
src["fit_split"] = split
json.dump(src, open(dst, "w"))
print(f"    pooled least-squares beta = {beta:.6f}  [FIT_SPLIT={split}] "
      f"(median of per-cell fits is 0.7165)")
if split == "all":
    print("    NOTE: the per-cell weights this is compared against are fitted on "
          "the\n          training episodes only. FIT_SPLIT=train matches them.")
PY
}

run_one() {   # <name:steps:configs> <outdir>
  local spec="$1" out="$2" name steps cfgs shrink SUF
  SUF=""; [ "${WAM_SEED:-42}" = 42 ] || SUF="_seed${WAM_SEED}"
  name=${spec%%:*};  spec=${spec#*:}
  steps=${spec%%:*}; cfgs=${spec#*:}
  case "$steps" in ''|*[!0-9]*) echo "matrix: steps must be numeric" >&2; return 2 ;; esac

  shrink="$REPO/inputs/weights/shrink_idm.json"
  if [ "$name" = m10global ]; then  # percell keeps the shipped 300-weight file
    shrink="$out/_shrink_pooled.json"
    pooled_beta_file "$shrink" || return 1
  fi

  echo "  -- $name ($steps steps, $cfgs, ${N_TASKS}x${EPISODES_PER_TASK} states)" >&2
  cd "$WAM_ROOT" && python probes/fw_libero.py \
    ckpt="$CKPT_DIR/libero_optional_idm_2cam224.pt" \
    model.redirect_common_files=false \
    EVALUATION.dataset_stats_path="$CKPT_DIR/libero_optional_idm_2cam224_dataset_stats.json" \
    EVALUATION.num_inference_steps="$steps" \
    seed="${WAM_SEED:-42}" \
    +PROBE.suite=libero_goal +PROBE.n_tasks="$N_TASKS" \
    +PROBE.n_episodes="$EPISODES_PER_TASK" \
    +PROBE.configs=\'"$cfgs"\' \
    +PROBE.timing=1 +PROBE.warmup_episodes=1 \
    +PROBE.shrink="$shrink" \
    +PROBE.out="$out/${name}${SUF}.json" || return 1
}

post_collect() {
  local out="$1"
  [ -n "${DRY_RUN:-}" ] && { echo "ANALYSIS_ONLY=$out"; return 0; }
  echo
  python3 "$REPO/probes/report_matrix.py" "$out" || return 1
}

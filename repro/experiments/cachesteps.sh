# Does caching add anything once you have already cut denoising steps?
#
# This repository's central negative result is that four denoising steps beats
# every cache setting tested, at lower latency. The obvious rejoinder is: why not
# both? Testing a cache ON TOP of a reduced-step schedule provides the missing
# comparator. If caching still helps at 4 or 6
# steps, the negative result is about a badly chosen operating point rather than
# about caching. If it does not, the result is about caching.
#
# TWO DESIGN DECISIONS, BOTH LOAD-BEARING, BOTH STATED RATHER THAN BURIED.
#
# 1. WHICH CELLS A SHORTER SCHEDULE CACHES. The published rule is "every block,
#    all but the last two denoise steps" -- [0,7] of 10, leaving the final
#    refinement exact. probes/fw_libero.py now generalises that by preserving the
#    STRUCTURE rather than the count: [0,3] of 6, [0,1] of 4. A short schedule
#    therefore caches a smaller fraction of its work (80% of steps at 10, 67% at
#    6, 50% at 4), so these caches are less aggressive by construction. Holding
#    the fraction fixed instead would cache into the refinement steps the method
#    deliberately leaves alone, which would be a different method.
#
# 2. THE WEIGHTS ARE MIS-SPECIFIED HERE, AND CANNOT CHEAPLY BE FIXED. The
#    per-cell weights shipped in inputs/weights/ were fit from Gram statistics
#    collected at TEN denoising steps. A 4-step schedule visits different noise
#    levels, so the least-squares weight for cell (k, b) is not the same number.
#    Refitting needs a fresh Gram collection at each step count, which is a GPU
#    run this budget does not cover. So this tests what a practitioner would
#    actually try first -- the shipped weights, applied at fewer steps -- and the
#    analysis says so. A null here is therefore weaker evidence against caching
#    than a null with refit weights would be.
#
# ARMS. At each of 4 and 6 steps: full compute, and attenuated tau=4 skipping.
# tau=4 is the middle setting and the one where attenuation is worth most per
# unit of damage at ten steps. 10 tasks x 10 episodes = 100 per arm, matching the
# tau=8 replication's protocol rather than the frontier's 200, because this is a
# direction test and not a ranking.
# REPLICATION. One seed is not enough here: the
# three-point success difference rests on 100 episodes at one seed and its
# bootstrap interval spans zero. WAM_SEED collects the same comparison at
# another seed; artefacts are named with it so seeds never overwrite.
SEED="${WAM_SEED:-42}"
# PARAM_SETS lists seeds whose artefacts are SHIPPED, so check_repro_map can
# claim them. Declaring an unshipped output makes it a phantom -- the same reason
# new experiments carry PENDING -- so this was deferred while seed 7 ran. It has
# landed: cs4_seed7.json and cs6_seed7.json are committed, and the full paper
# quotes the intervals derived from them, so the claim belongs here now.
PARAM_SETS="WAM_SEED=7"

TITLE="Caching on top of a reduced-step schedule"
SECTION="10c"
GATE=yes
ANALYSIS=none
CKPT=libero_optional_idm_2cam224.pt

# name:steps:configs
RUNS="${RUNS:-\
cs4:4:full,c3ache:4:sx \
cs6:6:full,c3ache:4:sx}"
DEFAULT_SEEDS="$RUNS"

run_one() {   # <name:steps:configs> <outdir>
  local spec="$1" out="$2" name steps cfgs SUF
  SUF=""; [ "$SEED" = 42 ] || SUF="_seed$SEED"
  name=${spec%%:*};  spec=${spec#*:}
  steps=${spec%%:*}; cfgs=${spec#*:}
  case "$steps" in ''|*[!0-9]*) echo "cachesteps: steps must be numeric" >&2; return 2 ;; esac

  echo "  -- $name ($steps denoising steps, $cfgs)" >&2
  cd "$WAM_ROOT" && python probes/fw_libero.py \
    ckpt="$CKPT_DIR/libero_optional_idm_2cam224.pt" \
    model.redirect_common_files=false \
    EVALUATION.dataset_stats_path="$CKPT_DIR/libero_optional_idm_2cam224_dataset_stats.json" \
    EVALUATION.num_inference_steps="$steps" \
    seed="$SEED" \
    +PROBE.suite=libero_goal +PROBE.n_tasks=10 +PROBE.n_episodes=10 \
    +PROBE.configs=\'"$cfgs"\' \
    +PROBE.timing=1 +PROBE.warmup_episodes=1 \
    +PROBE.shrink="$REPO/inputs/weights/shrink_idm.json" \
    +PROBE.out="$out/${name}${SUF}.json" || return 1
}

post_collect() {
  local out="$1"
  [ -n "${DRY_RUN:-}" ] && { echo "ANALYSIS_ONLY=$out"; return 0; }
  echo
  python3 "$REPO/probes/report_cachesteps.py" "$out" || return 1
}

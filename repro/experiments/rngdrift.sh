# Which random stream carried the arm-order drift?
#
# The A8 defect is measured and fixed: the same configuration reported different
# numbers depending on how many arms ran before it (79/100 agreement, while the
# arm that ran first in both runs was bit-identical). The fix reseeds the global
# torch stream per (task, episode), which removes the dependence by construction.
#
# What has never been established is WHICH consumer carried it. The first account
# blamed the denoiser's initial latent; probes/check_seed_path.py proves that is
# impossible, because with seed= set that latent comes from a generator built
# fresh from the seed inside every inference call and is one constant tensor for
# the whole run. So the carrier is unidentified, and an unidentified carrier is a
# live hazard for anyone else running multi-arm cache comparisons.
#
# This settles it. Four runs, two arm lists crossed with the reseed on and off:
#
#   armsN_reseed    3 and 5 arms, reseed ON   -- must now agree; the regression test
#   armsN_drift     3 and 5 arms, reseed OFF  -- reproduces the defect deliberately
#
# Every episode records a fingerprint of all four global streams (torch CPU,
# torch CUDA, numpy, python) taken at episode start. With the reseed on the
# fingerprints must match across arm lists for the same (task, episode). With it
# off, whichever stream stops matching is the one that was carrying the drift --
# named directly rather than inferred.
#
# Deliberately small: two tasks and ten episodes is enough for a later arm to
# diverge, and the question is which stream moves, not how much success changes.
TITLE="Which stream carried the arm-order drift (A8)"
SECTION="A8"
GATE=yes
ANALYSIS=none
CKPT=libero_optional_idm_2cam224.pt

SHORT="full,c3ache:4,c3ache:4:s"
LONG="full,c3ache:2,c3ache:2:s,c3ache:4,c3ache:4:s"

# name:configs:no_reseed
RUNS="${RUNS:-\
arms3_reseed:$SHORT:0 \
arms5_reseed:$LONG:0 \
arms3_drift:$SHORT:1 \
arms5_drift:$LONG:1}"
DEFAULT_SEEDS="$RUNS"

run_one() {   # <name:configs:no_reseed> <outdir>
  local spec="$1" out="$2" name cfgs nors
  name=${spec%%:*};  spec=${spec#*:}
  nors=${spec##*:};  cfgs=${spec%:*}
  echo "  -- $name (reseed $([ "$nors" = 1 ] && echo OFF || echo ON), arms: $cfgs)" >&2
  cd "$WAM_ROOT" && python probes/fw_libero.py \
    ckpt="$CKPT_DIR/libero_optional_idm_2cam224.pt" \
    model.redirect_common_files=false \
    EVALUATION.dataset_stats_path="$CKPT_DIR/libero_optional_idm_2cam224_dataset_stats.json" \
    seed=42 \
    +PROBE.suite=libero_goal +PROBE.n_tasks=2 +PROBE.n_episodes=10 \
    +PROBE.configs=\'"$cfgs"\' \
    +PROBE.no_reseed="$nors" +PROBE.rng_probe=1 \
    +PROBE.shrink="$REPO/inputs/weights/shrink_idm.json" \
    +PROBE.out="$out/${name}.json" || return 1
}

post_collect() {
  local out="$1"
  [ -n "${DRY_RUN:-}" ] && { echo "ANALYSIS_ONLY=$out"; return 0; }
  echo
  python3 "$REPO/probes/report_rngdrift.py" "$out" || return 1
}

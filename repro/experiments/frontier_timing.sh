# Latency for the frontier, measured once, on a quiet host.
#
# WHY THIS IS SEPARATE FROM baselines.sh. That experiment collects success and
# latency from the same episodes, which is the right design when the host is
# quiet and the wrong one when it is not. During the seed-7 collection this box
# ran at a load average of 63 on 36 CPUs and later at 1.8, and the full-compute
# arm read 5722 ms/cycle on task 0 against 360 ms on task 2 -- a 16x spread on
# identical work. Success is unaffected by that: outcomes are deterministic
# given seed, GPU and arm list, and a same-GPU re-run of this study reproduced
# all 500 episode outcomes exactly. Wall clock is not.
#
# So the two columns of the frontier are collected differently and on purpose:
#
#   success   from baselines.sh, 200 episodes per arm per seed. Load-independent.
#   latency   from HERE, 30 episodes per arm, one quiet host, one collection.
#
# LATENCY IS ONLY WEAKLY SEED-DEPENDENT, which is not the same as independent.
# An arm's ms/cycle is mostly a property of the arm, the hardware and the
# contention. But a cache arm averages expensive refresh cycles against cheap
# reuse cycles, and that mixture shifts with episode length, which the seed does
# affect. So one collection is a STANDARDISED BENCHMARK rather than the latency
# every seed would produce -- and it is still the honest option, because the
# alternative is a column whose entries were taken hours apart under thirty-fold
# different load.
#
# SHAPE. 3 tasks x 10 episodes, which is what latency.sh uses to reach sd 5-9 ms
# on 27 timed cycles. stocktime.sh tried 10 tasks x 3 episodes and got sd 126-217
# ms: warmup_episodes drops one episode per RUN, so thin coverage per task pays a
# settle cost on nearly every episode instead of amortising it over ten.
#
# CHECK THE LOAD BEFORE TRUSTING THE OUTPUT. Every artefact now records
# loadavg_1_5_15 and cpu_count in its provenance block. The analysis refuses to
# report a speedup when the run was collected above half the core count, because
# a number taken on a busy box is not a measurement of this software.
TITLE="Frontier latency, one quiet host"
SECTION="10b"
GATE=yes
ANALYSIS=none
CKPT=libero_optional_idm_2cam224.pt

# name:steps:configs
RUNS="${RUNS:-\
lat4:4:full \
lat6:6:full \
lat8:8:full \
lat10:10:full,c3ache:2:sx,c3ache:4:sx,c3ache:8:sx}"
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
  python3 "$REPO/probes/report_frontier_timing.py" "$out" || return 1
}

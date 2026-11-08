# Does caching improve success over a shorter schedule? Measured on states, not seeds.
#
# WHY THIS EXISTS. The deployment frontier compared eleven arms on 30 initial
# states, and repeating it at further seeds did not help: LIBERO draws initial
# states from the benchmark rather than from the seed, so every repeat revisits
# the SAME 30 states and varies denoiser sampling only. 87% of state-arm cells
# came back unanimous. Those repeats establish that the ordering is fragile; they
# cannot establish whether a cache is actually better or worse, because they add
# almost no independent information about success.
#
# The fix is more STATES, not more seeds. LIBERO Goal holds 10 tasks x 50 initial
# states; the frontier used 3 tasks x 10, leaving 94% of the benchmark unused.
#
# SIZED FROM THE OBSERVED DISCORDANCE. Arms disagree on about 1 state in 30, and
# a paired comparison's power is driven by that discordance rather than by the
# success rates. At 3.3% discordance, separating a 3-point paired difference at
# 80% power needs roughly 200 states and a 2-point difference roughly 570. That
# estimate rests on a single discordant state, so its own interval is wide
# ([0.1%, 17.2%]); 500 states is the most the suite offers and is read as a bound
# on the difference rather than a guaranteed resolution.
#
# FOUR ARMS, NOT ELEVEN. The question is whether a cache beats a shorter uncached
# schedule, so this runs the two cheapest uncached schedules against the two
# caches that survived any frontier. Dropping the other seven is what makes 500
# states affordable.
#
# SUCCESS ONLY. Latency is not recollected and must not be pooled with the
# frontier's: this is a different host and, at 8 GPU-hours, possibly several.
# Success is seed-dependent but not load-dependent -- a frontier arm read
# 5722 ms/cycle against 360 on identical work while its success rate was
# unaffected -- so success survives a busy or swapped host in a way latency does
# not. Uncached arms use `stock`, so recording is off where it is not needed.
TITLE="Caching against shorter schedules, at benchmark scale"
SECTION="10e"
GATE=yes
ANALYSIS=none
CKPT=libero_optional_idm_2cam224.pt

# Episodes per task. LIBERO indexes init states as inits[ep % len(inits)], so
# asking for more than the suite holds silently REPEATS states and recreates the
# correlation this experiment exists to remove. preflight_states() checks first.
EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
N_TASKS="${N_TASKS:-10}"

# name:steps:configs
RUNS="${RUNS:-\
s2:2:stock \
s4:4:stock,c3ache:4:sx \
s10:10:c3ache:4:sx}"
DEFAULT_SEEDS="$RUNS"

preflight_states() {
  echo "  -- checking how many initial states libero_goal actually holds" >&2
  cd "$WAM_ROOT" && python - "$N_TASKS" <<'PY'
import sys
# get_task_init_states loads a pickled numpy array, and PyTorch 2.6 flipped
# torch.load's weights_only default to True, so this raises UnpicklingError
# without the same shim the harness applies. An earlier version of this check
# omitted it, crashed, and reported the count as "unknown" -- which silently
# fell through to the default. A guard that cannot run is worse than no guard,
# because it reports success by saying nothing.
import torch
_ld = torch.load
torch.load = lambda *a, **k: _ld(*a, **{**k, "weights_only": False})
from libero.libero import benchmark
n = int(sys.argv[1])
s = benchmark.get_benchmark_dict()["libero_goal"]()
counts = [len(s.get_task_init_states(t)) for t in range(n)]
print(f"RESULT_PERTASK {counts}")
print(f"RESULT_MIN {min(counts)}")
if min(counts) < 50:
    print(f"    WARNING: fewer than 50 init states; set "
          f"EPISODES_PER_TASK={min(counts)} or inits[ep % len(inits)] repeats "
          f"states and the comparison regains the correlation it exists to drop")
PY
}

# Seed 7 repeats all four arms on the SAME 500 states. That is a replication
# test, not extra evidence: the states are shared, so the two collections are not
# pooled into 1000 cases -- the question is whether the paired difference keeps
# its sign and excludes zero at a second policy seed. Declared so
# check_repro_map.py claims the artefacts it writes.
# The seed-7 repeat, now that its artefacts are committed. Declaring this BEFORE
# they shipped made check_repro_map report three phantoms, which is the mistake
# baselines.sh warns about; declaring it after is what makes s2_seed7.json and
# friends claimed rather than orphaned.
PARAM_SETS="WAM_SEED=7"

run_one() {   # <name:steps:configs> <outdir>
  local spec="$1" out="$2" name steps cfgs SUF
  SUF=""; [ "${WAM_SEED:-42}" = 42 ] || SUF="_seed${WAM_SEED}"
  name=${spec%%:*};  spec=${spec#*:}
  steps=${spec%%:*}; cfgs=${spec#*:}
  case "$steps" in ''|*[!0-9]*) echo "statescale: steps must be numeric" >&2; return 2 ;; esac

  echo "  -- $name ($steps denoising steps, $cfgs, ${N_TASKS}x${EPISODES_PER_TASK} states)" >&2
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
    +PROBE.shrink="$REPO/inputs/weights/shrink_idm.json" \
    +PROBE.out="$out/${name}${SUF}.json" || return 1
}

post_collect() {
  local out="$1"
  [ -n "${DRY_RUN:-}" ] && { echo "ANALYSIS_ONLY=$out"; return 0; }
  echo
  python3 "$REPO/probes/report_statescale.py" "$out" || return 1
}

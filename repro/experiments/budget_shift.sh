# Denoising budget, cache coverage of the flow path, and noise resampling.
#
# Four groups, each arm in its own fresh Python process (arm-order effects in
# compiled graphs are documented in results/rngdrift/):
#
#   @frontier100  every schedule and cache setting for both checkpoints on the
#                 shared 100-state set (10 tasks x 10 initial states), on one host,
#                 so success and latency in the main table share a machine
#   @latency      on the same host: uncached schedules under upstream's CUDA
#                 graphs, the blockwise hook variants, and repeats of key arms
#   @shift        10-step uncached controls at both shifts for both checkpoints,
#                 and a sweep over how many leading steps are cached, so the shift
#                 effect can be separated from the share of the path the cache covers
#   @noise        the same comparisons with fresh initial noise on every call
#                 instead of FastWAM's fixed per-run latent
#   @long         2-step, 10-step and the tau=8 cache on LIBERO-Long (libero_10)
#
# The groups above record how the arms were planned and split between hosts. The
# directory groups at the end of this file regenerate the shipped layout: each
# one writes exactly the files of one directory of results/budget_shift/, and
# repro/regenerate_paper.sh runs them (@frontier100_a6000, @latency_a6000,
# @shift_a6000, @noise_a6000, @long_a6000, @repeats_a6000).
#
# Spec: name:ckpt:steps:shift:episodes_per_task:config[@opt,opt]
#   config  stock | whole:TAU | whole:TAU:a | c3ache:TAU | c3ache:TAU:sx
#   opts    prefix=N    cache steps 0..N-1 instead of all but the last two
#           noise       resample the initial latent on every call
#           cudagraphs  keep upstream reduce-overhead compilation (stock only)
#           suite=S     LIBERO suite (default libero_goal)
#           perturb=P   observation shift from fw_libero.make_perturb, e.g. cam:0.01
#                       (camera jitter in metres), seeded per (task, episode) so
#                       every arm sees the same perturbed world
#           seed=N      run seed (default 42): the fixed initial latent and, with
#                       noise, the per-episode noise streams
#           first_ep=N  start at initial state N of every task (default 0), so
#                       episodes_per_task states N, N+1, ... are run
#           tasks=N     only the first N tasks (default 10; smoke runs)
#           dump        record every executed action chunk (smoke runs)
#           ep_noise    hold each episode's first fresh latent for the whole episode
#           delay=D     execute each call's chunk D control steps late, continuing the
#                       previous chunk meanwhile; delay=R/U for refresh/reuse calls
#   config  also `screen`: the uncached policy, with every call replayed under
#           the cache configurations of probes/screen_engine.py

TITLE="Denoising budget, cached share of the flow path, and noise resampling"
SECTION="4-5"
GATE="${GATE:-yes}"
ANALYSIS=none

FRONTIER100="\
idm_stock_2step:idm:2:1.0:10:stock \
idm_stock_10step:idm:10:1.0:10:stock \
uncond_stock_2step:uncond:2:5.0:10:stock \
uncond_stock_10step:uncond:10:5.0:10:stock \
idm_whole_plain_tau8:idm:10:1.0:10:whole:8 \
uncond_whole_tau8:uncond:10:5.0:10:whole:8 \
idm_whole_agg_tau8:idm:10:1.0:10:whole:8:a \
idm_whole_4step_tau4:idm:4:1.0:10:whole:4 \
uncond_whole_4step_tau4:uncond:4:5.0:10:whole:4 \
idm_stock_1step:idm:1:1.0:10:stock \
idm_stock_4step:idm:4:1.0:10:stock \
uncond_stock_1step:uncond:1:5.0:10:stock \
uncond_stock_4step:uncond:4:5.0:10:stock \
idm_whole_plain_tau2:idm:10:1.0:10:whole:2 \
idm_whole_plain_tau4:idm:10:1.0:10:whole:4 \
uncond_whole_tau2:uncond:10:5.0:10:whole:2 \
uncond_whole_tau4:uncond:10:5.0:10:whole:4 \
uncond_whole_tau0:uncond:10:5.0:10:whole:0 \
idm_whole_plain_tau0:idm:10:1.0:10:whole:0"

# Default-compile latency for every arm above comes from its own 100-state run
# on the same host. These add upstream CUDA graphs for the uncached schedules,
# the blockwise hook variants, and a second run of key arms for run-to-run spread.
LATENCY="\
lat_uncond_stock1_cg:uncond:1:5.0:3:stock@cudagraphs \
lat_uncond_stock4_cg:uncond:4:5.0:3:stock@cudagraphs \
lat_uncond_stock10_cg:uncond:10:5.0:3:stock@cudagraphs \
lat_idm_block_skip_tau8:idm:10:1.0:3:c3ache:8:sx \
lat_idm_block_plain_tau8:idm:10:1.0:3:c3ache:8 \
lat_uncond_stock10_cg_r2:uncond:10:5.0:3:stock@cudagraphs \
lat_uncond_stock2_cg_r2:uncond:2:5.0:3:stock@cudagraphs \
lat_uncond_whole_tau8_r2:uncond:10:5.0:3:whole:8 \
lat_uncond_whole_4step_tau4_r2:uncond:4:5.0:3:whole:4 \
lat_uncond_stock10_r2:uncond:10:5.0:3:stock \
lat_uncond_stock2_r2:uncond:2:5.0:3:stock"

# The cudagraph arm leads so a failure of that path shows in minutes.
HOST1="lat_uncond_stock2_cg:uncond:2:5.0:3:stock@cudagraphs $FRONTIER100 $LATENCY"

SHIFT="\
idm_s1_stock10:idm:10:1.0:5:stock \
idm_s5_stock10:idm:10:5.0:5:stock \
uncond_s1_stock10:uncond:10:1.0:5:stock \
uncond_s5_stock10:uncond:10:5.0:5:stock \
idm_s1_p3_tau8:idm:10:1.0:5:whole:8@prefix=3 \
idm_s1_p5_tau8:idm:10:1.0:5:whole:8@prefix=5 \
idm_s1_p6_tau8:idm:10:1.0:5:whole:8@prefix=6 \
idm_s5_p9_tau8:idm:10:5.0:5:whole:8@prefix=9 \
uncond_s1_p5_tau8:uncond:10:1.0:5:whole:8@prefix=5 \
uncond_s5_p9_tau8:uncond:10:5.0:5:whole:8@prefix=9"

NOISE="\
idm_s1_stock2_rn:idm:2:1.0:5:stock@noise \
idm_s1_stock10_rn:idm:10:1.0:5:stock@noise \
idm_s1_whole_tau8_rn:idm:10:1.0:5:whole:8@noise \
uncond_s5_stock10_rn:uncond:10:5.0:5:stock@noise \
uncond_s5_whole_tau8_rn:uncond:10:5.0:5:whole:8@noise \
uncond_s5_whole_tau0_rn:uncond:10:5.0:5:whole:0@noise"

# The long-horizon suite, where C3ache reports its largest drops. Lowest
# priority, so it runs last.
LONG="\
long_uncond_stock2:uncond:2:5.0:5:stock@suite=libero_10 \
long_uncond_stock10:uncond:10:5.0:5:stock@suite=libero_10 \
long_uncond_whole_tau8:uncond:10:5.0:5:whole:8@suite=libero_10 \
long_idm_stock2:idm:2:1.0:5:stock@suite=libero_10 \
long_idm_stock10:idm:10:1.0:5:stock@suite=libero_10 \
long_idm_whole_tau8:idm:10:1.0:5:whole:8@suite=libero_10"

# Fixed-latent references for the @shift and @noise comparisons, collected on
# the same host so that no comparison in those groups crosses machines (outcomes
# reproduce exactly across two A6000 hosts, but not necessarily across GPU models).
REFS="\
idm_s1_whole_tau8:idm:10:1.0:5:whole:8 \
idm_s5_whole_tau8:idm:10:5.0:5:whole:8 \
uncond_s1_whole_tau8:uncond:10:1.0:5:whole:8 \
uncond_s5_whole_tau8:uncond:10:5.0:5:whole:8 \
idm_s1_stock2:idm:2:1.0:5:stock \
uncond_s5_stock2:uncond:2:5.0:5:stock \
uncond_s5_whole_tau0:uncond:10:5.0:5:whole:0"

# The prefix and noise arms lead so a failure of either path shows in minutes.
# Host 2 order is by priority, so an early loss of the host still leaves the
# figure's anchors: the matched-share pair and both shifts' default caches with
# their uncached 10-step controls, then the rest of the sweep, the fresh-noise
# comparisons, and LIBERO-Long last. One short fresh-noise arm runs second so
# that code path is exercised within minutes. The skip-if-collected check in
# run_one makes the repeated names below harmless.
HOST2="idm_s1_whole_tau8:idm:10:1.0:5:whole:8 \
uncond_s5_stock2_rn:uncond:2:5.0:5:stock@noise \
idm_s1_p4_tau8:idm:10:1.0:5:whole:8@prefix=4 \
idm_s5_whole_tau8:idm:10:5.0:5:whole:8 \
idm_s1_stock10:idm:10:1.0:5:stock \
idm_s5_stock10:idm:10:5.0:5:stock \
uncond_s1_whole_tau8:uncond:10:1.0:5:whole:8 \
uncond_s5_whole_tau8:uncond:10:5.0:5:whole:8 \
uncond_s1_stock10:uncond:10:1.0:5:stock \
uncond_s5_stock10:uncond:10:5.0:5:stock \
$SHIFT $REFS $NOISE $LONG"

# How the collection was actually split. Host 2's CPU made its queue too slow,
# so a third A6000 took the fresh-noise arms, their three fixed-latent
# references, and LIBERO-Long (host 2 skipped those names). The fixed-latent
# 54% -> fresh-noise 26% drop of the s=1 cache then raised the question of
# whether caching a smaller share of the path survives fresh noise; host 2 ran
# the three arms of @noise_share afterwards.
HOST3="$NOISE \
idm_s1_stock2:idm:2:1.0:5:stock \
uncond_s5_stock2:uncond:2:5.0:5:stock \
uncond_s5_whole_tau0:uncond:10:5.0:5:whole:0 \
$LONG"

NOISE_SHARE="\
idm_s1_p4_tau8_rn:idm:10:1.0:5:whole:8@prefix=4,noise \
idm_s1_p6_tau8_rn:idm:10:1.0:5:whole:8@prefix=6,noise \
idm_s5_whole_tau8_rn:idm:10:5.0:5:whole:8@noise"

# The shipped layout, one group per directory of results/budget_shift/. Two arms
# ran only in the host 2 queue above (the 40% share at s=1 and the original
# checkpoint's two-step arm under fresh noise), so they are named here too.
FRONTIER100_A6000="$FRONTIER100"
LATENCY_A6000="lat_uncond_stock2_cg:uncond:2:5.0:3:stock@cudagraphs $LATENCY"
SHIFT_A6000="@gate_uncond_s5 $SHIFT idm_s1_p4_tau8:idm:10:1.0:5:whole:8@prefix=4 $REFS"
NOISE_A6000="$NOISE uncond_s5_stock2_rn:uncond:2:5.0:5:stock@noise $NOISE_SHARE"
LONG_A6000="$LONG"
# Second copies of three @refs arms, run on host 2 after host 3 had collected
# them; the analysis uses them only to check that outcomes repeat across hosts.
# Collect them into their own directory, or the skip-if-collected check below
# skips them as already done.
REPEATS_A6000="\
idm_s1_stock2:idm:2:1.0:5:stock \
uncond_s5_stock2:uncond:2:5.0:5:stock \
uncond_s5_whole_tau0:uncond:10:5.0:5:whole:0"

# Every arm behind results/budget_shift/ except the repeats, in the order the
# hosts ran them.
DEFAULT_SEEDS="@host1 @host2 @noise_share"

run_one() {   # <spec | @group> <outdir>
  local spec="$1" out="$2" g
  case "$spec" in
    @frontier100) g="$FRONTIER100" ;;
    @latency) g="$LATENCY" ;;
    @shift) g="$SHIFT" ;;
    @noise) g="$NOISE" ;;
    @long) g="$LONG" ;;
    @refs) g="$REFS" ;;
    @host1) g="$HOST1" ;;
    @host2) g="$HOST2" ;;
    @host3) g="$HOST3" ;;
    @noise_share) g="$NOISE_SHARE" ;;
    @frontier100_a6000) g="$FRONTIER100_A6000" ;;
    @latency_a6000) g="$LATENCY_A6000" ;;
    @shift_a6000) g="$SHIFT_A6000" ;;
    @noise_a6000) g="$NOISE_A6000" ;;
    @long_a6000) g="$LONG_A6000" ;;
    @repeats_a6000) g="$REPEATS_A6000" ;;
    @gate_uncond_s5)
      # run_experiment.sh gates the Optional-IDM checkpoint at its default shift
      # before any arm. The original checkpoint at s=5 caches a different part of
      # the path, so the shift host also gated it (gate_uncond_s5.json).
      if [ -s "$out/gate_uncond_s5.json" ] && [ "${RERUN:-0}" != 1 ]; then
        echo "  -- gate_uncond_s5 already collected; skipping" >&2; return 0
      fi
      echo "  -- identity gate, original checkpoint at s=5" >&2
      ( cd "$WAM_ROOT" && python probes/gate_cache.py \
          ckpt="$CKPT_DIR/libero_uncond_2cam224.pt" \
          model.redirect_common_files=false \
          EVALUATION.dataset_stats_path="$CKPT_DIR/libero_uncond_2cam224_dataset_stats.json" \
          EVALUATION.sigma_shift=5.0 \
          +PROBE.out="$out/gate_uncond_s5.json" ) || return 1
      [ -n "${DRY_RUN:-}" ] || grep -q '"pass": true' "$out/gate_uncond_s5.json" \
        || { echo "budget_shift: identity gate FAILED for the original checkpoint" >&2; return 1; }
      return 0 ;;
    @*) echo "budget_shift: unknown group $spec" >&2; return 2 ;;
    *) g="" ;;
  esac
  if [ -n "$g" ]; then
    for s in $g; do run_one "$s" "$out"; done
    return 0
  fi

  local name ckpt_type steps shift n_episodes cfg opts=""
  name=${spec%%:*}; spec=${spec#*:}
  ckpt_type=${spec%%:*}; spec=${spec#*:}
  steps=${spec%%:*}; spec=${spec#*:}
  shift=${spec%%:*}; spec=${spec#*:}
  n_episodes=${spec%%:*}; cfg=${spec#*:}
  case "$cfg" in *@*) opts=${cfg#*@}; cfg=${cfg%%@*} ;; esac

  case "$steps" in ''|*[!0-9]*) echo "budget_shift: bad steps in $name" >&2; return 2 ;; esac
  case "$n_episodes" in ''|*[!0-9]*) echo "budget_shift: bad episodes in $name" >&2; return 2 ;; esac

  local ckpt_file stats_file task_name shrink_file extra=() cg=0 o suite=libero_goal seed=42 n_tasks=10
  if [ "$ckpt_type" = idm ]; then
    ckpt_file="$CKPT_DIR/libero_optional_idm_2cam224.pt"
    stats_file="$CKPT_DIR/libero_optional_idm_2cam224_dataset_stats.json"
    task_name="libero_optional_idm_2cam224_1e-4"
    extra+=("+EVALUATION.action_infer_mode=first_frame")
    shrink_file="$REPO/inputs/weights/shrink_idm.json"
  elif [ "$ckpt_type" = uncond ]; then
    ckpt_file="$CKPT_DIR/libero_uncond_2cam224.pt"
    stats_file="$CKPT_DIR/libero_uncond_2cam224_dataset_stats.json"
    task_name="libero_uncond_2cam224_1e-4"
    shrink_file="$REPO/inputs/weights/shrink_fw.json"
  else
    echo "budget_shift: unknown checkpoint type $ckpt_type" >&2; return 2
  fi

  local IFS_OLD="$IFS"; IFS=,
  for o in $opts; do
    case "$o" in
      prefix=*) extra+=("+PROBE.cache_prefix=${o#prefix=}") ;;
      noise) extra+=("+PROBE.resample_noise=1") ;;
      suite=*) suite=${o#suite=} ;;
      perturb=*) extra+=("+PROBE.perturb=${o#perturb=}") ;;
      seed=*) seed=${o#seed=} ;;
      first_ep=*) extra+=("+PROBE.first_episode=${o#first_ep=}") ;;
      tasks=*) n_tasks=${o#tasks=} ;;
      dump) extra+=("+PROBE.dump_chunks=1") ;;
      ep_noise) extra+=("+PROBE.episode_noise=1") ;;
      delay=*) extra+=("+PROBE.delay=${o#delay=}") ;;
      cudagraphs)
        [ "$cfg" = stock ] || { IFS="$IFS_OLD"; echo "budget_shift: cudagraphs needs stock ($name)" >&2; return 2; }
        cg=1 ;;
      "") ;;
      *) IFS="$IFS_OLD"; echo "budget_shift: unknown option $o in $name" >&2; return 2 ;;
    esac
  done
  IFS="$IFS_OLD"

  if [ -s "$out/${name}.json" ] && [ "${RERUN:-0}" != 1 ]; then
    echo "  -- $name already collected; skipping (RERUN=1 to repeat)" >&2
    return 0
  fi
  echo "  -- $name ($suite, $ckpt_type, $steps steps, shift $shift, $n_episodes/task, $cfg${opts:+, $opts}, fresh process)" >&2
  if ! ( cd "$WAM_ROOT" && WAM_CUDAGRAPHS="$cg" python probes/fw_libero.py \
      task="$task_name" \
      ckpt="$ckpt_file" \
      model.redirect_common_files=false \
      EVALUATION.dataset_stats_path="$stats_file" \
      EVALUATION.sigma_shift="$shift" \
      EVALUATION.num_inference_steps="$steps" \
      seed="$seed" \
      ${extra[@]+"${extra[@]}"} \
      +PROBE.suite="$suite" +PROBE.n_tasks="$n_tasks" +PROBE.n_episodes="$n_episodes" \
      +PROBE.configs=\'"$cfg"\' \
      +PROBE.timing=1 +PROBE.warmup_episodes=1 +PROBE.cycle_times=1 \
      +PROBE.shrink="$shrink_file" \
      +PROBE.out="$out/${name}.json" ); then
    # Keep going: one broken arm must not cost the arms queued behind it.
    echo "$name" >> "$out/FAILED_ARMS.txt"
    echo "  !! $name FAILED; continuing with the next arm" >&2
  fi
  return 0
}

post_collect() {
  local out="$1"
  [ -n "${DRY_RUN:-}" ] && { echo "ANALYSIS_ONLY=$out"; return 0; }
  python3 - "$out" <<'PY'
import glob, json, os, sys
out = sys.argv[1]
for p in sorted(glob.glob(os.path.join(out, "*.json"))):
    try:
        d = json.load(open(p))
    except Exception:
        continue
    if not isinstance(d, dict) or "rows" not in d:
        continue
    rows = d["rows"]
    ok = sum(r["success"] for r in rows)
    ms = [r["ms_per_cycle"] for r in rows if not r.get("warmup") and "ms_per_cycle" in r]
    lat = f"{sum(ms)/len(ms):7.1f} ms" if ms else "      -"
    m = d["meta"]
    print(f"  {os.path.basename(p)[:-5]:34s} {ok:3d}/{len(rows):3d} {lat}"
          f"  prefix={m.get('cache_prefix')} frac={m.get('cached_path_fraction')}"
          f" noise={m.get('resample_noise')} cg={m.get('cudagraphs')}")
failed = os.path.join(out, "FAILED_ARMS.txt")
if os.path.exists(failed):
    print("FAILED ARMS:", open(failed).read().split())
PY
}

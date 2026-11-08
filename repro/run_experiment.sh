#!/usr/bin/env bash
# Run one experiment ON a prepared GPU host. No cloud provider, no provisioning.
#
# The contract: given a machine that satisfies setup/HOST_REQUIREMENTS.md, this
# reproduces the paper's numbers for one experiment within the documented run-to-
# run variance. How you obtain that machine is your business.
#
#   ./repro/run_experiment.sh list
#   ./repro/run_experiment.sh frontier                  # default seeds
#   ./repro/run_experiment.sh frontier --seeds "42 7"   # pick your own
#   ./repro/run_experiment.sh frontier --episodes 4     # a fast smoke run
#
# Artefacts land in --out (default: results_new/<experiment>/) as JSON, which
# repro/verify.sh and probes/report_variance.py both read. Copy them back to a
# machine with the repo and run:
#
#   python3 probes/report_variance.py closedloop <dir>
#
# Paths are overridable, because a reviewer's layout will not match ours:
#   WAM_ROOT   FastWAM source, with probes/ inside it   (default /w)
#   CKPT_DIR   model checkpoints                        (default /workspace/ckpt)

set -u
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export WAM_ROOT="${WAM_ROOT:-/w}"
export CKPT_DIR="${CKPT_DIR:-/workspace/ckpt}"
export REPO="$ROOT"

# DiffSynth defaults its download source to modelscope, which sustains ~300 kB/s
# from most datacentres. The same files come from HuggingFace 60-170x faster.
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-huggingface}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { printf '\n!! %s\n' "$*" >&2; exit 1; }

usage() {
  cat >&2 <<USAGE
usage: run_experiment.sh <experiment> [--seeds "A B C"] [--episodes N] [--out DIR]
       run_experiment.sh list
       run_experiment.sh preflight     check this host can run anything at all
USAGE
  exit 2
}

preflight() {
  # Not every experiment drives the simulator. The residual probes, the replay
  # studies and the timing sweep read recorded video and never open an env, and
  # their setup scripts deliberately do not install mujoco/LIBERO/EGL. Demanding
  # those of every experiment fails a host that is perfectly able to run the one
  # you asked for -- which is what it did on an H100 that had everything the
  # residual probe needs.
  # One experiment needs no GPU at all. `weights` refits the substitution
  # coefficients from the recorded Gram matrices, which is the check a reviewer
  # should be able to run before renting anything -- it is what distinguishes a
  # closed-form fit from a sweep. Demanding CUDA and fastwam of it meant the one
  # experiment documented as running anywhere ran nowhere except a rented box.
  log "checking this host (simulator required: ${NEEDS_SIM:-yes}, gpu required: $([ "${NO_GPU:-no}" = yes ] && echo no || echo yes))"
  NEEDS_SIM="${NEEDS_SIM:-yes}" NO_GPU="${NO_GPU:-no}" \
  STACK="${STACK:-fastwam}" MUJOCO_PIN="${MUJOCO_PIN-3.3.2}" \
  "${PYBIN:-python3}" - <<'PY' || die "host is not ready; see setup/HOST_REQUIREMENTS.md"
import os, sys
ok = True
need_sim = os.environ.get("NEEDS_SIM", "yes") != "no"
need_gpu = os.environ.get("NO_GPU", "no") != "yes"
# Not every experiment runs on FastWAM. pi0.5 is served by lerobot in its own
# virtualenv, has no fastwam to import, and carries whatever mujoco lerobot
# pins -- 3.8.1, not the 3.3.2 that FastWAM's LIBERO replay data was rendered
# with. Demanding both of it failed a host that was entirely able to run it.
stack = os.environ.get("STACK", "fastwam")
mj_pin = os.environ.get("MUJOCO_PIN", "3.3.2")
def chk(name, fn):
    global ok
    try:
        v = fn(); print(f"  {name:22s} {v}")
    except Exception as e:
        print(f"  {name:22s} FAIL: {e}"); ok = False
import torch
chk("torch", lambda: torch.__version__)
def cuda():
    a = torch.zeros(8, device="cuda"); a += 1; torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    gb = total / 1e9
    if gb < 38: raise RuntimeError(f"only {gb:.0f} GB VRAM; need >= 40")
    return f"{torch.cuda.get_device_name(0)} ({gb:.0f} GB)"
if need_gpu: chk("cuda", cuda)
def egl():
    os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from OpenGL import EGL
    if not EGL.eglQueryString: raise RuntimeError("EGL present but not initialised")
    return "initialised"
if need_sim: chk("egl", egl)
def mj():
    import mujoco
    if mj_pin and mujoco.__version__ != mj_pin:
        raise RuntimeError(f"mujoco {mujoco.__version__}; this experiment needs {mj_pin} "
                           "(2.3.7 costs ~22 points of success on FastWAM's data)")
    return mujoco.__version__ + ("" if mj_pin else " (no pin for this stack)")
if need_sim: chk("mujoco", mj)
if need_sim:
    chk("robosuite", lambda: __import__("robosuite").__version__)
    chk("libero", lambda: (__import__("libero.libero", fromlist=["libero"]), "importable")[1])
if need_gpu and stack == "fastwam":
    chk("fastwam", lambda: (__import__("fastwam"), "importable")[1])
elif stack == "lerobot":
    chk("lerobot", lambda: (__import__("lerobot"), "importable")[1])
if not need_gpu: chk("numpy", lambda: __import__("numpy").__version__)
sys.exit(0 if ok else 1)
PY
  # Check the checkpoint THIS experiment loads, which each experiment file now
  # declares. The check used to name one file unconditionally, so the four
  # experiments using the unconditional checkpoint were never verified at all --
  # they are NEEDS_SIM=no, so they skipped even that -- and failed minutes later
  # inside model loading on a host preflight had just pronounced ready.
  if [ -n "${CKPT:-}" ]; then
    # CKPT is the experiment's default, but a tag can select a different one:
    # mechanism --seeds idm/idm7 replays the DEPLOYMENT checkpoint, not the
    # unconditional one CKPT names. Checking CKPT alone pronounced "host OK" on
    # a machine holding only the unconditional file and then failed inside model
    # loading -- the same shape of bug this block was written to end. An
    # experiment that maps tags to checkpoints declares ckpt_for_seed, and every
    # requested tag is verified through it.
    _ckpts="$CKPT"
    if declare -f ckpt_for_seed >/dev/null 2>&1 && [ -n "${SEEDS:-}" ]; then
      _ckpts=""
      for _s in $SEEDS; do _ckpts="$_ckpts $(ckpt_for_seed "$_s")"; done
    fi
    for _c in $_ckpts; do
      [ -s "$CKPT_DIR/$_c" ] || die "checkpoint missing: $CKPT_DIR/$_c
    $EXP${SEEDS:+ (seeds: $SEEDS)} needs this file. setup/HOST_REQUIREMENTS.md says which setup script fetches it."
    done
  elif [ "${NEEDS_SIM:-yes}" = yes ] && [ "${STACK:-fastwam}" = fastwam ]; then
    # Only the FastWAM stack loads a checkpoint from disk. pi0.5 pulls its model
    # through lerobot at run time, so demanding a local .pt failed a host that
    # had everything that experiment actually needs.
    [ -s "$CKPT_DIR/libero_optional_idm_2cam224.pt" ] \
      || die "checkpoint missing: $CKPT_DIR/libero_optional_idm_2cam224.pt"
  fi
  # Two experiments read recorded video from a directory that was hardcoded in
  # their probe invocation and documented nowhere. A reviewer whose videos are
  # elsewhere had no way to say so and no warning until the probe failed.
  if [ -n "${VIDEO_ROOT:-}" ]; then
    [ -d "$VIDEO_ROOT" ] || die "video root missing: $VIDEO_ROOT
    $EXP replays recorded episodes from there. Set VIDEO_ROOT to your copy."
  fi
  log "host OK"
}

list_experiments() {
  printf "\n  %-13s %-8s %s\n" EXPERIMENT SECTION TITLE
  for f in "$ROOT"/repro/experiments/*.sh; do
    ( TITLE=""; SECTION=""; DEFAULT_SEEDS=""; . "$f" >/dev/null 2>&1
      printf "  %-13s %-8s %s\n" "$(basename "$f" .sh)" "§$SECTION" "$TITLE" )
  done; echo
}

[ $# -ge 1 ] || usage
EXP="$1"; shift
case "$EXP" in
  list) list_experiments; exit 0 ;;
  preflight) preflight; exit 0 ;;
  -h|--help) usage ;;
esac
[ -f "$ROOT/repro/experiments/$EXP.sh" ] || { echo "unknown experiment: $EXP" >&2; list_experiments; exit 2; }

SEEDS=""; EPISODES=""; OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --seeds)    SEEDS="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --out)      OUT="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; usage ;;
  esac
done

# DRY_RUN=1 prints the probe invocations this experiment would issue and exits.
# A reviewer should be able to see what a run will do before committing GPU
# hours to it, and it is the only way to check that an experiment's defaults
# still name the artefacts that shipped.
if [ "${DRY_RUN:-0}" = 1 ]; then
  cd()      { :; }
  # Some probes take their output path from the environment as a command prefix
  # (PI_OUT=... python probes/x.py), which never appears in "$*". The shim reads
  # it from its own environment so the dry run reports what will actually be
  # written rather than only what is passed as an argument.
  _shim() { printf '  %s %s\n' "$1" "${*:2}"
            # Probes that take their output path from the environment rather than
            # an argument: each backbone's collector uses its own variable, and a
            # dry run that misses one silently reports an experiment as writing
            # nothing at all.
            for v in PI_OUT DZ_OUT FW_OUT OUT_PATH; do
              eval "val=\${$v:-}"
              [ -n "$val" ] && printf '  +PROBE.out=%s\n' "$val"
            done
            # budget_shift passes upstream CUDA graphs the same way, and a dry
            # run that hid it could not be checked against the shipped arms.
            [ "${WAM_CUDAGRAPHS:-0}" = 1 ] && printf '  env WAM_CUDAGRAPHS=1\n'
            return 0; }
  python()  { _shim python "$@"; }
  python3() { _shim python3 "$@"; }
  # pip is shimmed because one experiment reinstalls mujoco inside run_one to
  # measure the version hazard. A dry run that changes the environment it is
  # describing is worse than no dry run -- this one started building a mujoco
  # wheel on a laptop before it was caught.
  pip()     { _shim pip "$@"; }
  pip3()    { _shim pip3 "$@"; }
fi

. "$ROOT/repro/experiments/$EXP.sh"
[ -n "$SEEDS" ] || SEEDS="${DEFAULT_SEEDS:-42}"
[ -n "$EPISODES" ] && EPISODES_OVERRIDE="$EPISODES"
EPISODES="${EPISODES_OVERRIDE:-${EPISODES:-10}}"
[ -n "$OUT" ] || OUT="$ROOT/results_new/$EXP"
[ "${DRY_RUN:-0}" = 1 ] || mkdir -p "$OUT"
export EPISODES OUT

if [ "${DRY_RUN:-0}" = 1 ]; then
  _dry_ckpt="${CKPT:-none}"
  if declare -f ckpt_for_seed >/dev/null 2>&1; then
    _dry_ckpt=""
    for _s in $SEEDS; do _dry_ckpt="$_dry_ckpt $(ckpt_for_seed "$_s")"; done
    _dry_ckpt="$(echo $_dry_ckpt | tr ' ' '\n' | sort -u | tr '\n' ' ')"
  fi
  printf '%s\n  seeds: %s | episodes: %s | out: %s\n  checkpoint: %s\n' \
    "$TITLE" "$SEEDS" "$EPISODES" "$OUT" "$_dry_ckpt"
  [ -n "${VIDEO_ROOT:-}" ] && printf '  video root: %s\n' "$VIDEO_ROOT"
  for seed in $SEEDS; do run_one "$seed" "$OUT"; done
  # An experiment that assembles its shipped artefact after collection has to
  # declare it here too, or a dry run reports the shards and misses the file the
  # analysis actually reads.
  if declare -f post_collect >/dev/null 2>&1; then post_collect "$OUT"; fi
  exit 0
fi

PYBIN="${PYBIN:-python3}"; export PYBIN
if command -v "$PYBIN" >/dev/null 2>&1; then
  _pybin_full="$(command -v "$PYBIN")"
  PATH="$(dirname "$_pybin_full"):$PATH"; export PATH
fi
# Which interpreter this stack lives in. pi0.5 is installed by uv into its own
# virtualenv at /workspace/pienv and nothing puts that on PATH, so both the
# preflight and the probe were running under the system interpreter -- where
# lerobot, mujoco, robosuite and PyOpenGL are all absent. The preflight then
# reported five FAILs on a host that was perfectly set up, and the experiment
# could not have run even if the preflight were skipped. Prepending the venv to
# PATH fixes both at once and leaves `python probes/...` in the experiment
# scripts meaning the right thing.
if [ "${STACK:-fastwam}" = lerobot ]; then
  PIENV="${PIENV:-/workspace/pienv}"
  if [ -x "$PIENV/bin/python" ]; then
    PATH="$PIENV/bin:$PATH"; export PATH
    PYBIN="$PIENV/bin/python"; export PYBIN
    log "using the lerobot virtualenv at $PIENV"
  else
    die "STACK=lerobot but no interpreter at $PIENV/bin/python
    setup/onstart_pi05_libero.sh creates it. Set PIENV if yours is elsewhere."
  fi
fi

preflight

# The probe scripts must live inside WAM_ROOT, next to FastWAM's configs/.
# Hydra resolves each probe's config_path="../configs" relative to the FILE, so a
# probe run from a cloned repo elsewhere cannot find the model config at all.
# ...but only for experiments that load the model. `weights` reads recorded Gram
# matrices with numpy and never constructs a hydra config, so requiring a FastWAM
# checkout for it would put the no-GPU check behind the GPU setup it exists to
# precede.
if [ "${NO_GPU:-no}" != yes ] \
   && [ "$(cd "$REPO/probes" && pwd)" != "$(cd "$WAM_ROOT" 2>/dev/null && pwd)/probes" ]; then
  log "installing probes into $WAM_ROOT/probes (hydra needs them beside configs/)"
  mkdir -p "$WAM_ROOT/probes" || die "cannot write to $WAM_ROOT -- set WAM_ROOT to your FastWAM checkout"
  cp "$REPO"/probes/*.py "$WAM_ROOT/probes/" || die "failed to install probes"
fi
# Only hydra-driven probes need FastWAM's configs/ beside them. pi0.5 is
# configured entirely through environment variables and never constructs a hydra
# config, so requiring a FastWAM checkout of it blocked an experiment that does
# not use FastWAM at all. It still needs the probe files themselves, which the
# copy above provides.
if [ "${NO_GPU:-no}" != yes ] && [ "${STACK:-fastwam}" = fastwam ]; then
  [ -d "$WAM_ROOT/configs" ] \
    || die "no configs/ in $WAM_ROOT -- WAM_ROOT must be the FastWAM source tree"
fi

# The identity gate is not optional for cache experiments. Compiling with CUDA
# graph capture silently overwrites cached residuals and yields plausible wrong
# numbers with no error; this replays a cycle substituting the residuals it just
# recorded, which is an algebraic identity, and refuses to continue if it fails.
if [ "${GATE:-no}" = yes ]; then
  log "cache identity gate"
  gate_ok=no
  for attempt in 1 2 3; do
    rm -f "$OUT/gate.json"
    ( cd "$WAM_ROOT" && python probes/gate_cache.py \
        ckpt="$CKPT_DIR/libero_optional_idm_2cam224.pt" \
        model.redirect_common_files=false \
        EVALUATION.dataset_stats_path="$CKPT_DIR/libero_optional_idm_2cam224_dataset_stats.json" \
        +PROBE.out="$OUT/gate.json" )
    # The gate writes gate.json only once it has measured something. Its ABSENCE
    # means it never ran -- a broken model download, an OOM -- which is a
    # different thing from the gate running and failing, and must not be
    # reported as a cache fault.
    if [ -s "$OUT/gate.json" ]; then
      grep -q '"pass": true' "$OUT/gate.json" && { gate_ok=yes; break; }
      die "cache identity gate FAILED. Cached residuals are not surviving the call
    that produced them, so every number from this host would be an artefact. Check
    that nothing re-enabled CUDA graph capture (mode='reduce-overhead' or inductor
    cudagraphs); see the shim in probes/fw_libero.py."
    fi
    log "  gate produced no result (attempt $attempt/3) -- it never ran. Retrying;"
    log "  a partially downloaded model resumes rather than restarting."
    sleep 15
  done
  [ "$gate_ok" = yes ] || die "the identity gate could not be RUN after 3 attempts.
    This is NOT an identity failure -- it never got far enough to measure. The
    underlying error is above; a broken model download is the usual cause."
fi

log "$TITLE"
log "seeds: $SEEDS | episodes/arm/task: $EPISODES | out: $OUT"
t0=$(date +%s)
for seed in $SEEDS; do
  log "--- seed $seed ---"
  run_one "$seed" "$OUT" || die "seed $seed failed"
done
# An experiment whose shipped artefact is a merge of several collections has to
# assemble it here, or the analysis reads the shards and reports a different
# quantity from the one the write-up pools. attenuation collects libero_10 as two
# disjoint task ranges and merges them.
if declare -f post_collect >/dev/null 2>&1; then
  # A failing post-collection step is NOT a failing experiment, and conflating
  # them throws away good data. The latency run collected all 270 episodes
  # correctly and then exited 1 because the ANALYSIS on the box was a version
  # behind -- the driver ships the repo at launch, so a fix that lands mid-run
  # never reaches the machine. A chain reading that exit code stopped two more
  # experiments that had nothing wrong with them.
  #
  # Exit 3 instead, distinctly: the artefacts are on disk and worth pulling, and
  # the analysis can be re-run anywhere against them.
  if ! post_collect "$OUT"; then
    log "post_collect FAILED -- the artefacts in $OUT are complete and unaffected."
    log "Re-run the analysis locally against them; do not re-collect."
    exit 3
  fi
fi
# An experiment that mutates the host must put it back, or the next one on the
# same box inherits the change. mujoco downgrades the physics engine on purpose.
if command -v restore_env >/dev/null 2>&1 || declare -f restore_env >/dev/null 2>&1; then
  restore_env || log "restore_env failed -- the host may be left modified"
fi
log "done in $(( ($(date +%s)-t0)/60 )) min"

# ANALYSIS=none means the experiment reported its own result already -- `weights`
# prints the refit-vs-shipped comparison itself, and there are no episode
# outcomes to summarise. Calling the variance reporter anyway printed its usage
# string after a clean run, which reads like a failure at the end of a success.
if [ "${ANALYSIS:-closedloop}" != none ]; then
  echo
  python3 "$ROOT/probes/report_variance.py" "${ANALYSIS:-closedloop}" "$OUT" || true
fi

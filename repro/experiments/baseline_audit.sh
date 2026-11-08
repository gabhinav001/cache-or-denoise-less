# A bounded baseline audit, with an independent Python process for every arm.
#
# Original public checkpoint: uncond + scheduler shift 5.0 (upstream README).
# Optional-IDM checkpoint: its own task + explicit first_frame + shift 1.0.
# Neither choice establishes the otherwise unspecified C3ache checkpoint identity.
#
# Stages are deliberate, not automatically chained on a paid machine:
#   BASELINE_PHASE=smoke  1 task x 2 states, eager by default, stock repeated
#   BASELINE_PHASE=pilot 10 tasks x 5 states, compiled, stock repeated
#   BASELINE_PHASE=full  10 tasks x 50 states, compiled, three distinct arms
# Full requires a passed pilot in the same --out tree, variant, and policy seed.
# Parent/orchestrator enforces elapsed-time and dollar caps; this script never
# provisions hardware, retries failed runs, or automatically starts another stage.
#
# Example: BASELINE_VARIANT=original BASELINE_PHASE=smoke \
#   bash repro/run_experiment.sh baseline_audit --seeds 42 --out /workspace/baseline_audit
# A fresh --out directory is required for a repeat: outputs are never overwritten.
TITLE="Fresh-process stock, whole-stack, and blockwise baseline audit"
SECTION="01e"
PENDING=yes
ANALYSIS=none
# The common harness's gate hardcodes optional-IDM. run_one invokes the gate
# itself with this audit's exact checkpoint, task, scheduler, and compiler mode.
GATE=no
NEEDS_SIM=yes
STACK=fastwam

BASELINE_VARIANT="${BASELINE_VARIANT:-original}"
BASELINE_PHASE="${BASELINE_PHASE:-smoke}"
DEFAULT_SEEDS="42"
case "$BASELINE_VARIANT" in
  original)
    CKPT=libero_uncond_2cam224.pt
    BASELINE_TASK=libero_uncond_2cam224_1e-4
    BASELINE_STATS=libero_uncond_2cam224_dataset_stats.json
    BASELINE_SHIFT=5.0
    ;;
  optional)
    CKPT=libero_optional_idm_2cam224.pt
    BASELINE_TASK=libero_optional_idm_2cam224_1e-4
    BASELINE_STATS=libero_optional_idm_2cam224_dataset_stats.json
    BASELINE_SHIFT=1.0
    ;;
  *) echo "baseline_audit: BASELINE_VARIANT must be original or optional" >&2; return 2 ;;
esac
case "$BASELINE_PHASE" in
  smoke) BASELINE_TASKS=1; BASELINE_EPS=2; BASELINE_DEFAULT_COMPILE=false ;;
  pilot) BASELINE_TASKS=10; BASELINE_EPS=5; BASELINE_DEFAULT_COMPILE=true ;;
  full) BASELINE_TASKS=10; BASELINE_EPS=50; BASELINE_DEFAULT_COMPILE=true ;;
  *) echo "baseline_audit: BASELINE_PHASE must be smoke, pilot, or full" >&2; return 2 ;;
esac
BASELINE_COMPILE="${BASELINE_COMPILE:-$BASELINE_DEFAULT_COMPILE}"
case "$BASELINE_COMPILE" in true|false) ;; *) echo "BASELINE_COMPILE must be true or false" >&2; return 2 ;; esac
# The common wrapper displays this value, but the collector uses BASELINE_EPS;
# a generic --episodes override must not silently shrink a claimed full run.
EPISODES="$BASELINE_EPS"

baseline_audit_record_command() {  # <manifest.json> <command and arguments...>
  local manifest="$1"; shift
  python3 - "$manifest" "$@" <<'PY'
import datetime,json,sys
from pathlib import Path
p=Path(sys.argv[1])
p.write_text(json.dumps({"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
                         "argv":sys.argv[2:]},indent=2)+"\n")
PY
}

run_one() {  # <numeric policy seed> <stable audit root>
  local seed="$1" out="$2" run_dir compile="$BASELINE_COMPILE" tag arm result log config
  case "$seed" in ''|*[!0-9]*) echo "baseline_audit: seed must be numeric" >&2; return 2 ;; esac
  if [ "$EPISODES" != "$BASELINE_EPS" ]; then
    echo "baseline_audit: choose BASELINE_PHASE; --episodes is not supported" >&2; return 2
  fi
  run_dir="$out/$BASELINE_VARIANT/$BASELINE_PHASE/seed$seed"
  local -a common gate_args tags arms command
  common=("task=$BASELINE_TASK" "ckpt=$CKPT_DIR/$CKPT"
    "model.redirect_common_files=false" "mixed_precision=bf16" "seed=$seed"
    "EVALUATION.dataset_stats_path=$CKPT_DIR/$BASELINE_STATS"
    "EVALUATION.sigma_shift=$BASELINE_SHIFT" "EVALUATION.num_inference_steps=10"
    "EVALUATION.action_horizon=32" "EVALUATION.replan_steps=10"
    "EVALUATION.num_steps_wait=30" "EVALUATION.env_num=1"
    "EVALUATION.binarize_gripper=true" "EVALUATION.use_action_ensembler=false"
    "EVALUATION.visualize_future_video=false" "EVALUATION.rand_device=cpu"
    "EVALUATION.text_cfg_scale=1.0" "EVALUATION.compile_action_infer=$compile")
  if [ "$BASELINE_VARIANT" = optional ]; then
    common+=("+EVALUATION.action_infer_mode=first_frame")
  fi
  tags=(00_stock_a 01_whole_tau8 02_block_tau8)
  arms=(stock whole:8 c3ache:8)
  if [ "$BASELINE_PHASE" != full ]; then
    tags+=(03_stock_b); arms+=(stock)
  fi
  gate_args=("${common[@]}" "+PROBE.suite=libero_goal" "+PROBE.task=0" "+PROBE.out=$run_dir/gate.json")

  if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "  audit variant=$BASELINE_VARIANT phase=$BASELINE_PHASE seed=$seed compile=$compile"
    echo "  output=$run_dir; tasks=$BASELINE_TASKS; states/task=$BASELINE_EPS; fresh process per arm"
    python probes/gate_cache.py "${gate_args[@]}"
  else
    [ -s "$CKPT_DIR/$BASELINE_STATS" ] || { echo "Missing paired dataset stats: $CKPT_DIR/$BASELINE_STATS" >&2; return 1; }
    [ ! -e "$run_dir" ] || { echo "Refusing to overwrite audit directory: $run_dir" >&2; return 1; }
    if [ "$BASELINE_PHASE" = full ]; then
      python3 - "$out/$BASELINE_VARIANT/pilot/seed$seed/stock_repeat_check.json" "$compile" <<'PY' || return 1
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file(): raise SystemExit("Full evaluation requires a completed pilot in the same output tree")
d=json.loads(p.read_text())
if not d.get("pass") or d.get("compile_action_infer") != (sys.argv[2]=="true"):
    raise SystemExit("Full evaluation requires a passed pilot using the same compiler mode")
PY
    fi
    if [[ ! "${WAM_SOURCE_BUNDLE_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]; then
      if [ -f "${WAM_ROOT}/source_bundle.tar.gz" ]; then
        WAM_SOURCE_BUNDLE_SHA256=$(sha256sum "${WAM_ROOT}/source_bundle.tar.gz" | cut -d' ' -f1)
      elif [ -f "${ROOT}/inputs/source_bundle.tar.gz" ]; then
        WAM_SOURCE_BUNDLE_SHA256=$(sha256sum "${ROOT}/inputs/source_bundle.tar.gz" | cut -d' ' -f1)
      else
        # Dynamic digest of executed probe sources to avoid masking local code changes with an obsolete hash
        WAM_SOURCE_BUNDLE_SHA256=$(python3 -c "import hashlib, glob; h=hashlib.sha256(); [h.update(open(f,'rb').read()) for f in sorted(glob.glob('${ROOT}/probes/*.py'))]; print(h.hexdigest())")
      fi
      export WAM_SOURCE_BUNDLE_SHA256
    fi
    mkdir -p "$run_dir" || return 1
    baseline_audit_record_command "$run_dir/gate_command.json" python probes/gate_cache.py "${gate_args[@]}" || return 1
    (cd "$WAM_ROOT" && python probes/gate_cache.py "${gate_args[@]}") >"$run_dir/gate.log" 2>&1 || {
      tail -60 "$run_dir/gate.log" >&2; return 1;
    }
    PYTHONPATH="$ROOT/probes${PYTHONPATH:+:$PYTHONPATH}" python3 - "$run_dir" "$BASELINE_VARIANT" "$BASELINE_PHASE" <<'PYGATE' || return 1
import json,sys
from pathlib import Path
from report_baseline_audit import read_collection, require, validate_runtime, RUNTIME_FIELDS
root=Path(sys.argv[1]);variant,phase=sys.argv[2:]
gate=json.loads((root/"gate.json").read_text())
require(gate.get("pass") is True,"Matched-checkpoint cache identity gate failed")
validate_runtime(gate.get("runtime"),variant)
if phase=="full":
    pilot=read_collection(root.parents[1]/"pilot"/root.name,"pilot",variant)
    require(pilot["valid"],"Pilot did not pass validation")
    require(all(gate["runtime"][k]==pilot["identity"]["runtime"][k] for k in RUNTIME_FIELDS),
            "Current source/checkpoint/runtime differs from pilot; refusing costly full evaluation")
PYGATE
  fi

  local index
  for index in "${!tags[@]}"; do
    tag="${tags[$index]}"; arm="${arms[$index]}"
    result="$run_dir/$tag.json"; log="$run_dir/$tag.log"; config="$run_dir/$tag.config.yaml"
    command=(python probes/fw_libero.py "${common[@]}"
      "+PROBE.suite=libero_goal" "+PROBE.n_tasks=$BASELINE_TASKS" "+PROBE.n_episodes=$BASELINE_EPS"
      "+PROBE.configs=$arm" "+PROBE.timing=1" "+PROBE.cycle_times=1"
      "+PROBE.warmup_episodes=1" "+PROBE.no_reseed=0"
      "+PROBE.shrink=$run_dir/no_calibration_used.json" "+PROBE.out=$result")
    if [ "${DRY_RUN:-0}" = 1 ]; then
      echo "  resolved config -> $config"
      "${command[@]}" --cfg job --resolve
      "${command[@]}"
      continue
    fi
    baseline_audit_record_command "$run_dir/$tag.command.json" "${command[@]}" || return 1
    (cd "$WAM_ROOT" && "${command[@]}" --cfg job --resolve) >"$config" 2>"$run_dir/$tag.config.stderr" || return 1
    echo "  -- $tag: $BASELINE_TASKS tasks x $BASELINE_EPS states; log=$log" >&2
    (cd "$WAM_ROOT" && "${command[@]}") >"$log" 2>&1 || { tail -60 "$log" >&2; return 1; }
  done
  [ "${DRY_RUN:-0}" = 1 ] && { echo "  validate complete paired coverage and stock-repeat outcomes"; return 0; }

  # Share the strict collection validator with the manuscript analysis.
  PYTHONPATH="$ROOT/probes${PYTHONPATH:+:$PYTHONPATH}" python3 - "$run_dir" "$BASELINE_VARIANT" "$BASELINE_PHASE" <<'PYVALIDATE'
import json,sys
from pathlib import Path
from report_baseline_audit import read_collection, require
root=Path(sys.argv[1]);variant,phase=sys.argv[2:]
report=read_collection(root,phase,variant)
require(report["valid"],report.get("error","Invalid collection"))
if phase=="full":
    pilot=read_collection(root.parents[1]/"pilot"/root.name,"pilot",variant)
    require(pilot["valid"] and pilot["identity"]==report["identity"],
            "Full collection differs from its passed pilot")
report["pass"]=True
report["compile_action_infer"]=report["identity"]["runtime"]["compile_action_infer"]
(root/"audit_summary.json").write_text(json.dumps(report,indent=2)+"\n")
if phase!="full":
    (root/"stock_repeat_check.json").write_text(json.dumps(report,indent=2)+"\n")
print(f"Validated {phase}: {report['n_paired_states']} paired states per arm")
PYVALIDATE
}

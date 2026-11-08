# The non-transfer result. The claim being reproduced is a NULL, so repeats matter
# more here than anywhere else: one seed gave -0.3 and another +9.0 on the same
# configuration, and a five-point beta sweep spans 10 points -- this backbone's
# own seed noise.
TITLE="pi0.5 closed loop (replicated null)"
SECTION="07"
# The shipped artefacts, enumerated so a fresh host re-collects the same set
# under the same names rather than three similar runs the analysis would pool
# into a different number. Format:
#   name:episodes:seed:configs   ("?" = the seed was never recorded)
#
# As with the tau=8 set, none of these artefacts carries a seed -- the field did
# not exist when they were written -- so only pi_replicate_seed7 has one that can
# be read off its name. The rest run at 42 and will reproduce the configuration
# but not the episode outcomes. The beta sweep is the decisive part of this
# experiment and all five of its runs are in that position.
RUNS="${RUNS:-\
pi_cmp:10:?:full,c3ache:2,c3ache:2:s,c3ache:4,c3ache:4:s \
pi_cmp_n300:30:?:full,c3ache:2,c3ache:2:s,c3ache:4,c3ache:4:s \
pi_replicate_seed7:30:7:full,c3ache:2,c3ache:2:s \
pi_control_beta03:10:?:full,c3ache:2,c3ache:2:s \
beta_pi05_run_000:20:?:full,c3ache:2:s \
beta_pi05_run_025:20:?:full,c3ache:2:s \
beta_pi05_run_050:20:?:full,c3ache:2:s \
beta_pi05_run_075:20:?:full,c3ache:2:s \
beta_pi05_run_100:20:?:full,c3ache:2:s}"

DEFAULT_SEEDS="$RUNS"
GATE=no
# pi0.5 is served by lerobot in its own virtualenv, not FastWAM. It has no
# fastwam to import, and lerobot pins its own mujoco (3.8.1 at time of writing)
# rather than the 3.3.2 that FastWAM's LIBERO replay data was rendered with --
# that pin is a property of that data, not of this backbone. An empty MUJOCO_PIN
# means "do not check".
STACK=lerobot
MUJOCO_PIN=
ANALYSIS=closedloop

run_one() {   # <name:episodes:seed:configs> <outdir>
  local spec="$1" out="$2" name eps seed cfgs beta shrink
  name=${spec%%:*};  spec=${spec#*:}
  eps=${spec%%:*};   spec=${spec#*:}
  seed=${spec%%:*};  cfgs=${spec#*:}
  if [ "$seed" = "?" ]; then
    seed=42
    echo "  -- $name: seed never recorded for this artefact; running at 42" >&2
  fi

  # The sweep runs differ only in the substitution weight. probes/pi05_libero.py
  # has no flat-weight option -- a ":s" config reads PI_SHRINK and uses one
  # coefficient per cell -- so a constant weight is expressed as a shrink file
  # whose every coefficient is that constant. That is how the shipped sweep was
  # produced, and generating it here is what makes the sweep reproducible rather
  # than merely described.
  shrink="$REPO/inputs/weights/shrink_pi05.json"
  case "$name" in
    beta_pi05_run_*)
      beta=$(echo "${name##*_}" | awk '{printf "%.2f", $1/100}')
      shrink="$out/_shrink_beta_${name##*_}.json"
      python3 - "$REPO/inputs/weights/shrink_pi05.json" "$shrink" "$beta" <<'PYFLAT'
import json, sys
src, dst, b = sys.argv[1], sys.argv[2], float(sys.argv[3])
d = json.load(open(src))
d["coef"] = {k: [b] for k in d["coef"]}
d["note"] = f"flat weight {b} for the sweep; per-cell fit replaced by a constant"
json.dump(d, open(dst, "w"))
PYFLAT
      echo "  -- $name (beta=$beta, flat)" >&2 ;;
    pi_control_beta03)
      shrink="$out/_shrink_beta_030.json"
      python3 - "$REPO/inputs/weights/shrink_pi05.json" "$shrink" 0.3 <<'PYFLAT'
import json, sys
src, dst, b = sys.argv[1], sys.argv[2], float(sys.argv[3])
d = json.load(open(src))
d["coef"] = {k: [b] for k in d["coef"]}
d["note"] = f"flat weight {b}"
json.dump(d, open(dst, "w"))
PYFLAT
      echo "  -- $name (beta=0.3, flat control)" >&2 ;;
    *) echo "  -- $name ($eps episodes, seed $seed)" >&2 ;;
  esac

  cd "$WAM_ROOT" && \
    PI_SUITE=libero_goal PI_TASKS=10 PI_EPISODES="$eps" PI_SEED="$seed" \
    PI_EXEC=10 PI_SKIP_STEPS=8 \
    PI_CONFIGS="$cfgs" \
    PI_SHRINK="$shrink" \
    PI_OUT="$out/${name}.json" python probes/pi05_libero.py
}

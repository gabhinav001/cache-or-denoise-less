# Two follow-ups from the reviews of the workshop paper, on RTX A6000s in a
# machine rented whole (gpu_frac 1), so no other tenant shares its CPUs or GPUs.
# Every arm runs in its own process through budget_shift.sh.
#
#   @timing        the composed cache (K=2, step 0 cached, tau=4) against two
#                  uncached steps, in ROUNDS interleaved rounds whose order
#                  alternates (stock2 first in odd rounds, comp4 first in even
#                  ones), on the first three initial states of every task.
#                  Latency only: success on these states is known from the
#                  100-state runs, and rollouts repeat on an A6000. Nothing else
#                  may run on the machine while these arms do.
#   @jitter        two against ten uncached steps with Optional-IDM under the
#                  1.5 cm camera jitter of followup.sh, on the first 25 initial
#                  states of every task (250 states). Planned before the run: a
#                  five-point non-inferiority margin tested with a Tango 90%
#                  interval; the 50-state run had [-9.2, 9.2]. Its first 5 states
#                  per task repeat the 50-state run exactly.
#   @jitter_orig   the same for the original checkpoint. Not in the defaults:
#                  our budget stopped these arms after 3 and 6 of 10 tasks, so
#                  nothing is shipped for them (--seeds @jitter_orig runs them).
#
# Only @timing measures time, so nothing else may run beside it; run in sequence
# on one GPU, the defaults below do that and collect exactly the shipped arms.
# How those were collected on a machine with eight A6000s (jitter arms side by
# side and split by task range, three of the five planned timing rounds within
# our budget) is in results/budget_shift/idle_a6000/README.md.

. "${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/repro/experiments/budget_shift.sh"

TITLE="Idle-host composed-cache timing, and camera-jitter non-inferiority on 250 states"
SECTION="3"
GATE=yes
ANALYSIS=none

ROUNDS="${ROUNDS:-3}"
CAM="${CAM:-0.015}"
CAMTAG=$(echo "$CAM" | tr -d '.')
TIMING=""
for r in $(seq 1 "$ROUNDS"); do
  s2="it${r}_idm_stock2:idm:2:1.0:3:stock"
  c4="it${r}_idm_comp4:idm:2:1.0:3:whole:4@prefix=1"
  if [ $((r % 2)) = 1 ]; then TIMING="$TIMING $s2 $c4"; else TIMING="$TIMING $c4 $s2"; fi
done
JITTER="\
cam${CAMTAG}n250_idm_stock10:idm:10:1.0:25:stock@perturb=cam:$CAM \
cam${CAMTAG}n250_idm_stock2:idm:2:1.0:25:stock@perturb=cam:$CAM"
JITTER_ORIG="\
cam${CAMTAG}n250_uncond_stock10:uncond:10:5.0:25:stock@perturb=cam:$CAM \
cam${CAMTAG}n250_uncond_stock2:uncond:2:5.0:25:stock@perturb=cam:$CAM"

# In sequence, no arm runs beside another, so the order does not affect timing.
DEFAULT_SEEDS="@timing @jitter"

# Keep budget_shift.sh's run_one under another name, then wrap it so the groups
# above resolve here and every single arm goes through the original.
eval "$(declare -f run_one | sed '1s/^run_one/_budget_shift_run_one/')"
run_one() {   # groups of this file, then anything budget_shift.sh understands
  local spec="$1" out="$2" g=""
  case "$spec" in
    @timing) g="$TIMING" ;; @jitter) g="$JITTER" ;; @jitter_orig) g="$JITTER_ORIG" ;;
  esac
  if [ -n "$g" ]; then
    local s; for s in $g; do run_one "$s" "$out"; done
    return 0
  fi
  _budget_shift_run_one "$spec" "$out"
}

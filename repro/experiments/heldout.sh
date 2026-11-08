# Two follow-ups to the reviews of the workshop paper. The predictions below were
# written, committed and pushed before any of these arms ran; the analysis tests
# them as stated here. Every arm runs in its own process through budget_shift.sh,
# on the first 5 initial states of every LIBERO-Goal task (50 states), Optional-IDM.
#
#   @heldout   a shift outside the sweep of Sec. 4, s = 2, under fixed noise:
#              ten uncached steps, and the tau = 8 cache over the first 7 and
#              the first 9 of ten steps. By Eq. 1, at s = 2 the first 7 steps
#              cover 53.8% of the flow path (fresh computation resumes at
#              sigma = 0.46) and the first 9 cover 81.8% (sigma = 0.18).
#              From the sweep (no cache covering 64% of the path or less lost
#              more than two points against its uncached policy; both caches at
#              80% lost 22-42), we predict:
#                P1  idm_s2_p7_tau8 is within 4 points of idm_s2_stock10
#                P2  idm_s2_p9_tau8 loses at least 20 points to idm_s2_stock10
#   @streams   a second, independent noise stream for the fresh-noise loss at 60%
#              of the path (s = 1, first 6 steps cached, tau = 8): the same cache
#              under fixed and under fresh noise with run seed 7. The paper's runs
#              used seed 42, where it succeeded on 96% under fixed noise and 82%
#              under fresh noise (uncorrected p = 0.039). We predict:
#                P3  under fresh noise it loses at least 5 points to fixed noise
#              and report the exact McNemar test pooled over both seeds
#              (100 paired states).

. "${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/repro/experiments/budget_shift.sh"

TITLE="Held-out shift prediction, and a second noise stream for the fresh-noise loss"
SECTION="4"
GATE=yes
ANALYSIS=none

HELDOUT="\
idm_s2_stock10:idm:10:2.0:5:stock \
idm_s2_p7_tau8:idm:10:2.0:5:whole:8@prefix=7 \
idm_s2_p9_tau8:idm:10:2.0:5:whole:8@prefix=9"
STREAMS="\
idm_s1_p6_tau8_seed7:idm:10:1.0:5:whole:8@prefix=6,seed=7 \
idm_s1_p6_tau8_rn_seed7:idm:10:1.0:5:whole:8@prefix=6,noise,seed=7"

DEFAULT_SEEDS="@heldout @streams"

# Keep budget_shift.sh's run_one under another name, then wrap it so the groups
# above resolve here and every single arm goes through the original.
eval "$(declare -f run_one | sed '1s/^run_one/_budget_shift_run_one/')"
run_one() {   # groups of this file, then anything budget_shift.sh understands
  local spec="$1" out="$2" g=""
  case "$spec" in
    @heldout) g="$HELDOUT" ;; @streams) g="$STREAMS" ;;
  esac
  if [ -n "$g" ]; then
    local s; for s in $g; do run_one "$s" "$out"; done
    return 0
  fi
  _budget_shift_run_one "$spec" "$out"
}

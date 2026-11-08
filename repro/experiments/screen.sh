# Can a cache's closed-loop failure be predicted from uncached rollouts?
#
# The predictions below were written, committed and pushed before any of these
# runs; probes/report_screen.py, committed with them, defines every quantity and
# tests them as stated here. Every arm runs in its own process through
# budget_shift.sh, with run seed 42, on LIBERO-Goal.
#
#   @screen    an uncached rollout of each checkpoint, shift and noise setting
#              below on the first 5 initial states of every task (50 states).
#              At every replanning call the `screen` arm of fw_libero.py also
#              replays what each of ten cache configurations would have returned
#              for the same observation (probes/screen_engine.py): 3 to 9 of ten
#              steps cached at tau = 8, and 8 cached at tau = 2, 4 and 0. The
#              executed action is the uncached one. For each configuration, D is
#              the mean over states of the mean over reuse calls of d, the RMS
#              difference between its action chunk and the uncached one over the
#              ten executed actions (normalised action space).
#   @validate  closed-loop runs of three configurations never run before, on the
#              OTHER five initial states of every task (states 5-9), with their
#              uncached baselines: the original checkpoint at s = 1 under fresh
#              noise with the first 6 and 8 steps cached, and Optional-IDM at
#              s = 2 under fresh noise with the first 7 cached (tau = 8).
#   @smoke     one episode of the screen and of the closed-loop tau = 8 cache,
#              with their actions recorded: the screen's replay at the first
#              reuse call must equal the closed-loop cache's action there
#              (probes/check_screen_smoke.py).
#
# Predictions (S: closed-loop success of the cached configuration on the same
# 50 states, from the shipped runs listed in probes/report_screen.py; share:
# 1 - sigma_m from Eq. 1):
#   P4  at s = 1 with 8 cached steps and tau = 8, D is larger for Optional-IDM
#       than for the original checkpoint (which keeps 72% against 54%)
#   P5  for Optional-IDM at s = 1, D is larger under fresh than under fixed
#       noise at each of 4, 6, 7 and 8 cached steps (tau = 8)
#   P6  for Optional-IDM with 8 cached steps (tau = 8, fixed noise), D falls
#       with the shift: s = 1 > s = 2 > s = 3 > s = 5
#   P7  over the 24 tau = 8 configurations with closed-loop runs on these
#       states, Spearman's rank correlation between D and S is at most -0.7
#       and below that between the cached share and S (-0.74, known from the
#       shipped runs before the screen)
#   P8  for the three @validate configurations, a binomial logistic regression
#       of S on log D, fitted to the 24 configurations of P7, predicts the
#       success observed on states 5-9 with a smaller mean absolute error than
#       the same regression on the cached share
#
# The screen's own rollouts are uncached, so where a shipped uncached run of the
# same policy exists they should repeat its outcomes (reported, not predicted).

. "${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/repro/experiments/budget_shift.sh"

TITLE="Screening cache configurations on uncached rollouts, and a held-out closed-loop test"
SECTION="4"
GATE=yes
ANALYSIS=none

SCREEN="\
scr_idm_s1:idm:10:1.0:5:screen \
scr_idm_s1_rn:idm:10:1.0:5:screen@noise \
scr_idm_s2:idm:10:2.0:5:screen \
scr_idm_s2_rn:idm:10:2.0:5:screen@noise \
scr_idm_s3:idm:10:3.0:5:screen \
scr_idm_s3_rn:idm:10:3.0:5:screen@noise \
scr_idm_s5:idm:10:5.0:5:screen \
scr_idm_s5_rn:idm:10:5.0:5:screen@noise \
scr_uncond_s1:uncond:10:1.0:5:screen \
scr_uncond_s1_rn:uncond:10:1.0:5:screen@noise \
scr_uncond_s5:uncond:10:5.0:5:screen \
scr_uncond_s5_rn:uncond:10:5.0:5:screen@noise"
VALIDATE="\
val_uncond_s1_stock10_rn:uncond:10:1.0:5:stock@noise,first_ep=5 \
val_uncond_s1_p6_tau8_rn:uncond:10:1.0:5:whole:8@prefix=6,noise,first_ep=5 \
val_uncond_s1_p8_tau8_rn:uncond:10:1.0:5:whole:8@noise,first_ep=5 \
val_idm_s2_stock10_rn:idm:10:2.0:5:stock@noise,first_ep=5 \
val_idm_s2_p7_tau8_rn:idm:10:2.0:5:whole:8@prefix=7,noise,first_ep=5"
SMOKE="\
smoke_screen:idm:10:1.0:1:screen@dump,tasks=1 \
smoke_whole8:idm:10:1.0:1:whole:8@dump,tasks=1"

DEFAULT_SEEDS="@smoke @screen @validate"

eval "$(declare -f run_one | sed '1s/^run_one/_budget_shift_run_one/')"
run_one() {   # groups of this file, then anything budget_shift.sh understands
  local spec="$1" out="$2" g=""
  case "$spec" in
    @screen) g="$SCREEN" ;; @validate) g="$VALIDATE" ;; @smoke) g="$SMOKE" ;;
  esac
  if [ -n "$g" ]; then
    local s; for s in $g; do run_one "$s" "$out"; done
    return 0
  fi
  _budget_shift_run_one "$spec" "$out"
}

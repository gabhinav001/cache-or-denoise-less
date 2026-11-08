# Two follow-ups to the round 8 reviews of the workshop paper. The predictions
# below were written, committed and pushed before any of these runs;
# probes/report_latent_delay.py, committed with them, tests them as stated here.
# Every arm runs in its own process through budget_shift.sh, with run seed 42,
# on the first 5 initial states of every LIBERO-Goal task (50 states).
#
#   @latent    Why does fresh noise hurt the cache? Fresh noise changes the
#              latent between calls AND gives each episode a different latent
#              from the run's fixed one. These arms separate the two: the tau = 8
#              cache over the first 6 and 8 of ten steps (60% and 80% of the path,
#              Optional-IDM, s = 1) with each episode's first fresh latent held for
#              the whole episode (episode_noise; the latent equals the first draw
#              of the fresh-noise arms, probes/test_latent_delay.py). The shipped
#              runs give the same caches with the run's fixed latent (96% and 54%
#              on these states) and with fresh noise on every call (82% and 26%).
#              If reuse across calls is what fresh noise breaks, we predict:
#                P9   at each share, the episode-latent cache succeeds at least 8
#                     points more often than the fresh-noise cache
#                P10  at each share, it succeeds within 10 points of the
#                     fixed-latent cache
#   @delay     Does a faster call help a robot that does not wait for it? Uncached
#              two and ten steps executed with a delay of 2 or 5 control steps
#              (50 ms each at 20 Hz): until a call's chunk arrives, the robot keeps
#              executing the previous chunk (fw_libero.delayed_actions). On the
#              Table 1 machine a two-step call takes 94 ms (2 steps) and a ten-step
#              call 202 ms (5 steps). We predict:
#                P11  two steps at their own delay (2) succeed at least as often as
#                     ten steps at theirs (5), on the same 50 states
#              The 2 x 2 design (steps x delay) and the synchronous runs separate
#              the delay from the step count; reported, not predicted.
#   @delaycache  The original checkpoint at s = 5: two steps at delay 2, and its
#              tau = 8 cache with delay 5 after refresh calls (about 210 ms) and
#              2 after reuse calls (about 100 ms). Reported, not predicted.
#   @screenep  The cache screen under episode_noise (Optional-IDM, s = 1), to
#              measure how far stored residuals drift when the latent is held per
#              episode, and to check that every call of an episode starts from one
#              latent. Reported, not predicted.

. "${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/repro/experiments/budget_shift.sh"

TITLE="Episode-latent control for the fresh-noise loss, and delayed execution"
SECTION="4"
GATE=yes
ANALYSIS=none

LATENT="\
ld_idm_s1_p6_tau8_ep:idm:10:1.0:5:whole:8@prefix=6,ep_noise \
ld_idm_s1_p8_tau8_ep:idm:10:1.0:5:whole:8@ep_noise"
DELAY="\
ld_idm_s1_stock2_d2:idm:2:1.0:5:stock@delay=2 \
ld_idm_s1_stock10_d5:idm:10:1.0:5:stock@delay=5 \
ld_idm_s1_stock2_d5:idm:2:1.0:5:stock@delay=5 \
ld_idm_s1_stock10_d2:idm:10:1.0:5:stock@delay=2"
DELAYCACHE="\
ld_uncond_s5_stock2_d2:uncond:2:5.0:5:stock@delay=2 \
ld_uncond_s5_p8_tau8_d52:uncond:10:5.0:5:whole:8@delay=5/2"
SCREENEP="\
ld_scr_idm_s1_ep:idm:10:1.0:5:screen@ep_noise"

DEFAULT_SEEDS="@latent @delay @delaycache @screenep"

eval "$(declare -f run_one | sed '1s/^run_one/_budget_shift_run_one/')"
run_one() {   # groups of this file, then anything budget_shift.sh understands
  local spec="$1" out="$2" g=""
  case "$spec" in
    @latent) g="$LATENT" ;; @delay) g="$DELAY" ;; @delaycache) g="$DELAYCACHE" ;; @screenep) g="$SCREENEP" ;;
  esac
  if [ -n "$g" ]; then
    local s; for s in $g; do run_one "$s" "$out"; done
    return 0
  fi
  _budget_shift_run_one "$spec" "$out"
}

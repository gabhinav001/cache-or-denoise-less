# A follow-up to the round 9 reviews of the workshop paper, committed and pushed
# with its analysis (probes/report_latent_delay.py, analyze_graphs) before any of
# these runs. It states no prediction: the outcome bounds a claim of the paper
# either way, and the analysis below is fixed in advance.
#
# latent_delay.sh executed each call's chunk late by the call's time under
# PyTorch's default compile mode, rounded up to 50 ms control steps: 94 ms for
# two steps (delay 2) and 202 ms for ten (delay 5). Under upstream CUDA graphs
# the same calls take 86 and 143 ms on the Table 1 machine (original checkpoint;
# both checkpoints run the same network), so a deployment with graphs would
# delay two steps by 2 control steps and ten steps by 3. These arms add the
# missing delays, each in its own process through budget_shift.sh, with run seed
# 42 on the first 5 initial states of every LIBERO-Goal task (50 states):
#
#   @graphs  ten uncached steps at delay 3, Optional-IDM at s = 1 and the original
#            checkpoint at s = 5; then ten steps at delay 4 (Optional-IDM), the
#            missing point of the delay curve
#
# Reported, for each checkpoint, as paired differences on the 50 states with
# Tango 90% intervals and exact McNemar tests (uncorrected):
#   - two steps at delay 2 (latent_delay_a6000) minus ten steps at delay 3
#   - ten steps at delay 3 minus ten steps without delay (shift_a6000)
#   - Optional-IDM: ten steps at delay 3 and 4 minus ten steps at delay 2

. "${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/repro/experiments/budget_shift.sh"

TITLE="Delays of CUDA-graph execution for the delayed-execution comparison"
SECTION="3"
GATE=yes
ANALYSIS=none

GRAPHS="\
ld_idm_s1_stock10_d3:idm:10:1.0:5:stock@delay=3 \
ld_uncond_s5_stock10_d3:uncond:10:5.0:5:stock@delay=3 \
ld_idm_s1_stock10_d4:idm:10:1.0:5:stock@delay=4"

DEFAULT_SEEDS="@graphs"

eval "$(declare -f run_one | sed '1s/^run_one/_budget_shift_run_one/')"
run_one() {   # the group of this file, then anything budget_shift.sh understands
  local spec="$1" out="$2" s
  if [ "$spec" = "@graphs" ]; then
    for s in $GRAPHS; do run_one "$s" "$out"; done
    return 0
  fi
  _budget_shift_run_one "$spec" "$out"
}

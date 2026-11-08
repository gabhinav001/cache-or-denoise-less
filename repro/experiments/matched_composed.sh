# Matched 1-step, 2-step uncached and composed 2-step caching on one host
#
# Arms:
# 1. Optional-IDM, 1-step uncached (100 states)
# 2. Optional-IDM, 2-step uncached (100 states)
# 3. Native/Original, 1-step uncached (100 states)
# 4. Native/Original, 2-step uncached (100 states)
# 5. Optional-IDM, K=2, tau=2, prefix=1 (100 states)
# 6. Native/Original, K=2, tau=2, prefix=1 (100 states)
# 7. Optional-IDM, K=2, tau=4, prefix=1 (100 states)
# 8. Native/Original, K=2, tau=4, prefix=1 (100 states)

. "${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/repro/experiments/budget_shift.sh"

TITLE="Matched 1-step, 2-step uncached and composed 2-step caching on one host"
SECTION="3"
GATE=yes
ANALYSIS=none

MATCHED_ARMS="\
idm_stock_1step:idm:1:1.0:10:stock \
idm_stock_2step:idm:2:1.0:10:stock \
uncond_stock_1step:uncond:1:5.0:10:stock \
uncond_stock_2step:uncond:2:5.0:10:stock \
idm_whole_2step_p1_tau2:idm:2:1.0:10:whole:2@prefix=1 \
uncond_whole_2step_p1_tau2:uncond:2:5.0:10:whole:2@prefix=1 \
idm_whole_2step_p1_tau4:idm:2:1.0:10:whole:4@prefix=1 \
uncond_whole_2step_p1_tau4:uncond:2:5.0:10:whole:4@prefix=1"

RUNS="${RUNS:-$MATCHED_ARMS}"
DEFAULT_SEEDS="$RUNS"

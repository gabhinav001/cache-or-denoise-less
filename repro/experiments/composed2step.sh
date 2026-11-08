# Composed 2-step schedule with step 0 cached (prefix=1)
# Testing residual reuse on top of a reduced-step schedule.
#
# Arm 1: Optional-IDM (s=1.0), K=2, tau=2, prefix=1 (10 tasks x 10 episodes = 100 states)
# Arm 2: Original/Native (s=5.0), K=2, tau=2, prefix=1 (10 tasks x 10 episodes = 100 states)
# Arm 3: Optional-IDM (s=1.0), K=2, tau=4, prefix=1 (10 tasks x 10 episodes = 100 states)
# Arm 4: Original/Native (s=5.0), K=2, tau=4, prefix=1 (10 tasks x 10 episodes = 100 states)

. "${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/repro/experiments/budget_shift.sh"

TITLE="Composed 2-step schedule with step 0 cached (residual reuse on few-step model)"
SECTION="3"
GATE=yes
ANALYSIS=probes/report_budget_shift.py

ARMS_TAU2="\
idm_whole_2step_p1_tau2:idm:2:1.0:10:whole:2@prefix=1 \
uncond_whole_2step_p1_tau2:uncond:2:5.0:10:whole:2@prefix=1"

ARMS_TAU4="\
idm_whole_2step_p1_tau4:idm:2:1.0:10:whole:4@prefix=1 \
uncond_whole_2step_p1_tau4:uncond:2:5.0:10:whole:4@prefix=1"

RUNS="${RUNS:-$ARMS_TAU2 $ARMS_TAU4}"
DEFAULT_SEEDS="$RUNS"

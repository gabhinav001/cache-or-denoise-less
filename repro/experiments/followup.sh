# Follow-up collection for the workshop paper's review round, on one RTX A6000.
#
# Four questions, each arm in its own fresh process through budget_shift.sh:
#
#   @share    where between 64% and 80% of the flow path a tau=8 cache stops
#             being safe (7 cached steps at s=1 covers 70%), and whether a third
#             shift falls on the same curve: at s=3 the default rule (8 cached
#             steps, two fresh) covers 57% and 9 cached steps covers 75%
#   @timing1..3  the composed two-step cache timed against one- and two-step
#             uncached calls on the same machine, in three interleaved rounds so
#             drift in the host cannot favour one arm (latency only; the success
#             of these configurations is in results/budget_shift/frontier100_a6000
#             and results/matched_composed, and it is deterministic)
#   @pilot    ten episodes of uncached ten steps under camera jitter of three
#             sizes, to choose a shift that lowers success without flattening it
#             (generalize.sh found 0.005 m nearly harmless and 0.02 m destructive)
#   @hard     two against ten steps, one step and the tau=8 cache under the chosen
#             jitter (CAM, metres; the pilot chose 0.015), where the benchmark is
#             no longer at its ceiling
#   @noise70  fresh noise per call at the 70% (s=1) and 75% (s=3) shares
#
# The defaults regenerate exactly the files in results/budget_shift/followup_a6000/.
# Stage 2 was cut to fit the budget: round 3 of the timing ran only the three
# arms whose ratios the paper reports, and fresh noise ran at 70% and 75% only.

. "${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/repro/experiments/budget_shift.sh"

TITLE="Follow-up: share threshold, third shift, interleaved composed timing, camera jitter"
SECTION="3-4"
GATE=yes
ANALYSIS=none

SHARE="\
idm_s1_p7_tau8:idm:10:1.0:5:whole:8@prefix=7 \
idm_s3_stock10:idm:10:3.0:5:stock \
idm_s3_p8_tau8:idm:10:3.0:5:whole:8@prefix=8 \
idm_s3_p9_tau8:idm:10:3.0:5:whole:8@prefix=9 \
uncond_s1_p7_tau8:uncond:10:1.0:5:whole:8@prefix=7"

TIMING1="\
tm1_idm_stock1:idm:1:1.0:3:stock \
tm1_idm_stock2:idm:2:1.0:3:stock \
tm1_idm_comp4:idm:2:1.0:3:whole:4@prefix=1 \
tm1_idm_stock10:idm:10:1.0:3:stock \
tm1_idm_tau8:idm:10:1.0:3:whole:8"
TIMING2="\
tm2_idm_comp4:idm:2:1.0:3:whole:4@prefix=1 \
tm2_idm_stock2:idm:2:1.0:3:stock \
tm2_idm_stock1:idm:1:1.0:3:stock"
TIMING3="\
tm3_idm_stock2:idm:2:1.0:3:stock \
tm3_idm_stock1:idm:1:1.0:3:stock \
tm3_idm_comp4:idm:2:1.0:3:whole:4@prefix=1"

PILOT="\
campilot_0075_idm_stock10:idm:10:1.0:1:stock@perturb=cam:0.0075 \
campilot_010_idm_stock10:idm:10:1.0:1:stock@perturb=cam:0.01 \
campilot_015_idm_stock10:idm:10:1.0:1:stock@perturb=cam:0.015"

CAM="${CAM:-0.015}"
CAMTAG=$(echo "$CAM" | tr -d '.' )
HARD="\
cam${CAMTAG}_idm_stock10:idm:10:1.0:5:stock@perturb=cam:$CAM \
cam${CAMTAG}_idm_stock2:idm:2:1.0:5:stock@perturb=cam:$CAM \
cam${CAMTAG}_idm_stock1:idm:1:1.0:5:stock@perturb=cam:$CAM \
cam${CAMTAG}_idm_tau8:idm:10:1.0:5:whole:8@perturb=cam:$CAM \
cam${CAMTAG}_uncond_stock10:uncond:10:5.0:5:stock@perturb=cam:$CAM \
cam${CAMTAG}_uncond_stock2:uncond:2:5.0:5:stock@perturb=cam:$CAM"

NOISE70="\
idm_s1_p7_tau8_rn:idm:10:1.0:5:whole:8@prefix=7,noise \
idm_s3_p9_tau8_rn:idm:10:3.0:5:whole:8@prefix=9,noise"

# Stage 1 leads with a short timing round and the pilot, so a broken code path
# shows within minutes; stage 2 is launched once the pilot has chosen CAM.
STAGE1="$TIMING1 $PILOT $SHARE $TIMING2"
STAGE2="$HARD $TIMING3 $NOISE70"

DEFAULT_SEEDS="@stage1 @stage2"

# Keep budget_shift.sh's run_one under another name, then wrap it so the groups
# above resolve here and every single arm goes through the original.
eval "$(declare -f run_one | sed '1s/^run_one/_budget_shift_run_one/')"
run_one() {   # groups of this file, then anything budget_shift.sh understands
  local spec="$1" out="$2" g=""
  case "$spec" in
    @share) g="$SHARE" ;; @timing1) g="$TIMING1" ;; @timing2) g="$TIMING2" ;;
    @timing3) g="$TIMING3" ;; @pilot) g="$PILOT" ;; @hard) g="$HARD" ;;
    @noise70) g="$NOISE70" ;; @stage1) g="$STAGE1" ;; @stage2) g="$STAGE2" ;;
  esac
  if [ -n "$g" ]; then
    local s; for s in $g; do run_one "$s" "$out"; done
    return 0
  fi
  _budget_shift_run_one "$spec" "$out"
}

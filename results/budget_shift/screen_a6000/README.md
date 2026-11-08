# Cache screen on uncached rollouts and a held-out closed-loop test (`repro/experiments/screen.sh`)

The predictions these runs test (P4-P8 in the experiment file) and the analysis
that tests them (`probes/report_screen.py`) were committed and pushed as
`f439f32` at 2026-10-02 23:04 UTC, before any machine was rented. Every run
records `commit` `f439f32bc2ec+src.1e69d71ed2372dc8`; `../SOURCE_MANIFEST_screen.json`
hashes the files shipped to the host (`git archive` of that commit), which
matched it on the host before any run started.

One machine with four RTX A6000s and a 48-thread Xeon Gold 6248R (shared host,
prepared by `setup/onstart_libero_shrink.sh`) ran four copies of
`repro/run_experiment.sh screen`, one per GPU, from 23:23 UTC. Each copy ran the
identity gate first; all four passed with identical numbers (whole-stack replay
error 0.122%), and `gate.json` is GPU 0's. `phases.txt` and `host_check.txt`
record when each copy started and the GPUs' state before and after.

**What the screen does.** The `screen` arm of `probes/fw_libero.py` runs the
uncached policy. After every call, `probes/screen_engine.py` replays that call's
Euler loop from the same initial latent under ten cache configurations (3 to 9
of 10 steps cached at tau = 8, and 8 cached at tau = 2, 4 and 0), each with the
whole-stack residuals it would hold: those recorded on its own last refresh call
of the same rollout. Nothing replayed is executed. Per configuration and reuse
call it records `d` (RMS difference between the replayed and the uncached action
chunk over the ten executed actions, normalised action space), `grip` (executed
actions whose binarised gripper command differs) and `e` (relative difference of
the latent where fresh computation resumes); `drift8` records, per step, how far
the tau = 8 store is from the current call's residual.

**Checks.**

- `smoke_screen.json`, `smoke_whole8.json` (`@smoke`): one episode of the screen
  and of the closed-loop tau = 8 cache. Their first calls agree exactly, and the
  screen's replay at the first reuse call equals the closed-loop cache's action
  there exactly (`SMOKE.txt`, `probes/check_screen_smoke.py`; also
  `screen.smoke` in the snapshot). That is the exactness the screen has: the
  cache's computation for the same observation, initial latent and stored
  residuals. From the next call on, the closed-loop cache has executed its own
  actions while the replays follow the uncached rollout, and the two differ
  (by 0.013 at the next call and up to 0.95 later, `screen.smoke.later_max_abs_diff`).
  The screen therefore measures a cache's local error along uncached states, not
  the trajectory its own actions would produce.
- The screen's own rollouts are uncached. In the eight cells with a shipped
  uncached run of the same policy on the same states, they repeat all 400
  outcomes (`screen.cells.*.vs_shipped_uncached`).

**Results** (`summary.json`, section `screen`; Sec. 4 and the supplement):

| prediction | result | met |
|---|---|---|
| P4: at s = 1, 8 cached steps, D larger for Optional-IDM than for the original | 0.401 vs 0.378 | yes, narrowly |
| P5: fresh noise raises D at 4, 6, 7, 8 cached steps (Optional-IDM, s = 1) | 0.050 vs 0.028, 0.231 vs 0.190, 0.355 vs 0.344, 0.516 vs 0.401 | yes |
| P6: D at 8 cached steps falls with the shift, s = 1 > 2 > 3 > 5 | 0.401, 0.287, 0.121, 0.029 | yes |
| P7: over the 24 tau = 8 configurations, rho(D, success) <= -0.7 and below rho(share, success) | -0.762 vs -0.738 (bootstrap 90% of the difference [-0.09, 0.04]) | yes, by a margin 50 states do not resolve |
| P8: on the three held-out configurations, a logistic fit on log D predicts success with a smaller mean absolute error than one on the share | 16.4 vs 16.7 points | yes, by 0.3 points |

The held-out configurations (`val_*.json`, initial states 5-9 of every task):
the original checkpoint at s = 1 under fresh noise succeeded on 92% and 20% with
6 and 8 cached steps (96% uncached), and Optional-IDM at s = 2 under fresh noise
on 88% with 7 (98% uncached). Both fits predicted the first and the last within
six points and put the 8-step configuration near 61-62%, where it reached 20%.

Descriptive, chosen after the runs and not among the predictions: every
configuration with D below 0.13 kept at least 90% success (16 of them) and every
one above 0.37 at most 72% (6); under fixed noise at s = 1 with 8 cached steps,
27-28% of reuse calls inverted the gripper command on all ten executed actions.

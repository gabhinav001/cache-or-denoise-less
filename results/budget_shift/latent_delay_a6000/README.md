# Per-episode latent and delayed execution (`repro/experiments/latent_delay.sh`)

The predictions these runs test (P9-P11 in the experiment file) and the analysis
that tests them (`probes/report_latent_delay.py`) were committed and pushed as
`3765e42` at 2026-10-03 05:45 UTC, before any machine was rented. Every run
records `commit` `3765e42fe9f1+src.978d3877b4ebf446`; `../SOURCE_MANIFEST_latent_delay.json`
hashes the files shipped to the host (`git archive` of that commit), which
matched it on the host before any run started.

One RTX A6000 on a shared machine (28 threads of a Xeon Gold 6248R; other
tenants kept the load average near 14 before any arm), prepared by
`setup/onstart_libero_shrink.sh`, ran `repro/run_experiment.sh latent_delay`
from 06:04 to 09:27 UTC: the identity gate first (`gate.json`; whole-stack
replay error 0.122%, as on every earlier machine), then the arms in the order of
their priority (`phases.txt`, `host_check.txt`). A first rental on another host
failed before any arm ran: its driver (CUDA 12.5) could not run the image's
CUDA 12.8 build, and it was destroyed after its GPU check. Host load changes
only wall time here: the simulator advances synchronously, and the delays below
are counted in control steps, not measured.

**What the arms do.** All arms run on the first five initial states of every
LIBERO-Goal task (50 states), run seed 42, each in its own process.

- *Per-episode latent* (`episode_noise`): Fast-WAM is given a seed per episode,
  the one with which the fresh-noise arms reseed their global generator, so
  every call of an episode starts from that episode's latent. Fast-WAM draws the
  action latent from a CPU generator as the first random draw of a call
  (`FastWAM.infer_action`), so this latent equals the first latent of the same
  episode under fresh noise (`probes/test_latent_delay.py` checks the generator
  equivalence). Each row records a digest of its episode's latent
  (`episode_latent`). The tau = 8 cache over the first 6 and 8 of 10 steps,
  Optional-IDM at s = 1, is compared with the shipped runs of the same caches
  with the run's single fixed latent (`../shift_a6000/`) and with fresh noise on
  every call (`../noise_a6000/`).
- *Delayed execution* (`delay`): while a call runs, the robot keeps executing
  the previous chunk. With a delay of d control steps (50 ms each at 20 Hz), it
  executes the previous chunk's actions planned for the next d steps, then the
  new chunk's actions d to 9; before an episode's first chunk arrives it holds
  still (`fw_libero.delayed_actions`; `probes/test_latent_delay.py` checks that
  every executed action is the newest plan available for its time step). On the
  Table 1 machine a two-step call takes 94 ms (delay 2) and a ten-step call
  202 ms (delay 5). The cache arm uses delay 5 after refresh calls (about
  210 ms) and 2 after reuse calls (about 100 ms).
- *Screen under the per-episode latent*: the cache screen of `../screen_a6000/`
  with `episode_noise`, recording the latent digest of every call.

**Results** (`summary.json`, section `latent_delay`; Secs. 3 and 4, Table A7):

| prediction | result | met |
|---|---|---|
| P9: at each share, the per-episode latent succeeds at least 8 points more often than fresh noise | 60%: 96 vs 82% (+14); 80%: 64 vs 26% (+38) | yes |
| P10: at each share, it succeeds within 10 points of the fixed latent | 60%: 96 vs 96% (0); 80%: 64 vs 54% (+10) | yes, at 80% only at its boundary |
| P11: two steps at delay 2 succeed at least as often as ten steps at delay 5 | 48/50 vs 43/50 (Tango 90% interval [1.6, 20.0], exact McNemar p = 0.125) | yes |

Reported, not predicted (Optional-IDM at s = 1 unless noted; successes of 50;
synchronous runs from `../shift_a6000/`):

| | no delay | delay 2 | delay 5 |
|---|---|---|---|
| two steps | 47/50 | 48/50 | 41/50 |
| ten steps | 48/50 | 48/50 | 43/50 |

At equal delay the two schedules differ by at most four points; delay 5 costs
both 10-14 points against delay 2 (two steps: p = 0.039; ten steps:
p = 0.062), and 10-12 against no delay. Shorter delays of CUDA-graph execution
are in `../delay_graphs_a6000/`. The original checkpoint at s = 5: two steps at delay 2
47/50 (synchronous 47/50); its tau = 8 cache with delay
5/2 44/50 (synchronous 46/50).

The screen under the per-episode latent: every call of every episode started
from that episode's latent (50 distinct latents in 50 episodes, each call's
digest equal to its episode's), the uncached rollouts succeeded on 48/50,
the tau = 8 store differed from the current call's residual by
65% over steps 0-7 (63% under the fixed latent and 85% under fresh
noise for the same policy, `../screen_a6000/`), and D at 8 cached steps was
0.40 (0.40 fixed, 0.52 fresh).

Two analysis changes came after the runs and do not touch the tests of P9-P11:
`probes/report_budget_shift.py` now includes the per-episode latent and the
delay in its configuration key, so the cross-machine tallies never pair these
runs with synchronous or fixed-latent runs, and `probes/report_latent_delay.py`
also reports D for the per-episode screen. `../SOURCE_MANIFEST_latent_delay.json`
lists them.

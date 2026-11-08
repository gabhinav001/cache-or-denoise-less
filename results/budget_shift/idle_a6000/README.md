# Idle-host follow-up (`repro/experiments/idle_host.sh`)

On 2026-09-28 we rented a whole machine with eight RTX A6000s and a 48-thread
Xeon Gold 6248R (hashed host tag `6ee7b337`), so that no other tenant shared its
CPUs or GPUs, and prepared it with `setup/onstart_libero_shrink.sh` (30 minutes).
Every run records `commit` `68e61c9f8032+src.9bcc297a5efb8638`; the hashes of the
shipped files are in `../SOURCE_MANIFEST_idle.json`. The identity gate ran five
times, on GPUs 0-3 before the jitter arms and on GPU 0 before the timing rounds
(`gate.json`), and passed each time with the same replay error as on earlier
machines (0.122% for the whole stack).

**Camera jitter on 250 states** (`cam0015n250_idm_stock{2,10}.json`). Two
against ten uncached steps for Optional-IDM under the 1.5 cm camera jitter of
`followup.sh`, on the first 25 initial states of every task. The four jitter
arms of `idle_host.sh` started side by side, one per GPU. To finish inside our
compute budget, the two Optional-IDM arms were then split by task range across
the four idle GPUs: shards ran the same probe invocation as `budget_shift.sh`
with `+PROBE.first_task=A +PROBE.n_tasks=B` added (tasks 4-5, 6-7 and 8-9 for
ten steps, 7-9 for two), and each original process was stopped once its JSON
held every task below its shards (tasks 0-3 and 0-6). `fw_libero.py` writes its
JSON after every task and seeds every episode and every camera perturbation on
(task, episode) alone, so the parts are the rows one process would have written;
each merged file lists its parts, their tasks and their provenance under
`meta.parts`. The first five states of every task repeat the 50-state runs of
`../followup_a6000/` (another machine) in all 100 paired outcomes. The two
arms for the original checkpoint were stopped after 3 and 6 of 10 tasks, when
the budget ran out; incomplete runs are not analysed and are not shipped.

**Composed-cache timing** (`it{1,2,3}_idm_{stock2,comp4}.json`, `gate.json`).
After the jitter arms, the interleaved rounds ran alone on one GPU: the composed
cache (K = 2, step 0 cached, tau = 4) against two uncached steps, on the first
three states of every task, stock first in odd rounds. The run stopped after
three of the five planned rounds, because a fourth would have ended after our
budget did; `idle_host.sh` now defaults to the three rounds and the Optional-IDM
jitter arms, so `repro/run_experiment.sh idle_host` regenerates exactly these
arms.

Supporting logs, not read by the analysis:

- `host_check.txt`: CPU model and every GPU's utilisation, memory and compute
  processes before any arm and again before the timing rounds (all eight GPUs
  at 0% and 1 MiB, no compute processes).
- `host_trace.txt`: every 30 s, UTC, the 1/5/15-minute load average and
  cumulative busy and idle CPU jiffies of the whole machine (`/proc`). During
  the timing rounds the CPUs were 2.5-3.5% busy, one of 48 threads, except while
  an arm loaded its model.
- `phases.txt`: when each arm, shard and phase started and stopped.

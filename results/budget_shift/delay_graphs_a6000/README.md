# Delays of CUDA-graph execution (`repro/experiments/delay_graphs.sh`)

The design and analysis of these runs (`probes/report_latent_delay.py`,
`analyze_graphs`; summary section `delay_graphs`) were committed and pushed as
`e8edaaf` at 2026-10-03 18:39 UTC, before any machine was rented. They state no
prediction. Every run records `commit` `e8edaafef610+src.6dabd7adf38b50e4`;
`../SOURCE_MANIFEST_delay_graphs.json` hashes the files shipped to the host
(`git archive` of that commit), which matched it on the host before any run.

One RTX A6000 on a shared machine (64 threads of a Xeon E5-2683 v4; other
tenants kept the load average near 40 before any arm), prepared by
`setup/onstart_libero_shrink.sh`, ran `repro/run_experiment.sh delay_graphs`
from 19:20 to 20:44 UTC: the identity gate first (`gate.json`; whole-stack
replay error 0.122%, as on every earlier machine), then the three arms
(`phases.txt`, `host_check.txt`). Two earlier rentals never ran anything: one
offer was taken before the instance was created, and on one host our tooling
mistook a package name in the image build (`liberror-perl`) for an error and
destroyed it while it was still setting up.

**Why.** `../latent_delay_a6000/` delayed each call's chunk by the call's time
under PyTorch's default compile mode, rounded up to 50 ms control steps: two
steps 94 ms (delay 2), ten steps 202 ms (delay 5). Under upstream CUDA graphs
the same calls take 86 and 143 ms on the Table 1 machine, delays of 2 and 3.
These arms add ten steps at delay 3 for both checkpoints and at delay 4 for
Optional-IDM, on the same 50 states (run seed 42), each in its own process.

**Results** (successes of 50; `summary.json`, section `delay_graphs`; Table A7):

| | no delay | delay 2 | delay 3 | delay 4 | delay 5 |
|---|---|---|---|---|---|
| Optional-IDM, ten steps | 48 | 48 | **45** | **47** | 43 |
| original, ten steps | 47 | | **47** | | |

Bold: these runs; the rest are from `../latent_delay_a6000/` and `../shift_a6000/`.

- Two steps at delay 2 against ten steps at delay 3: Optional-IDM +6 points, Tango 90% interval [0.6, 14.1], exact p = 0.25;
  original 0 points, Tango 90% interval [-6.8, 6.8], exact p = 1.00.
- Ten steps at delay 3 against no delay: Optional-IDM -6 points, Tango 90% interval [-16.0, 3.2], exact p = 0.45;
  original 0 points, Tango 90% interval [-8.1, 8.1], exact p = 1.00.
- Optional-IDM, ten steps at delay 4 against delay 2: -2 points, Tango 90% interval [-9.7, 5.1], exact p = 1.00.

On these 50 states, delays of 2-4 control steps changed success by at most six
points with no difference resolved, while delay 5 cost both schedules 10-12
points against no delay (`../latent_delay_a6000/`). Under the timings of CUDA
graphs, the faster two-step call showed no resolved success advantage.

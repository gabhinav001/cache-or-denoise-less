# Independent-host check of the regeneration scripts (smoke tier)

On 2026-09-28 a rented RTX A6000 that none of the shipped runs used (hashed host
tag `a79c23a6`) was prepared with `setup/onstart_libero_shrink.sh` alone (8 minutes),
given a copy of this repository, and ran

    bash repro/regenerate_paper.sh smoke

The identity gate and its first compilation took 23 minutes; the eleven arms, one
initial state per task of eleven of the paper's configurations, took 75 minutes.

- `smoke/`: the runs and the gate, as the host wrote them. Their `commit` field is
  `09b7769b4d97+src.b8a0c9728de39267`: the repository revision plus a hash of the
  working tree that was shipped (probes/, repro/, setup/, inputs/).
- `REPORT_host.txt`: the report the host wrote.
- `REPORT.txt`, `compare.json`: the same runs re-analysed with the scripts as
  shipped here (`bash repro/regenerate_paper.sh --analyse-only --out <copy>`).

Outcome: the 110 regenerated episodes, paired with every shipped run of the same
configuration and initial state (39 runs, up to eight machines per configuration,
390 pairs), ended the same way in all 390 pairs, and after the same number of steps
in 387. The gate replayed the cache with the same error as
on every earlier machine (0.122%), and read the model facts the paper states: a
5.00B video expert and a 1.02B action expert, 30 action blocks, a 32-action chunk,
replanning every 10 steps, K = 10.

After this run, three files changed in ways that do not touch a rollout. The gate's
`whole_cache_bytes` counted every step the full-compute pass records (10) rather
than the 8 it caches, so the host's report prints 0.62 MiB; the shape it recorded
gives the cached size, 8 x [1, 32, 1024] x 2 bytes = 0.5 MiB, which `REPORT.txt`
shows. `probes/compare_regen.py` now records run paths relative to the collection,
and `repro/regenerate_paper.sh` derives the cache size as above.

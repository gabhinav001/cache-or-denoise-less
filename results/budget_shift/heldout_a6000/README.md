# Held-out shift and a second noise stream (`repro/experiments/heldout.sh`)

The predictions these runs test (P1-P3 in the experiment file) were committed and
pushed as `746ce6b` at 2026-10-02 17:21 UTC, before any machine was rented. Every
run records `commit` `746ce6b96be0+src.2329bbcd595d92ab`; `../SOURCE_MANIFEST_heldout.json`
hashes the files shipped to the hosts, which match that commit with no uncommitted
changes (checked on each host).

Both hosts were single RTX A6000s on machines shared with other tenants. These
runs measure success only; their latencies are not used anywhere.

- Host `16af74ff` (Xeon Gold 6248R) ran all five arms and the identity gate
  (`gate.json`, whole-stack replay error 0.122%), starting at 17:44 UTC.
- Only the first six tasks of `idm_s1_p6_tau8_rn_seed7` reached us from that host:
  our local disk was full during the final transfer, and the instance was
  released before a complete copy arrived. That partial copy is kept in
  `first_copy/`; the analysis skips incomplete runs.
- Host `5a2738e7` (AMD EPYC 7402P), shipped by `git archive` of the same commit,
  re-ran that arm in full. On the 30 episodes the two copies share, outcomes and
  episode lengths are identical.

Results, tested exactly as stated in the experiment file (`summary.json`,
section `heldout`). Each prediction is a statement about the observed
difference on these 50 states, and that is what "met" refers to:

| prediction | runs | result | met (observed) |
|---|---|---|---|
| P1: caching the first 7 of 10 steps at s = 2 (53.8% of the path) stays within 4 points | `idm_s2_p7_tau8` vs `idm_s2_stock10` | 94% vs 96%, -2 points (Tango 90% [-9.7, 5.1]) | yes |
| P2: caching the first 9 (81.8%) loses at least 20 points | `idm_s2_p9_tau8` vs `idm_s2_stock10` | 70% vs 96%, -26 points (Tango 90% [-38.5, -13.7]; 15 vs 2 discordant, McNemar p = 0.002) | yes |
| P3: with run seed 7, fresh noise loses at least 5 points to fixed noise at 60% of the path | `idm_s1_p6_tau8_rn_seed7` vs `idm_s1_p6_tau8_seed7` | 76% vs 90%, -14 points (Tango 90% [-25.4, -3.5], p = 0.065) | yes |

The intervals show what 50 states can establish. P2's and P3's exclude zero, so
those losses are real at the 90% level, but none of the three intervals lies
entirely within its predicted bound: P1's reaches -9.7 points, P2's -13.7 and
P3's -3.5.

Pooled with the seed-42 runs of `../shift_a6000/` and `../noise_a6000/` (96% fixed,
82% fresh), the 60% cache loses 14 points under fresh noise on 100 paired states
(17 vs 3 discordant, exact McNemar p = 0.003). Both seeds use the same 50 initial
states, so the state-seed pairs are not independent; with the initial state as
the unit (an exact sign-flip test on each state's summed difference over the two
seeds, 18 states changed), p = 0.006.

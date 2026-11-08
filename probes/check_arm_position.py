"""Do two runs of the same configuration agree, when the ARM LIST differs?

They did not. An arm's results depended on how many arms ran before it, because
some global random state advanced continuously through a run and was never reset
between episodes. Measured on seed 42, libero_goal:

    arm                   position   same outcome   same steps
    full                    1 vs 1      100/100        100/100
    c3ache_tau4             4 vs 2       79/100         15/100
    c3ache_tau4_shrink      5 vs 3       93/100         11/100

The arm that ran first in both runs was bit-identical; the others diverged in
proportion to how far their position shifted. That is a reproducibility hazard:
the same nominal configuration reports different numbers depending on what else
was measured alongside it.

An earlier account of this blamed the denoiser's noise. That was wrong, and
check_seed_path.py now proves it: with seed= set, the initial action latent is
drawn from a generator built fresh from cfg.seed inside every call, so it is one
constant tensor for the whole run -- already identical across arms, and immune to
any global reseed. The carrier was some other consumer of the global torch
stream, and which one has not been isolated.

fw_libero.py now reseeds that global stream per (task, episode), which makes
every torch-global draw a function of the episode alone, so arm position cannot
matter. This script checks any two artefacts for the symptom.

usage: check_arm_position.py <a.json> <b.json>
"""
from __future__ import annotations

import json, sys


def index(d):
    return {(r["config"], r["task"], r["episode"]): (r["success"], r["steps"])
            for r in d["rows"]}


def main(pa, pb):
    A, B = json.load(open(pa)), json.load(open(pb))
    a, b = index(A), index(B)
    ca = [c for c in A["meta"].get("configs", sorted({k[0] for k in a}))]
    cb = [c for c in B["meta"].get("configs", sorted({k[0] for k in b}))]
    print(f"\n  {pa.split('/')[-1]}: {len(ca)} arms {ca}")
    print(f"  {pb.split('/')[-1]}: {len(cb)} arms {cb}\n")
    print(f"  {'arm':24s} {'position':>10s} {'shared':>7s} {'same outcome':>13s} {'same steps':>12s}")
    print("  " + "-" * 72)
    worst = 1.0
    for cfg in cb:
        shared = [k for k in a if k[0] == cfg and k in b]
        if not shared: continue
        so = sum(1 for k in shared if a[k][0] == b[k][0])
        ss = sum(1 for k in shared if a[k][1] == b[k][1])
        pos = f"{ca.index(cfg)+1} vs {cb.index(cfg)+1}" if cfg in ca else "-"
        print(f"  {cfg:24s} {pos:>10s} {len(shared):7d} {so:9d}/{len(shared):<3d} "
              f"{ss:8d}/{len(shared):<3d}")
        worst = min(worst, so / len(shared))
    if worst < 1.0:
        print(f"\n  POSITION-DEPENDENT: the worst arm agrees on only {100*worst:.0f}% of episodes.")
        print( "  Both artefacts must come from a build that seeds per (task, episode);")
        print( "  without it, an arm's numbers depend on what ran beside it.")
    else:
        print("\n  Every shared arm agrees on every episode: position-independent.")


if __name__ == "__main__":
    if len(sys.argv) != 3: sys.exit("usage: check_arm_position.py <a.json> <b.json>")
    main(sys.argv[1], sys.argv[2])

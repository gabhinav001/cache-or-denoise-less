"""Emit substitution coefficients indexed by STALENESS, not just per cell.

The shipped weights are fit against the residual one cycle back and then applied
unchanged however stale the cache is. At tau=2 that is exactly right -- the cache
is only ever one cycle old. At tau=4 it is not: the cache is used at staleness 1,
2 and 3, where the optimal weights differ sharply (0.758 / 0.531 / 0.387 on
Fast-WAM). Applying the staleness-1 weight at staleness 3 under-attenuates badly.

This fits one coefficient per (cell, staleness) in closed form, on the training
episodes only, exactly as export_shrink.py does for staleness 1:

    beta_k[cell] = sum <R[c], R[c-k]> / sum ||R[c-k]||^2

The output keeps a `coef` field holding the staleness-1 weights, so the file is a
drop-in for any consumer that ignores staleness, and adds `coef_k` for consumers
that do not.

usage: export_shrink_k.py <npz> <out.json> [max_staleness]
"""
from __future__ import annotations

import json, sys
import numpy as np

from report_families import TEST_EVERY, load

EPS = 1e-30


def main(src, dst, kmax=4):
    z, G, cell, ep = load(src)
    keep = G[:, 0, 0] > 0
    G, cell, ep = G[keep], cell[keep], ep[keep]
    kmax = min(int(kmax), G.shape[1] - 1)

    # training episodes only: the closed loop is scored on rollouts, but the
    # held-out split is what makes the offline numbers honest
    tr = (ep % TEST_EVERY) != 0
    fit_eps = sorted(int(e) for e in np.unique(ep[tr]))
    keys = sorted(np.unique(cell))

    coef_k = {}
    for k in range(1, kmax + 1):
        per = {}
        for c in keys:
            m = tr & (cell == c)
            den = float(G[m, k, k].sum())
            b = float(G[m, 0, k].sum() / den) if den > EPS else 1.0
            per[f"{int(c)//1000},{int(c)%1000}"] = [b]
        coef_k[str(k)] = per
        v = sorted(x[0] for x in per.values())
        print(f"  staleness {k}: {len(per)} cells | median {v[len(v)//2]:.3f} "
              f"| range [{v[0]:.3f}, {v[-1]:.3f}] | share <1: "
              f"{100*sum(1 for x in v if x < 1)/len(v):.1f}%")

    out = {"cols": ["tap1"], "staleness_aware": True, "max_staleness": kmax,
           "fit_episodes": fit_eps,
           "coef": coef_k["1"],          # drop-in for staleness-unaware consumers
           "coef_k": coef_k}
    json.dump(out, open(dst, "w"))
    print(f"  -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: export_shrink_k.py <npz> <out.json> [max_staleness]")
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else 4)

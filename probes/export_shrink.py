"""Emit the per-cell substitution coefficients for a closed-loop test.

C3ache substitutes the cached residual unchanged: R_hat = R[c-1]. The trace study
says the least-squares coefficient on R[c-1] is not 1 but something smaller, and
that it differs per cell. This writes those coefficients out so the closed-loop
harness can use them in place of the plain hold, at identical compute -- the
substitution costs one scalar multiply on a tensor that is being copied anyway.

Coefficients are fit on the training episodes only, so the closed-loop run scores
them on data they were not fit to.
"""
from __future__ import annotations

import json, sys
import numpy as np

import analyze_families as A
from report_families import TEST_EVERY, load, per_cell_sum

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    cols = sys.argv[3].split(",") if len(sys.argv) > 3 else ["tap1"]
    z, G, cell, ep = load(src)
    tr = (ep % TEST_EVERY) != 0
    ids, inv, Gs = per_cell_sum(G, cell, tr)
    a, _ = A._solve(Gs, cols)
    out = {"cols": cols, "fit_episodes": sorted(set(ep[tr].tolist())),
           "coef": {f"{i//1000},{i%1000}": [float(x) for x in row]
                    for i, row in zip(ids, a)}}
    json.dump(out, open(dst, "w"), indent=1)
    b = a[:, 0]
    print(f"{len(ids)} cells | coefficient on R[c-1]: median {np.median(b):.3f}, "
          f"mean {b.mean():.3f}, range [{b.min():.3f}, {b.max():.3f}]")
    print(f"share below 1.0 (shrinking): {100*(b<1).mean():.1f}%  -> {dst}")

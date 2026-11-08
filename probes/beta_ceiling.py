"""How much headroom does a LEARNED scalar weight actually have?

The per-channel oracle bound in the family table (+40%) is often misread as the
headroom for a learned predictor. It is not: it fits a separate scalar per
CHANNEL per observation with the true residual in hand, so nothing that lacks the
true residual can approach it.

The honest ceiling for any predictor that emits ONE weight per cell is the
per-observation oracle: the best single weight for that specific cycle, again
chosen with the true residual. The gap between what we ship (one weight per cell,
fit on training episodes) and that ceiling is the entire prize available to a
better estimator -- an MLP, a Kalman filter, anything.

Three levels, all scored on held-out episodes:
  fixed    one weight per cell                      what we ship
  learned  weight predicted from features available what a small model could
           WITHOUT the true residual                reach
  oracle   best weight for this observation         the ceiling

`learned` uses ridge regression on a quadratic expansion of tap-only features,
which stands in for a small MLP: it has comparable capacity over the same inputs
and fits in closed form, so the number is reproducible rather than dependent on
an optimiser's mood.

usage: beta_ceiling.py <npz> [<npz> ...]
"""
from __future__ import annotations

import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report_families as RF

EPS = 1e-30


def analyse(path, label):
    z, G, cell, ep = RF.load(path)
    k = G[:, 0, 0] > 0
    G, cell, ep = G[k], cell[k], ep[k]
    tt, t1, oo = G[:, 0, 0], G[:, 0, 1], G[:, 1, 1]
    tr = (ep % RF.TEST_EVERY) != 0
    te = ~tr
    n = int(cell.max()) + 1

    def rel(w):
        return float(np.mean(np.sqrt(np.maximum(tt[te] - 2 * w * t1[te] + w * w * oo[te], 0)
                                     / np.maximum(tt[te], EPS))))

    unit = rel(1.0)
    num = np.bincount(cell[tr], weights=t1[tr], minlength=n)
    den = np.bincount(cell[tr], weights=oo[tr], minlength=n)
    per_cell = np.where(den > EPS, num / np.maximum(den, EPS), 0.0)
    b_fix = per_cell[cell[te]]
    b_orc = (t1 / np.maximum(oo, EPS))[te]          # uses the true residual

    # features a predictor could actually see at inference: statistics among the
    # cached taps only. Nothing here involves the target.
    F = np.column_stack([
        np.log(np.maximum(G[:, 1, 1], EPS)),
        np.log(np.maximum(G[:, 2, 2], EPS)),
        G[:, 1, 2] / np.sqrt(np.maximum(G[:, 1, 1] * G[:, 2, 2], EPS)),
        G[:, 1, 3] / np.sqrt(np.maximum(G[:, 1, 1] * G[:, 3, 3], EPS)),
        G[:, 1, 5] / np.sqrt(np.maximum(G[:, 1, 1] * G[:, 5, 5], EPS)),
        np.sqrt(np.maximum(G[:, 2, 2] / np.maximum(G[:, 1, 1], EPS), 0)),
        z["step"][k].astype(float), z["block"][k].astype(float),
        np.nan_to_num(z["obs_delta"][k].astype(float)),
        per_cell[cell],
    ])
    F = np.nan_to_num(F)
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
    Fz = (F - mu) / sd
    y = np.clip(t1 / np.maximum(oo, EPS), -0.5, 2.0)

    def expand(X):
        cols = [np.ones(len(X)), X, X ** 2]
        for i in range(X.shape[1]):
            for j in range(i + 1, X.shape[1]):
                cols.append((X[:, i] * X[:, j])[:, None])
        return np.column_stack(cols)

    Xtr, Xte = expand(Fz[tr]), expand(Fz[te])
    w = np.linalg.solve(Xtr.T @ Xtr + 1e-3 * np.eye(Xtr.shape[1]), Xtr.T @ y[tr])
    b_lrn = np.clip(Xte @ w, 0.0, 1.5)

    g = lambda b: 100 * (1 - rel(b) / unit)
    print(f"\n{label}")
    print(f"  {'published beta=1':32s} {unit:.4f}")
    print(f"  {'fixed per cell (we ship)':32s} {rel(b_fix):.4f}   {g(b_fix):+6.2f}%")
    print(f"  {'learned per observation':32s} {rel(b_lrn):.4f}   {g(b_lrn):+6.2f}%")
    print(f"  {'ORACLE per observation':32s} {rel(b_orc):.4f}   {g(b_orc):+6.2f}%")
    head, got = g(b_orc) - g(b_fix), g(b_lrn) - g(b_fix)
    print(f"  -> headroom for ANY scalar predictor: {head:+.2f} pts")
    print(f"  -> this model captures:               {got:+.2f} pts "
          f"({100*got/max(head,1e-9):.0f}%)")
    return {"file": os.path.basename(path), "unit": unit, "fixed": rel(b_fix),
            "learned": rel(b_lrn), "oracle": rel(b_orc),
            "headroom_pts": head, "captured_pts": got}


if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit("usage: beta_ceiling.py <npz> ...")
    out = [analyse(p, os.path.basename(p)) for p in sys.argv[1:]]
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "results", "residual", "beta_ceiling.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(f"\nwrote {os.path.normpath(dst)}")

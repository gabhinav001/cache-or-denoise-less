"""Is the attenuation weight staleness-dependent, and can one parameter predict it?

The correction shipped so far uses ONE weight per cell, fit against the residual
from one cycle back, and then applies that same weight however stale the cache
actually is. The closed-loop data says that is wrong: the correction is worth
-1.2 points at one cycle of staleness, +5.2 at three and +14.6 at seven. A weight
that does not know how stale its input is cannot be right at both ends.

This asks the question offline, from Grams already recorded. The recorded
features tap1..tap4 ARE the residuals from 1..4 cycles back, so the optimal
weight at staleness k is available in closed form per cell:

    beta_k = <R_t, R_{t-k}> / ||R_{t-k}||^2 = G[0,k] / G[k,k]

Four strategies are scored on held-out episodes, all at identical inference cost:

  unit         beta = 1                      the published method
  constant     beta_1 at every staleness     what we ship today
  per-k        beta_k, fit per staleness     4x the parameters; an upper bound
  ar1          beta_1^k                      SAME parameters as `constant`,
                                             extrapolated by a decay model

`ar1` is the interesting one. If a residual decorrelates multiplicatively --
R_t = rho * R_{t-1} + noise, which is the state equation a Kalman filter assumes
-- then beta_k = beta_1^k exactly, and one fitted scalar per cell gives the right
weight at EVERY refresh interval, including intervals never measured. That is a
falsifiable prediction, and this script is the falsification test.

The two-tap rows go one step further: at staleness k, the best linear predictor
from {tap_k, tap_{k+1}} is what a Kalman filter with a two-dimensional state
would produce. It bounds what tracking buys over plain shrinkage.

usage: kalman_beta.py <npz> [<npz> ...]
"""
from __future__ import annotations

import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report_families as RF

TEST_EVERY = RF.TEST_EVERY          # every Nth episode is held out
# tap1..tapK are recorded, K depending on when the collection was made: the
# original recording stored four, the extended one seven. Derived per file, so
# an old artefact still reports ages 1-4 and a new one reaches the ages tau=8
# actually reuses.
def maxk_of(G):
    return G.shape[-1] - 4          # columns are target, taps..., mean8, ema09, ema06
EPS = 1e-30


def per_cell_sums(G, cell, mask, i, j):
    """Sum G[:, i, j] within each cell over the masked cycles."""
    n = int(cell.max()) + 1
    return np.bincount(cell[mask], weights=G[mask, i, j], minlength=n)


def solve2(a11, a12, a22, b1, b2):
    """Per-cell 2x2 least squares, returning both coefficients."""
    det = a11 * a22 - a12 * a12
    ok = np.abs(det) > EPS
    w1 = np.where(ok, (b1 * a22 - b2 * a12) / np.where(ok, det, 1.0), 0.0)
    w2 = np.where(ok, (b2 * a11 - b1 * a12) / np.where(ok, det, 1.0), 0.0)
    return w1, w2


def err2(G, cell, mask, k, w_per_cell):
    """Squared error of substituting w * tap_k, summed over masked observations."""
    w = w_per_cell[cell[mask]]
    return float(np.sum(G[mask, 0, 0] - 2 * w * G[mask, 0, k] + w * w * G[mask, k, k]))


def analyse(path):
    z, G, cell, ep = RF.load(path)
    keep = G[:, 0, 0] > 0
    G, cell, ep = G[keep], cell[keep], ep[keep]
    MAXK = maxk_of(G)          # 4 for the original recording, 7 for the extended
    tr = (ep % TEST_EVERY) != 0            # training episodes
    te = ~tr                               # held out, never used in any fit
    meta = json.loads(str(z["meta"][0]))

    print(f"\n{'='*80}\n{os.path.basename(path)}   {meta.get('gpu','?')}")
    print(f"  {len(G):,} cell-observations | {len(np.unique(cell))} cells | "
          f"{len(np.unique(ep))} episodes | held out every {TEST_EVERY}th")

    # ---- how the optimal weight moves with staleness -----------------------
    beta = {}
    for k in range(1, MAXK + 1):
        num = per_cell_sums(G, cell, tr, 0, k)
        den = per_cell_sums(G, cell, tr, k, k)
        beta[k] = np.where(den > EPS, num / np.maximum(den, EPS), 0.0)

    valid = per_cell_sums(G, cell, tr, 1, 1) > EPS
    print(f"\n  optimal weight by staleness (median over {int(valid.sum())} cells)")
    print(f"    {'cycles stale':16s} " + "".join(f"{k:>9d}" for k in range(1, MAXK + 1)))
    print(f"    {'beta_k':16s} " + "".join(f"{np.median(beta[k][valid]):9.4f}" for k in range(1, MAXK + 1)))
    print(f"    {'beta_1^k (model)':16s} " +
          "".join(f"{np.median(beta[1][valid] ** k):9.4f}" for k in range(1, MAXK + 1)))

    # does the decay model hold per cell, not just at the median?
    rel = []
    for k in range(2, MAXK + 1):
        pred = beta[1][valid] ** k
        act = beta[k][valid]
        rel.append(float(np.median(np.abs(pred - act) / np.maximum(np.abs(act), EPS))))
    # This is the falsification statistic the docstring promises, and it lived
    # only on stdout: nothing downstream could quote it and no check could verify
    # it. Note it is a MEDIAN OF PER-CELL RATIOS, which is not the ratio of the
    # two medians printed above it -- at age 7 they are 59.8% and 68.2%. Writing
    # it out means a write-up can cite the per-cell number and be checked.
    rel_by_age = {str(k): round(r, 6) for k, r in zip(range(2, MAXK + 1), rel)}
    print(f"    median |model - actual| / actual, k=2..{MAXK}: " +
          "  ".join(f"{r*100:.1f}%" for r in rel))

    # ---- held-out error under each strategy --------------------------------
    ones = np.ones_like(beta[1])
    strategies = {
        "unit (beta=1, published)":        lambda k: ones,
        "constant beta_1 (shipped today)": lambda k: beta[1],
        "ar1 beta_1^k (1 param/cell)":     lambda k: beta[1] ** k,
        "per-k beta_k (4 params/cell)":    lambda k: beta[k],
    }
    tgt = {k: float(np.sum(G[te, 0, 0])) for k in range(1, MAXK + 1)}

    print(f"\n  held-out relative L2 error by staleness (lower is better)")
    print(f"    {'strategy':34s} " + "".join(f"{'k='+str(k):>9s}" for k in range(1, MAXK + 1)) + f"{'mean':>9s}")
    print(f"    {'-'*34} " + "-" * (9 * MAXK + 9))
    rows = {}
    for name, fn in strategies.items():
        vals = [np.sqrt(err2(G, cell, te, k, fn(k)) / tgt[k]) for k in range(1, MAXK + 1)]
        rows[name] = vals
        print(f"    {name:34s} " + "".join(f"{v:9.4f}" for v in vals) + f"{np.mean(vals):9.4f}")

    # two-tap: what a Kalman filter with a 2-d state could reach
    two = []
    for k in range(1, MAXK):
        a11 = per_cell_sums(G, cell, tr, k, k)
        a22 = per_cell_sums(G, cell, tr, k + 1, k + 1)
        a12 = per_cell_sums(G, cell, tr, k, k + 1)
        b1 = per_cell_sums(G, cell, tr, 0, k)
        b2 = per_cell_sums(G, cell, tr, 0, k + 1)
        w1, w2 = solve2(a11, a12, a22, b1, b2)
        u, v = w1[cell[te]], w2[cell[te]]
        e2 = np.sum(G[te, 0, 0]
                    - 2 * u * G[te, 0, k] - 2 * v * G[te, 0, k + 1]
                    + u * u * G[te, k, k] + v * v * G[te, k + 1, k + 1]
                    + 2 * u * v * G[te, k, k + 1])
        two.append(float(np.sqrt(e2 / tgt[k])))
    print(f"    {'two-tap {k,k+1} (2-d Kalman bound)':34s} " +
          "".join(f"{v:9.4f}" for v in two) + f"{'--':>9s}{'':>9s}")

    base = rows["unit (beta=1, published)"]
    cur = rows["constant beta_1 (shipped today)"]
    ar1 = rows["ar1 beta_1^k (1 param/cell)"]
    perk = rows["per-k beta_k (4 params/cell)"]
    print(f"\n  reduction vs the published method, averaged over staleness")
    for nm, v in (("constant beta_1", cur), ("ar1 beta_1^k", ar1), ("per-k beta_k", perk)):
        print(f"    {nm:22s} {100*(1-np.mean(v)/np.mean(base)):+6.2f}%")
    gain = 100 * (1 - np.mean(ar1) / np.mean(cur))
    verdict = ("worth taking: same parameter count, so it costs nothing"
               if gain > 0.5 else
               "NOT worth taking: the decay model does not hold on this backbone"
               if gain < -0.5 else "a wash on this backbone")
    print(f"\n  ar1 vs what we ship today: {gain:+.2f}%  -- {verdict}")

    # does the multiplicative-decay assumption actually hold here?
    mono = all(np.median(beta[k][valid]) >= np.median(beta[k + 1][valid])
               for k in range(1, MAXK))
    print(f"  decay model check: beta_k monotonically decreasing? "
          f"{'yes' if mono else 'NO -- beta rises again at some k'};  "
          f"worst |model-actual| {max(rel)*100:.0f}%")

    # where the shipped weight is wrong, and in which direction. It is fit at
    # k=1, so it is exactly right there by construction; the question is only
    # how wrong it becomes as the cache goes stale.
    print(f"\n  the shipped weight is fit at one cycle of staleness and applied")
    print(f"  unchanged however stale the cache is:")
    for k in range(2, MAXK + 1):
        d = float(np.median(beta[1][valid] - beta[k][valid]))
        direction = ("under-attenuates: applies %.3f where %.3f is optimal"
                     % (np.median(beta[1][valid]), np.median(beta[k][valid]))) if d > 0 else \
                    ("over-attenuates: applies %.3f where %.3f is optimal"
                     % (np.median(beta[1][valid]), np.median(beta[k][valid])))
        print(f"    k={k}: off by {d:+.4f}  -- {direction}")

    return {"file": os.path.basename(path), "n": int(len(G)),
            "beta_median": {k: float(np.median(beta[k][valid])) for k in range(1, MAXK + 1)},
            "heldout": {n: [float(x) for x in v] for n, v in rows.items()},
            "two_tap": two, "ar1_vs_constant_pct": float(gain),
            "decay_model_holds": bool(mono and max(rel) < 0.25),
            "rel_err_by_age": rel_by_age,
            "rel_err_is": ("median over cells of |beta_1^k - beta_k| / |beta_k|; "
                           "NOT |median(beta_1^k) - median(beta_k)| / median(beta_k)")}


if __name__ == "__main__":
    # Writing is now OPT-IN. This probe used to overwrite kalman_beta.json with
    # exactly the files named on the command line, so inspecting ONE artefact --
    # the obvious thing to do, and what the audit map told a reader to do --
    # silently replaced a five-entry committed artefact with a one-entry file.
    # That happened three times, twice to this very file, and each time the only
    # defence was a comment asking the reader to be careful. A read-only default
    # removes the hazard instead of documenting it; verify.sh passes --write.
    args = [a for a in sys.argv[1:] if a != "--write"]
    write = "--write" in sys.argv[1:]
    if not args:
        sys.exit(__doc__.strip().splitlines()[-1])
    out = [analyse(p) for p in args]
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "results", "residual", "kalman_beta.json")
    if not write:
        print("\n  (read-only: pass --write with the FULL file set to regenerate\n"
              "   %s)" % os.path.normpath(dst))
        raise SystemExit(0)
    json.dump(out, open(dst, "w"), indent=1)
    print(f"\nwrote {os.path.normpath(dst)}")

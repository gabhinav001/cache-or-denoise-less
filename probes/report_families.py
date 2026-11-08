"""Headline numbers: how much every cheap predictor family buys over a flat hold.

Reports each family twice.
  oracle -- coefficients fit per cell on the very observations they are scored
            on. Undeployable; an upper bound on per-cell tuning.
  honest -- coefficients fit per cell by summing Grams over 3/4 of the episodes,
            scored on the held-out 1/4. What a real scheme would get.

The conditioned rows additionally fit a separate coefficient set per observation-
change quartile, testing whether knowing how much the scene moved helps predict
the residual. Held out the same way.
"""
from __future__ import annotations

import json, sys
import numpy as np

import analyze_families as A
from famlib import COLS

TEST_EVERY = 4          # every 4th episode is held out


def load(path):
    z = np.load(path, allow_pickle=True)
    G = A.unpack(z["gram"].astype(np.float64))
    cell = z["step"].astype(np.int32) * 1000 + z["block"].astype(np.int32)
    return z, G, cell, z["ep"].astype(np.int32)


def per_cell_sum(G, cell, mask):
    """Sum Grams within each cell over the selected observations."""
    ids, inv = np.unique(cell, return_inverse=True)
    out = np.zeros((len(ids),) + G.shape[1:])
    np.add.at(out, inv[mask], G[mask])
    return ids, inv, out


MIN_TRAIN = 20          # observations a fit needs before it is trusted


def _coeffs(G, key, tr, cols):
    """Per-key least squares over the training observations, plus each key's count."""
    ids, inv, Gs = per_cell_sum(G, key, tr)
    cnt = np.zeros(len(ids))
    np.add.at(cnt, inv[tr], 1.0)
    a, _ = A._solve(Gs, cols)
    return a, inv, cnt


def honest(G, cell, ep, cols, cond=None, glob=False):
    """Fit on train episodes, score on held-out.

    Conditioning splits each cell's training data further, and a bin that no
    training episode reached would otherwise be solved from an all-zero Gram,
    yielding zero coefficients and a relative error of exactly 1. Such bins fall
    back to the cell's unconditioned fit, so conditioning is credited only where
    it actually had data to learn from.
    """
    tr, te = (ep % TEST_EVERY) != 0, (ep % TEST_EVERY) == 0
    base = np.zeros(len(cols)) if glob else None
    if glob:
        a_all, _ = A._solve(per_cell_sum(G, np.zeros_like(cell), tr)[2], cols)
        aa = np.repeat(a_all, len(G), axis=0)
    else:
        a_c, inv_c, cnt_c = _coeffs(G, cell, tr, cols)
        aa = a_c[inv_c]
        if cond is not None:
            a_k, inv_k, cnt_k = _coeffs(G, cell * 10 + cond, tr, cols)
            ok = (cnt_k >= MIN_TRAIN)[inv_k]
            aa = np.where(ok[:, None], a_k[inv_k], aa)
    idx = A._cols_idx(G, cols)   # layout of THIS Gram, not of the current recorder
    Ae = G[..., idx, :][..., :, idx]
    b = G[..., idx, 0]
    err2 = G[..., 0, 0] - 2 * (b * aa).sum(-1) + np.einsum("ni,nij,nj->n", aa, Ae, aa)
    return err2, te


def task_groups(meta, ep):
    """Map each observation to a task id, when the manifest records tasks.

    Holding out episodes tests whether a coefficient transfers to new rollouts of
    tasks it was fit on. Holding out whole tasks is the deployment question:
    does it transfer to instructions and scenes never calibrated against?
    """
    eps = meta.get("episodes") or []
    if not eps or "task" not in eps[0]: return None
    names = [str(e["task"]) for e in eps]
    uniq = {t: i for i, t in enumerate(sorted(set(names)))}
    tid = np.array([uniq[n] for n in names])
    return tid[np.clip(ep, 0, len(tid) - 1)], len(uniq)


def summarize(path, label):
    z, G, cell, ep = load(path)
    meta = json.loads(str(z["meta"][0]))
    tgt2 = G[:, 0, 0]
    keep = tgt2 > 0
    G, cell, ep, tgt2 = G[keep], cell[keep], ep[keep], tgt2[keep]
    hold2 = A.hold_err2(G)
    r_hold = A.relative(hold2, tgt2)
    te = (ep % TEST_EVERY) == 0

    # observation-change quartiles, for the conditioned rows
    od = z["obs_delta"][keep].astype(np.float64)
    fin = np.isfinite(od)
    q = np.zeros(len(od), np.int32)
    if fin.sum() > 10:
        edges = np.quantile(od[fin], [0.25, 0.5, 0.75])
        q = np.digitize(od, edges).astype(np.int32)
        q[~fin] = 0

    print(f"\n{'='*78}\n{label}   {meta.get('gpu','?')}")
    print(f"  {len(G):,} cell-observations | {len(np.unique(cell))} cells | "
          f"{len(np.unique(ep))} episodes | held out every {TEST_EVERY}th")
    print(f"  zero-order hold (the baseline, no free parameters): rel-L2 "
          f"{r_hold.mean():.4f}  [held-out {r_hold[te].mean():.4f}]")

    a_star, alpha2 = A.alpha_family(G)
    print(f"  optimal alpha in the published two-tap family: median {np.median(a_star):+.3f}, "
          f"mean {a_star.mean():+.3f}, share negative {100*(a_star<0).mean():.1f}%")

    # beta factors exactly as cosine x magnitude ratio, which separates the two ways
    # a cached residual can be wrong: pointing elsewhere, or being the wrong size
    beta = G[:, 0, 1] / np.maximum(G[:, 1, 1], 1e-30)
    rho = G[:, 0, 1] / np.sqrt(np.maximum(G[:, 0, 0] * G[:, 1, 1], 1e-30))
    nr = np.sqrt(G[:, 0, 0] / np.maximum(G[:, 1, 1], 1e-30))
    print(f"  cached vs current residual: cosine {np.median(rho):.3f} "
          f"({np.degrees(np.arccos(np.clip(np.median(rho),-1,1))):.0f} degrees apart), "
          f"magnitude ratio {np.median(nr):.3f}")
    print(f"  => least-squares weight on the cached residual: median {np.median(beta):.3f}, "
          f"share below 1 (attenuating): {100*(beta<1).mean():.1f}%")
    st, bl = z["step"][keep], z["block"][keep]
    def spread(v, lab):
        u = np.unique(v)
        pick = u if len(u) <= 6 else u[np.linspace(0, len(u) - 1, 6).astype(int)]
        return f"  weight by {lab}: " + "  ".join(
            f"{lab[0]}{int(k)}={np.median(beta[v == k]):.2f}" for k in pick)
    print(spread(st, "denoise step")); print(spread(bl, "block depth"))

    rows = [("alpha (published family)", A.relative(alpha2, tgt2).mean(), None)]
    for name, cols in A.LINEAR.items():
        orc = A.relative(A.fit_eval(G, G, cols), tgt2).mean()
        e2, tmask = honest(G, cell, ep, cols)
        rows.append((f"{name} [{'+'.join(cols)}]", orc, A.relative(e2[tmask], tgt2[tmask]).mean()))
    for name in A.NONLINEAR:
        rows.append((f"{name} (oracle, per-slice rescale)", z[name][keep].astype(np.float64).mean(), None))
    for name in ("shrink1", "lin2", "full7"):
        e2, tmask = honest(G, cell, ep, A.LINEAR[name], cond=q)
        rows.append((f"{name} + obs-change quartile", None,
                     A.relative(e2[tmask], tgt2[tmask]).mean()))
    for name in ("shrink1", "full7"):
        e2, tmask = honest(G, cell, ep, A.LINEAR[name], glob=True)
        rows.append((f"{name} one coefficient set for all cells", None,
                     A.relative(e2[tmask], tgt2[tmask]).mean()))
    tg = task_groups(meta, np.arange(len(np.unique(ep)) + 1))
    if tg is not None:
        tid_all = tg[0][ep]
        for name in ("shrink1", "full7"):
            e2, tmask = honest(G, cell, tid_all, A.LINEAR[name])
            # This row holds out TASKS, not episodes, so it must be scored against
            # the hold error on those same tasks. Dividing it by the episode-split
            # baseline compares two different populations: the held-out tasks are
            # harder (hold error 0.6955 against 0.6530), so the mismatch understated
            # the gain as +2.9% when like-for-like it is +8.8%, and turned intact
            # cross-task transfer into the paper's headline "mostly does not
            # transfer" limitation.
            rows.append((f"{name} fit and scored on disjoint tasks", None,
                         A.relative(e2[tmask], tgt2[tmask]).mean(),
                         r_hold[tmask].mean()))

    base_o, base_h = r_hold.mean(), r_hold[te].mean()
    # (name, oracle, held-out, held-out baseline) -- the baseline is per-row because
    # not every row holds out the same thing
    rows = [r if len(r) == 4 else (*r, base_h) for r in rows]
    print(f"\n  {'family':44s} {'oracle':>16s} {'held-out':>16s}")
    print(f"  {'-'*44} {'-'*16} {'-'*16}")
    print(f"  {'zero-order hold':44s} {base_o:8.4f}  {'--':>6s} {base_h:8.4f}  {'--':>6s}")
    for name, o, h, bh in rows:
        so = f"{o:8.4f} {100*(1-o/base_o):+5.1f}%" if o is not None else " " * 15
        sh = f"{h:8.4f} {100*(1-h/bh):+5.1f}%" if h is not None else " " * 15
        print(f"  {name:44s} {so} {sh}")
    return {"label": label, "n": int(len(G)), "episodes": int(len(np.unique(ep))),
            "cells": int(len(np.unique(cell))), "hold": float(base_o),
            "hold_heldout": float(base_h), "alpha_median": float(np.median(a_star)),
            "alpha_frac_neg": float((a_star < 0).mean()),
            # 4-tuples: (name, oracle, held-out, held-out baseline). The baseline
            # travels with the row because the disjoint-task rows hold out tasks
            # rather than episodes and must be scored against those tasks' hold
            # error, not the episode split's.
            "rows": [(n, o, h, bh) for n, o, h, bh in rows]}


if __name__ == "__main__":
    out = [summarize(p, l) for p, l in zip(sys.argv[1::2], sys.argv[2::2])]
    # Write beside the tensors, not into the working directory. This wrote to
    # cwd, so the copy the results page quotes sat at the repo root, was never
    # regenerated by verify.sh, and had been truncated to an empty list without
    # anything noticing -- five rows of the predictor-family table had no
    # artefact behind them at all.
    import os
    dst = os.environ.get("WAM_FAMILIES_OUT") or os.path.join(
        os.path.dirname(os.path.abspath(sys.argv[1])), "families_summary.json")
    json.dump(out, open(dst, "w"), indent=1, default=float)
    print(f"\nwrote {dst}")

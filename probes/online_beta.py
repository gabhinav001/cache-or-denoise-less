"""Can a practitioner get the attenuation weight without any extra full compute?

The weight is beta = <R[c], R[c-k]> / ||R[c-k]||^2, which needs the TRUE current
residual R[c] -- the thing caching exists to avoid computing. So the obvious
reading is that beta demands an offline calibration pass under full compute.

It does not, and the reason is structural: any cache with a finite refresh
interval already recomputes everything every tau cycles. On a refresh cycle you
hold both the freshly computed R[c] and the cache from the previous refresh,
R[c-tau]. That is exactly the inner product beta needs. The estimate rides along
on work the method already performs, costs one scalar accumulator per cell, and
adapts to the scene rather than being frozen at calibration time.

This scores four strategies on held-out cycles, all against the same data:

  unit        beta = 1                     the published method
  global      beta = 0.8 everywhere        no calibration of any kind
  online      estimated from refresh       zero extra compute, adaptive
              cycles only, prequentially
  offline     per-cell least squares       what the paper ships; needs a
              on training episodes         calibration pass

"Prequential" matters: at each held-out cycle the online weight uses only
refreshes that happened strictly EARLIER, so it never sees its own answer. The
warm-up curve then says how many refreshes are needed before it is useful.

usage: online_beta.py <npz> [<npz> ...]
"""
from __future__ import annotations

import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report_families as RF

TAUS = (2, 4)          # tap_k is the residual k cycles back, so k == staleness
EPS = 1e-30


def rel(e2, t2):
    return float(np.mean(np.sqrt(np.maximum(e2, 0.0) / np.maximum(t2, EPS))))


def analyse(path):
    z, G, cell, ep = RF.load(path)
    keep = G[:, 0, 0] > 0
    G, cell, ep = G[keep], cell[keep], ep[keep]
    cyc = z["cycle"][keep].astype(np.int64)
    meta = json.loads(str(z["meta"][0]))
    # cell ids are step*1000+block, so sparse: max+1 sizes the bincount, but the
    # count worth printing is how many distinct cells there actually are
    ncell = int(cell.max()) + 1
    n_distinct = len(np.unique(cell))

    print(f"\n{'='*78}\n{os.path.basename(path)}   {meta.get('gpu','?')}")
    print(f"  {len(G):,} cell-observations | {n_distinct} cells | "
          f"{len(np.unique(ep))} episodes")

    order = np.lexsort((cyc, ep))          # walk cycles in the order they happen
    out = {}
    for tau in TAUS:
        if tau >= G.shape[1]: continue
        tt = G[:, 0, 0]
        tr = (ep % RF.TEST_EVERY) != 0
        te = ~tr

        # THE SUBTLETY THAT MAKES OR BREAKS THIS. A cache with interval tau is
        # refreshed at c % tau == 0 and then reused at staleness 1..tau-1. But a
        # refresh cycle only ever exposes the correlation at staleness EXACTLY
        # tau -- it compares the residual it just computed against the cache from
        # the previous refresh. Residuals decorrelate with lag, so beta_tau is
        # smaller than the weight the used staleness levels want, and applying it
        # directly over-attenuates. Estimate at tau, then correct for the
        # staleness actually in play:  beta_k = beta_tau ** (k / tau).
        est_k = tau                                # what a refresh can measure
        use_k = [k for k in range(1, tau)] or [1]  # what the cache actually uses

        # --- offline: per-cell least squares at each used staleness ----------
        b_off = {}
        for k in use_k:
            num = np.bincount(cell[tr], weights=G[tr, 0, k], minlength=ncell)
            den = np.bincount(cell[tr], weights=G[tr, k, k], minlength=ncell)
            b_off[k] = np.where(den > EPS, num / np.maximum(den, EPS), 0.0)

        # --- online: prequential, from refresh cycles only -------------------
        run_n = np.zeros(ncell); run_d = np.zeros(ncell)
        nref = np.zeros(ncell, dtype=np.int64)
        b_tau = np.ones(len(G)); seen = np.zeros(len(G), dtype=np.int64)
        for i in order:
            c = cell[i]
            b_tau[i] = run_n[c] / run_d[c] if run_d[c] > EPS else 1.0
            seen[i] = nref[c]
            if cyc[i] % tau == 0:                  # a refresh happens here
                run_n[c] += G[i, 0, est_k]; run_d[c] += G[i, est_k, est_k]
                nref[c] += 1

        def err(wsel):
            e = t = 0.0
            for k in use_k:
                w = wsel(k)
                e += np.sum(tt[te] - 2 * w * G[te, 0, k] + w * w * G[te, k, k])
                t += np.sum(tt[te])
            return float(np.sqrt(max(e, 0.0) / max(t, EPS)))

        bt = np.clip(b_tau[te], 0.0, 1.5)
        res = {
            "unit":    err(lambda k: 1.0),
            "global":  err(lambda k: 0.8),
            "naive":   err(lambda k: bt),
            "decay":   err(lambda k: np.clip(bt ** (k / tau), 0.0, 1.5)),
            "offline": err(lambda k: b_off[k][cell[te]]),
        }
        print(f"\n  tau={tau}: refreshes measure staleness {est_k}; "
              f"the cache is used at staleness {use_k}")
        print(f"    {'strategy':34s} {'rel L2':>8s} {'vs published':>13s}")
        for k, note in (("unit", "the published method"),
                        ("global", "beta=0.8, no calibration"),
                        ("naive", "apply beta_tau as measured"),
                        ("decay", "beta_tau^(k/tau)  <- the fix"),
                        ("offline", "per-cell per-staleness fit")):
            print(f"    {note:34s} {res[k]:8.4f} {100*(1-res[k]/res['unit']):+12.2f}%")
        d = 100 * (1 - res["decay"] / res["unit"])
        o = 100 * (1 - res["offline"] / res["unit"])
        print(f"    -> free online estimate recovers {100*d/max(o,1e-9):.0f}% of "
              f"a full calibration pass")
        out[tau] = res

    return {"file": os.path.basename(path), "by_tau": {str(k): v for k, v in out.items()}}


if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(__doc__.strip().splitlines()[-1])
    res = [analyse(p) for p in sys.argv[1:]]
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "results", "residual", "online_beta.json")
    json.dump(res, open(dst, "w"), indent=1)
    print(f"\nwrote {os.path.normpath(dst)}")

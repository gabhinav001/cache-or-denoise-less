"""Verify the Gram machinery against brute force before it costs GPU time."""
import numpy as np, torch
from famlib import MultiRecorder, HIST, EMA_RATES, COLS
import analyze_families as A

torch.manual_seed(0); rng = np.random.default_rng(0)
N, T, C, CYC = 5, 8, 16, 16
KEYS = [(k, b) for k in range(1) for b in range(N)]


def make(kind):
    """AR(1) sequences (predictable) or white noise (nothing to predict)."""
    seq, prev = [], rng.standard_normal((N, T, C))
    for _ in range(CYC):
        prev = (0.7 * prev + 0.5 * rng.standard_normal((N, T, C))
                if kind == "ar" else rng.standard_normal((N, T, C)))
        seq.append(prev.copy())
    return seq


def features(seq, c):
    """Rebuild the recorder's feature columns independently, from the past only."""
    ema = [None, None]
    for i in range(c):
        for j, lam in enumerate(EMA_RATES):
            ema[j] = seq[i].copy() if ema[j] is None else lam * ema[j] + (1 - lam) * seq[i]
    # taps 1..7 then the three summary columns, matching famlib.COLS order.
    # Built independently of the recorder on purpose: if both were derived from
    # one list the test could not catch a mis-ordered feature stack.
    taps = [seq[c - k] for k in range(1, 8)]
    return np.stack([seq[c], *taps,
                     np.mean(seq[c-HIST:c], 0), ema[0], ema[1]], 1)   # [N, 11, T, C]


def brute(F):
    """Direct least squares / oracle fits on the raw vectors."""
    flat = F.reshape(F.shape[0], len(COLS), -1)
    tgt, p1, p2 = flat[:, 0], flat[:, 1], flat[:, 2]
    out = {"tgt2": (tgt * tgt).sum(-1), "hold": ((p1 - tgt) ** 2).sum(-1)}
    d, v = tgt - p1, p1 - p2
    out["alpha_a"] = (d * v).sum(-1) / (v * v).sum(-1)
    out["alpha"] = (d * d).sum(-1) - (d * v).sum(-1) ** 2 / (v * v).sum(-1)
    for name, cols in A.LINEAR.items():
        X = np.stack([flat[:, COLS.index(c)] for c in cols], 1)      # [N, k, D]
        coef = np.linalg.solve(X @ X.transpose(0, 2, 1), X @ tgt[:, :, None])
        out[name] = ((tgt - (coef[:, :, 0][:, :, None] * X).sum(1)) ** 2).sum(-1)
    for name, red in (("chan1", 1), ("tok1", 2)):
        num, den = (F[:, 0] * F[:, 1]).sum(red), (F[:, 1] ** 2).sum(red)
        out[name] = out["tgt2"] - (num ** 2 / den).sum(-1)
    a11, a22 = (F[:, 1] ** 2).sum(1), (F[:, 2] ** 2).sum(1)
    a12 = (F[:, 1] * F[:, 2]).sum(1)
    b1, b2 = (F[:, 0] * F[:, 1]).sum(1), (F[:, 0] * F[:, 2]).sum(1)
    det = a11 * a22 - a12 ** 2
    out["chan2"] = out["tgt2"] - ((a22*b1*b1 - 2*a12*b1*b2 + a11*b2*b2) / det).sum(-1)
    return out


fails = 0
for kind in ("ar", "white"):
    seq = make(kind)
    rec = MultiRecorder(torch.device("cpu"))
    got = {}
    for c in range(CYC):
        r = rec.push({k: torch.tensor(seq[c][i]) for i, k in enumerate(KEYS)})
        if r: got[c] = r[(T, C)][1]
    print(f"\n=== {kind}: measured cycles {sorted(got)} ===")
    for c in sorted(got):
        G = A.unpack(got[c]["gram"].astype(np.float64))
        ref = brute(features(seq, c))
        checks = [("hold", A.hold_err2(G), ref["hold"]),
                  ("alpha", A.alpha_family(G)[1], ref["alpha"]),
                  ("alpha_a", A.alpha_family(G)[0], ref["alpha_a"])]
        checks += [(n, A.fit_eval(G, G, cols), ref[n]) for n, cols in A.LINEAR.items()]
        checks += [(n, (got[c][n].astype(np.float64) ** 2) * G[..., 0, 0], ref[n])
                   for n in A.NONLINEAR]
        for name, mine, want in checks:
            scale = max(np.abs(want).max(), 1e-12)
            err = np.abs(mine - want).max() / scale
            if err > 2e-4:
                print(f"  FAIL c={c} {name}: rel {err:.2e}"); fails += 1
    # nesting: a richer family can never do worse than one it contains
    for c in sorted(got):
        G = A.unpack(got[c]["gram"].astype(np.float64))
        e = {n: A.fit_eval(G, G, cols) for n, cols in A.LINEAR.items()}
        e["alpha"], e["hold"] = A.alpha_family(G)[1], A.hold_err2(G)
        for n in A.NONLINEAR:
            e[n] = (got[c][n].astype(np.float64) ** 2) * G[..., 0, 0]
        for a, b in [("shrink1","hold"),("lin2","alpha"),("lin3","lin2"),("lin4","lin3"),
                     ("full7","lin4"),("full7","level"),("chan1","shrink1"),("chan2","chan1")]:
            if (e[a] > e[b] * (1 + 1e-6) + 1e-12).any():
                print(f"  FAIL c={c} nesting {a} <= {b}"); fails += 1
    tgt2 = A.unpack(got[max(got)]["gram"].astype(np.float64))[..., 0, 0]
    for n in ("hold", "lin4", "full7"):
        v = A.relative(A.fit_eval(A.unpack(got[max(got)]["gram"].astype(np.float64)),
                                  A.unpack(got[max(got)]["gram"].astype(np.float64)),
                                  A.LINEAR.get(n, ["tap1"])), tgt2) if n != "hold" else \
            A.relative(A.hold_err2(A.unpack(got[max(got)]["gram"].astype(np.float64))), tgt2)
        print(f"  rel_l2 {n:6s} = {v.mean():.4f}")

print("\nALL CHECKS PASSED" if fails == 0 else f"\n{fails} FAILURES")

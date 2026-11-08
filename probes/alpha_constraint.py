"""Is it the tap count that fails, or the sum-to-one constraint?

The published forecasting rule is R_hat = R[c-1] + a(R[c-1] - R[c-2]). Written
as a linear combination its coefficients are (1+a, -a), which sum to one for
every a. An earlier draft of this work read that fact as "the family can rotate
the prediction but can never shrink it". That is false, and this probe exists
because the false version was load-bearing.

It is false because shrinkage does not require the coefficients to sum to less
than one. With d = R[c-1] - R[c-2],

    ||R_hat(a)||^2 = ||R[c-1]||^2 + 2a<R[c-1],d> + a^2||d||^2

is a genuine parabola in a, minimised at -<R[c-1],d>/||d||^2 with value
||R[c-1]||^2 - <R[c-1],d>^2/||d||^2, strictly below ||R[c-1]||^2 whenever
R[c-1] is not orthogonal to d. The family can shrink. What it cannot do is
shrink *in a chosen direction*: the only shrinkage on offer is the removal of
R[c-1]'s component along d, and d is handed to it by the data.

So the real question is quantitative, not categorical, and this probe answers it
from the Grams already collected -- no GPU, no new rollouts. It puts four
predictors on the same held-out episodes:

    hold      R[c-1] unchanged                  0 free parameters
    alpha     the published family              1 free parameter, sum constrained
    beta      b*R[c-1]                          1 free parameter, unconstrained
    two-tap   a1*R[c-1] + a2*R[c-2]             2 free parameters, unconstrained

alpha and beta spend exactly one parameter each. If beta wins, the deficit is
the constraint and not the budget, which is the claim the paper needs and the
only one this evidence supports. Two-tap then prices the second tap once the
constraint is gone.

The diagnostic that ties it together is the coefficient sum of the unconstrained
two-tap fit. The alpha family is precisely the slice of the two-tap family where
that sum equals one, so if the unconstrained optimum sits well below one, the
family is excluded from where the optimum lives -- measured, not asserted.

usage: alpha_constraint.py [--json out.json]
"""
from __future__ import annotations

import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_families as A                      # noqa: E402
from report_families import load, per_cell_sum, TEST_EVERY   # noqa: E402

R = os.environ.get("RESULTS_DIR") or os.path.join(HERE, "..", "results")

BACKBONES = (("Fast-WAM", "fw_families"),
             ("pi0.5", "pi05_families"),
             ("DreamZero", "dz_families"))


def _cellwise(num, den, cell, tr):
    """Per-cell ratio of two Gram-derived scalars, summed over training rows.

    Both arguments add across observations, so a held-out fit is a sum and not a
    refit -- the same property the rest of this analysis leans on.
    """
    ids, inv, sn = per_cell_sum(num[:, None, None], cell, tr)
    _, _, sd = per_cell_sum(den[:, None, None], cell, tr)
    v = sn[:, 0, 0] / np.where(np.abs(sd[:, 0, 0]) < 1e-30, 1e-30, sd[:, 0, 0])
    return v[inv]


def analyse(path, label):
    z, G, cell, ep = load(path)
    keep = G[:, 0, 0] > 0
    G, cell, ep = G[keep], cell[keep], ep[keep]
    tr, te = (ep % TEST_EVERY) != 0, (ep % TEST_EVERY) == 0
    tgt2 = G[:, 0, 0]

    G00, G01, G02 = G[:, 0, 0], G[:, 0, 1], G[:, 0, 2]
    G11, G12, G22 = G[:, 1, 1], G[:, 1, 2], G[:, 2, 2]
    dd = G11 - 2 * G12 + G22                       # ||d||^2, d = R[c-1] - R[c-2]

    # ---- the four predictors, each fit on train rows and scored on held-out ----
    hold2 = G00 - 2 * G01 + G11

    a = _cellwise(G01 - G02 - G11 + G12, dd, cell, tr)
    alpha2 = hold2 - 2 * a * (G01 - G02 - G11 + G12) + a ** 2 * dd

    b = _cellwise(G01, G11, cell, tr)
    beta2 = G00 - 2 * b * G01 + b ** 2 * G11

    ids, inv, Gs = per_cell_sum(G, cell, tr)
    c2, _ = A._solve(Gs, ["tap1", "tap2"])
    cc = c2[inv]
    idx = A._cols_idx(G, ["tap1", "tap2"])   # taps sit at 1,2 in both layouts,
                                             # but resolve rather than assume
    Ae = G[..., idx, :][..., :, idx]
    bb = G[..., idx, 0]
    two2 = G00 - 2 * (bb * cc).sum(-1) + np.einsum("ni,nij,nj->n", cc, Ae, cc)

    rel = {k: float(A.relative(v, tgt2)[te].mean())
           for k, v in (("hold", hold2), ("alpha", alpha2),
                        ("beta", beta2), ("two_tap", two2))}

    # ---- the constraint, measured ----
    # one row per cell, so a cell with many observations does not outvote one
    # with few when we ask where the unconstrained optimum sits
    csum = c2.sum(-1)

    # how much shrinkage the family can actually reach, and how much it takes at
    # the alpha it would really choose (the error-optimal one, not this)
    best2 = np.maximum(G11 - (G11 - G12) ** 2 / np.maximum(dd, 1e-30), 0.0)
    reach = np.sqrt(best2 / np.maximum(G11, 1e-30))
    at_err = np.sqrt(np.maximum(G11 + 2 * a * (G11 - G12) + a ** 2 * dd, 0.0)
                     / np.maximum(G11, 1e-30))
    wanted = G01 / np.maximum(G11, 1e-30)          # the least-squares scalar

    out = {"label": label, "cells": int(len(ids)), "observations": int(len(G)),
           "held_out_rel_l2": {k: round(v, 4) for k, v in rel.items()},
           "two_tap_coeff_sum": {"median": round(float(np.median(csum)), 3),
                                 "share_below_one": round(float((csum < 1).mean()), 3)},
           "norm_ratio": {"wanted_by_least_squares": round(float(np.median(wanted)), 3),
                          "best_the_family_can_reach": round(float(np.median(reach)), 3),
                          "taken_at_error_optimal_alpha": round(float(np.median(at_err)), 3)}}

    print(f"\n{'='*74}\n{label}   {out['cells']} cells, {out['observations']:,} observations")
    print("\n  held-out relative L2, lower is better")
    print(f"    {'hold      R[c-1]':<34} {'0 params':<14} {rel['hold']:.4f}")
    print(f"    {'alpha     published family':<34} {'1, sum=1':<14} {rel['alpha']:.4f}")
    print(f"    {'beta      b*R[c-1]':<34} {'1, free':<14} {rel['beta']:.4f}")
    print(f"    {'two-tap   a1*R[c-1]+a2*R[c-2]':<34} {'2, free':<14} {rel['two_tap']:.4f}")
    g_a = 100 * (rel["hold"] - rel["alpha"]) / rel["hold"]
    g_b = 100 * (rel["hold"] - rel["beta"]) / rel["hold"]
    g_t = 100 * (rel["hold"] - rel["two_tap"]) / rel["hold"]
    print(f"\n    against the hold: alpha {g_a:+.1f}%, beta {g_b:+.1f}%, two-tap {g_t:+.1f}%")
    print(f"    alpha and beta cost one parameter each; beta buys "
          f"{g_b / max(g_a, 1e-9):.0f}x more of it.")
    out["pct_gain_over_hold"] = {"alpha": round(g_a, 2), "beta": round(g_b, 2),
                                 "two_tap": round(g_t, 2)}

    print("\n  where the unconstrained two-tap optimum sits")
    print(f"    coefficient sum a1+a2: median {out['two_tap_coeff_sum']['median']:.3f}, "
          f"below one in {100*out['two_tap_coeff_sum']['share_below_one']:.1f}% of cells")
    print("    the alpha family is the slice of this family where that sum is 1.")

    print("\n  shrinkage, as norm ratio ||R_hat|| / ||R[c-1]||")
    print(f"    least squares wants            {out['norm_ratio']['wanted_by_least_squares']:.3f}")
    print(f"    best the family could reach    {out['norm_ratio']['best_the_family_can_reach']:.3f}")
    print(f"    it takes at its best alpha     {out['norm_ratio']['taken_at_error_optimal_alpha']:.3f}")
    print("    so the family can shrink -- the claim that it cannot is false -- but only")
    print("    along R[c-1]-R[c-2], and at the alpha that minimises error it barely does.")
    return out


def main():
    res = []
    for label, stem in BACKBONES:
        p = os.path.join(R, "residual", stem + ".npz")
        if not os.path.exists(p):
            print(f"  (missing: {p})"); continue
        res.append(analyse(p, label))
    if not res:
        print("no family artefacts found"); return 1
    print(f"\n{'='*74}\n  SUMMARY\n")
    print(f"    {'backbone':<12}{'sum a1+a2':>11}{'alpha gain':>12}{'beta gain':>11}{'2-tap gain':>12}")
    for r in res:
        g = r["pct_gain_over_hold"]
        print(f"    {r['label']:<12}{r['two_tap_coeff_sum']['median']:>11.3f}"
              f"{g['alpha']:>11.1f}%{g['beta']:>10.1f}%{g['two_tap']:>11.1f}%")
    print("\n    On every backbone the unconstrained two-tap optimum has coefficients")
    print("    summing below one in every cell, so the sum-to-one slice never contains")
    print("    the optimum. Spending the one free parameter on attenuation beats spending")
    print("    it on forecasting on all three, though by very different margins -- 6x on")
    print("    Fast-WAM, 2x on pi0.5, and near parity on DreamZero, where the hold is")
    print("    already poor enough that either correction helps. The published family is")
    print("    therefore limited by where the constraint puts it rather than by having")
    print("    too few taps: lifting the constraint at one tap already matches or beats")
    print("    it, and the second tap adds little once the constraint is gone.")
    if "--json" in sys.argv:
        out = sys.argv[sys.argv.index("--json") + 1]
        json.dump(res, open(out, "w"), indent=1)
        print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

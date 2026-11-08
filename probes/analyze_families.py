"""Fit every cheap predictor family from the recorded Grams.

Two modes per family:
  oracle -- coefficients fit per cell on the very cycles they are scored on.
            Not deployable; an upper bound on what per-cell tuning can reach.
  honest -- coefficients fit per cell on the training episodes, scored on
            held-out ones. What a real scheme would get.

Both are exact: Grams add, so pooling a set of cycles is a sum, and the optimum
over that set is one linear solve. No search, no gradient descent, no chance
that a better coefficient was missed.
"""
from __future__ import annotations

import numpy as np

from famlib import COLS, NCOL, IU

IDX = {c: i for i, c in enumerate(COLS)}

# The Gram width encodes its own layout: n columns pack to n(n+1)/2 floats, so
# 36 is the original eight-column recording and 66 the extended one that adds
# tap5..tap7. Readers must infer it rather than assume the CURRENT NCOL --
# unpacking a shipped 36-wide Gram as 11x11 would not fail loudly, it would
# silently reinterpret every column, which is the failure mode this file can
# least afford.
LEGACY_COLS = ["target", "tap1", "tap2", "tap3", "tap4", "mean8", "ema09", "ema06"]


def ncol_of(width):
    """Number of Gram columns that packs to `width` upper-triangle entries."""
    n = int(round((-1 + (1 + 8 * width) ** 0.5) / 2))
    if n * (n + 1) // 2 != width:
        raise ValueError(f"gram width {width} is not a triangular number")
    return n


def cols_of(width):
    """Column names for a Gram of this width; taps always start at index 1."""
    n = ncol_of(width)
    if n == len(COLS): return list(COLS)
    if n == len(LEGACY_COLS): return list(LEGACY_COLS)
    raise ValueError(f"no known column layout for {n} columns")


def idx_of(width):
    return {c: i for i, c in enumerate(cols_of(width))}

# name -> feature columns fit with free coefficients
LINEAR = {
    "shrink1": ["tap1"],
    "lin2":    ["tap1", "tap2"],
    "lin3":    ["tap1", "tap2", "tap3"],
    "lin4":    ["tap1", "tap2", "tap3", "tap4"],
    "level":   ["tap1", "mean8"],
    "ema1":    ["ema09"],
    "ema2":    ["tap1", "ema09", "ema06"],
    "full7":   ["tap1", "tap2", "tap3", "tap4", "mean8", "ema09", "ema06"],
}
NONLINEAR = ["chan1", "tok1", "chan2"]


def unpack(flat):
    """[..., w] upper triangle -> [..., n, n] symmetric, n inferred from w."""
    import torch
    n = ncol_of(flat.shape[-1])
    iu_t = IU if n == NCOL else torch.triu_indices(n, n)
    G = np.zeros(flat.shape[:-1] + (n, n), dtype=np.float64)
    iu = (iu_t[0].numpy(), iu_t[1].numpy())
    G[..., iu[0], iu[1]] = flat
    G[..., iu[1], iu[0]] = flat
    return G


def _cols_idx(G, cols):
    """Column indices for `cols` in the layout G actually has.

    The module-level IDX describes the CURRENT recording. A Gram loaded from an
    older artefact has fewer columns and a different position for mean8 and the
    EMAs, so indexing it with the current map either raises -- as it did here --
    or, worse, silently reads the wrong feature. Resolve names against the array
    in hand instead.
    """
    m = {c: i for i, c in enumerate(cols_of(G.shape[-1] * (G.shape[-1] + 1) // 2))}
    missing = [c for c in cols if c not in m]
    if missing:
        raise KeyError(f"{missing} not recorded in this {G.shape[-1]}-column Gram")
    return [m[c] for c in cols]


def _solve(G, cols, ridge=1e-9):
    """Least squares over `cols`; returns (coeffs [...,k], residual sum of squares)."""
    idx = _cols_idx(G, cols)
    A = G[..., idx, :][..., :, idx]
    b = G[..., idx, 0]
    scale = np.trace(A, axis1=-2, axis2=-1)[..., None, None] / len(idx)
    A = A + ridge * np.maximum(scale, 1e-30) * np.eye(len(idx))
    a = np.linalg.solve(A, b[..., None])[..., 0]
    return a, G[..., 0, 0] - (b * a).sum(-1)


def hold_err2(G):
    """Zero-order hold: R_hat = R[c-1], no free parameters. The real baseline."""
    return G[..., 0, 0] - 2 * G[..., 0, 1] + G[..., 1, 1]


def alpha_family(G):
    """The published two-tap family R[c-1] + a(R[c-1]-R[c-2]); returns (a*, err2)."""
    dd = hold_err2(G)
    dv = G[..., 0, 1] - G[..., 0, 2] - G[..., 1, 1] + G[..., 1, 2]
    vv = G[..., 1, 1] - 2 * G[..., 1, 2] + G[..., 2, 2]
    a = np.where(vv > 0, dv / np.maximum(vv, 1e-30), 0.0)
    return a, dd - np.where(vv > 0, dv * dv / np.maximum(vv, 1e-30), 0.0)


def fit_eval(G_fit, G_eval, cols, ridge=1e-9):
    """Coefficients from G_fit, error scored on G_eval. G_fit==G_eval gives the oracle."""
    a, _ = _solve(G_fit, cols, ridge)
    idx = _cols_idx(G_eval, cols)
    A = G_eval[..., idx, :][..., :, idx]
    b = G_eval[..., idx, 0]
    quad = np.einsum("...i,...ij,...j->...", a, A, a)
    return G_eval[..., 0, 0] - 2 * (b * a).sum(-1) + quad


def relative(err2, tgt2):
    return np.sqrt(np.maximum(err2, 0) / np.maximum(tgt2, 1e-30))

"""Sufficient statistics for every predictor family cheap enough to beat a block.

The alpha-sweep result closes exactly one family: R_hat = R[c-1] + a(R[c-1]-R[c-2]),
capped at 2.9% over a flat hold even with oracle per-cell a. This module collects
what is needed to close the rest.

Linear families. For any predictor R_hat = sum_i a_i f_i built from vectors f_i
derived from the cached history,

    ||R - sum_i a_i f_i||^2 = G00 - 2 sum_i a_i G0i + sum_ij a_i a_j Gij

so the Gram matrix G over [R, f_1..f_m] determines both the optimal coefficients
(normal equations) and the error they achieve. Nothing else about the residuals
matters. Grams also ADD across cycles, so pooling episodes and splitting
train/test are exact sums, not refits -- an honest held-out number costs nothing.

Columns, in the order the analysis expects:
    0 target R[c]      1..4 taps R[c-1..c-4]
    5 mean of R[c-1..c-8]   (a level, for mean reversion)
    6 EMA lambda=0.9        7 EMA lambda=0.6

Nonlinear families. Two are not linear in the history and are bounded online
instead, each fit against the true residual -- an oracle no deployable scheme
reaches:
    chan1  per-channel rescale of R[c-1]      (C free scalars)
    tok1   per-token rescale of R[c-1]        (T free scalars)
    chan2  per-channel 2-tap on R[c-1],R[c-2] (2C free scalars)

Anything richer needs a weight matrix of order C^2, which is the size of the
projection inside the block it replaces -- so it cannot pay for itself. That
argument, not measurement, bounds the remainder of the space.
"""
from __future__ import annotations

import numpy as np
import torch

# Taps 5 through 7 were not recorded until now, and their absence bounded what
# the cache-age diagnostic could say: tau=8 reuses a residual at ages one to
# seven, while the analysis could only reach four. HIST already retains eight
# cycles, so the history was in memory and simply never written out. Taps stay
# at indices 1..K so raw-index readers (kalman_beta) work on either layout, and
# the eight-column recording remains readable -- analyze_families.unpack infers
# the layout from the packed width rather than assuming this constant.
NCOL = 11                     # target + 7 taps + 3 summary features
HIST = 8                      # cycles of history retained
IU = torch.triu_indices(NCOL, NCOL)
COLS = ["target", "tap1", "tap2", "tap3", "tap4", "tap5", "tap6", "tap7",
        "mean8", "ema09", "ema06"]
EMA_RATES = (0.9, 0.6)


def _ridge(a11, a22, scale=1e-6):
    return scale * (a11 + a22).clamp_min(1e-30)


class FamilyRecorder:
    """One instance per (shape-group of) cells. Cells are stacked so every
    statistic is a single batched GPU op rather than 300 python-level ones."""

    def __init__(self, keys, shape, device):
        self.keys = list(keys)                    # [(step, block), ...]
        self.T, self.C = shape                    # tokens, channels
        self.N = len(self.keys)
        self.device = device
        self.hist: list[torch.Tensor] = []        # newest last, each [N, T, C]
        self.ema = [None, None]
        self.run_sum = None                       # running sum over hist, kept incrementally

    def _stack(self, cur):
        return torch.stack([cur[k].reshape(self.T, self.C) for k in self.keys]).to(
            self.device, torch.float32)

    def push(self, cur):
        """Add cycle c. Returns per-cell stats, or None while warming up."""
        tgt = self._stack(cur)
        out = None
        if len(self.hist) >= HIST:
            out = self._measure(tgt)
        # the window mean is maintained incrementally; stacking HIST copies to
        # average them would multiply peak memory by HIST on the larger backbone
        self.run_sum = tgt.clone() if self.run_sum is None else self.run_sum + tgt
        self.hist.append(tgt)
        if len(self.hist) > HIST:
            self.run_sum -= self.hist.pop(0)
        for i, lam in enumerate(EMA_RATES):       # EMA over past only, updated after use
            self.ema[i] = tgt.clone() if self.ema[i] is None else lam * self.ema[i] + (1 - lam) * tgt
        return out

    def _measure(self, tgt):
        h = self.hist                              # h[-1] is R[c-1]
        feats = [tgt, h[-1], h[-2], h[-3], h[-4], h[-5], h[-6], h[-7],
                 self.run_sum / len(h), self.ema[0], self.ema[1]]

        # TF32 keeps only 10 mantissa bits; these are inner products of 30k terms
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        G = torch.empty(self.N, NCOL, NCOL, device=tgt.device, dtype=torch.float32)
        for i in range(NCOL):                      # pairwise, so no [N, NCOL, D] buffer
            fi = feats[i].reshape(self.N, 1, -1)
            for j in range(i, NCOL):
                v = torch.bmm(fi, feats[j].reshape(self.N, -1, 1)).reshape(self.N)
                G[:, i, j] = v
                G[:, j, i] = v
        torch.backends.cuda.matmul.allow_tf32 = prev

        tgt2 = G[:, 0, 0].clamp_min(1e-30)
        p1, p2 = h[-1], h[-2]
        # red is the axis summed over: 1 (tokens) leaves one scalar per channel,
        # 2 (channels) leaves one scalar per token.
        nl = {"chan1": self._axis1(tgt, p1, red=1), "tok1": self._axis1(tgt, p1, red=2),
              "chan2": self._axis2(tgt, p1, p2, red=1)}
        return {"gram": G[:, IU[0], IU[1]].cpu().numpy().astype(np.float32),
                **{k: (v / tgt2).clamp_min(0).sqrt().cpu().numpy().astype(np.float32)
                   for k, v in nl.items()}}

    @staticmethod
    def _axis1(tgt, p, red):
        """Oracle scalar per slice, fit with the true residual in hand."""
        num = (tgt * p).sum(red)
        den = (p * p).sum(red)
        gain = torch.where(den > 0, num * num / den.clamp_min(1e-30),
                           torch.zeros_like(num)).sum(1)
        return (tgt * tgt).sum((1, 2)) - gain

    @staticmethod
    def _axis2(tgt, p1, p2, red):
        """Oracle 2x2 solve per slice: b^T A^-1 b, with a small ridge."""
        a11 = (p1 * p1).sum(red); a22 = (p2 * p2).sum(red); a12 = (p1 * p2).sum(red)
        b1 = (tgt * p1).sum(red); b2 = (tgt * p2).sum(red)
        lam = _ridge(a11, a22)
        a11, a22 = a11 + lam, a22 + lam
        det = (a11 * a22 - a12 * a12).clamp_min(1e-30)
        gain = ((a22 * b1 * b1 - 2 * a12 * b1 * b2 + a11 * b2 * b2) / det).sum(1)
        return (tgt * tgt).sum((1, 2)) - gain


class MultiRecorder:
    """Routes a cycle's residuals to one FamilyRecorder per tensor shape."""

    def __init__(self, device):
        self.device = device
        self.groups: dict[tuple, FamilyRecorder] = {}

    def push(self, cur):
        by_shape: dict[tuple, list] = {}
        for k, v in cur.items():
            t = v.reshape(-1, v.shape[-1])
            by_shape.setdefault((t.shape[0], t.shape[1]), []).append(k)
        rows = {}
        for shape, keys in by_shape.items():
            keys = sorted(keys)
            g = self.groups.get(shape)
            if g is None or g.keys != keys:
                g = self.groups[shape] = FamilyRecorder(keys, shape, self.device)
            r = g.push(cur)
            if r is not None:
                rows[shape] = (keys, r)
        return rows

    def reset(self):
        self.groups = {}

"""The whole idea in ~15 lines, with a runnable demo.

A cross-inference cache already recomputes every block on a refresh cycle. On
that cycle you are holding TWO things at once: the residual you just computed,
R[c], and the cached residual you have been substituting, R[c-tau]. Their inner
product is exactly what the attenuation weight is made of. So the weight costs
two scalar accumulators per cell and no extra forward passes.

Run: python3 online_beta_minimal.py
"""
from __future__ import annotations
import numpy as np


class AttenuatedCache:
    """A cross-inference cache that learns its own substitution weight.

    The ONLY difference from a plain cache is the three marked lines.
    """

    def __init__(self, tau: int):
        self.tau = tau
        self.cache: dict = {}                 # cell -> residual from the last refresh
        self.num: dict = {}                   # cell -> running <R[c], R[c-tau]>
        self.den: dict = {}                   # cell -> running ||R[c-tau]||^2
        self.cycle = -1

    def beta(self, cell, staleness: int) -> float:
        """Weight for a cache that is `staleness` cycles old.

        The estimator can only ever MEASURE staleness tau -- a refresh compares
        what it just computed against the cache from the previous refresh. But
        the cache is USED at staleness 1..tau-1, and residuals decorrelate with
        lag, so beta_tau is smaller than those cycles want. Applying it directly
        over-attenuates and throws most of the benefit away (measured: +13.4%
        instead of +16.4% on Fast-WAM at tau=4).

        Correct for it by assuming the correlation decays multiplicatively,
        beta_k = rho^k, which gives beta_k = beta_tau ** (k / tau). That
        assumption is itself checkable -- see the note at the bottom of the file.
        """
        d = self.den.get(cell, 0.0)
        if d <= 0:
            return 1.0          # nothing learned yet: behave as published
        b_tau = self.num.get(cell, 0.0) / d
        return float(np.clip(b_tau ** (staleness / self.tau), 0.0, 1.5))

    def begin_cycle(self) -> None:
        self.cycle += 1
        self.refreshing = (self.cycle % self.tau == 0)

    def block(self, cell, h_in, compute):
        """Call in place of the block. `compute()` runs the real block."""
        if self.refreshing:
            out = compute()                                   # full compute
            R = out - h_in
            old = self.cache.get(cell)
            if old is not None:
                # ---- the whole trick: both vectors are in hand right now ----
                self.num[cell] = self.num.get(cell, 0.0) + float(R @ old)
                self.den[cell] = self.den.get(cell, 0.0) + float(old @ old)
                # -------------------------------------------------------------
            self.cache[cell] = R
            return out
        R = self.cache[cell]
        staleness = self.cycle % self.tau                     # 1 .. tau-1
        return h_in + self.beta(cell, staleness) * R          # attenuated reuse


# --------------------------------------------------------------------- demo
def demo(tau=4, cycles=80, dim=256, rho=0.945, seed=0):
    """Residuals that drift like the real ones: magnitude preserved, direction
    decorrelating. rho is picked so the lag-1 cosine lands near the measured 0.80."""
    rng = np.random.RandomState(seed)
    cell = ("step3", "block7")
    R = rng.randn(dim)
    truth, cache = [], AttenuatedCache(tau)
    err_unit = err_online = 0.0

    for c in range(cycles):
        # the residual this block WOULD produce, if we computed it
        R = rho * R + np.sqrt(1 - rho ** 2) * rng.randn(dim)
        truth.append(R.copy())
        h_in = rng.randn(dim) * 0.1

        cache.begin_cycle()
        out = cache.block(cell, h_in, compute=lambda: h_in + R)

        if not cache.refreshing:                 # only cached cycles have error
            err_online += np.sum((out - (h_in + R)) ** 2)
            unit = h_in + cache.cache[cell]      # what the published method emits
            err_unit += np.sum((unit - (h_in + R)) ** 2)

    # what an oracle would have used: the true lag-tau correlation
    lag = [(truth[i] @ truth[i - tau]) / (truth[i - tau] @ truth[i - tau])
           for i in range(tau, cycles)]
    print(f"  tau = {tau},  {cycles} cycles,  {cycles // tau} refreshes")
    print(f"  learned weight at lag {tau}   {cache.beta(cell, tau):.3f}"
          f"   (true {np.mean(lag):.3f})")
    for k in range(1, tau):
        tk = [(truth[i] @ truth[i-k]) / (truth[i-k] @ truth[i-k])
              for i in range(k, cycles)]
        print(f"  corrected weight at lag {k}   {cache.beta(cell, k):.3f}"
              f"   (true {np.mean(tk):.3f})")
    print(f"  error, published      {np.sqrt(err_unit):.2f}")
    print(f"  error, attenuated     {np.sqrt(err_online):.2f}"
          f"   ({100*(1-np.sqrt(err_online/err_unit)):+.1f}%)")


def warmup(tau=4, dim=256, rho=0.945, seed=1):
    """How many refreshes before the estimate is usable?"""
    rng = np.random.RandomState(seed)
    R = rng.randn(dim); cache = AttenuatedCache(tau); cell = "c"
    seen, shown = 0, []
    for c in range(9 * tau):
        R = rho * R + np.sqrt(1 - rho ** 2) * rng.randn(dim)
        h_in = rng.randn(dim) * 0.1
        cache.begin_cycle()
        if cache.refreshing and cache.cache.get(cell) is not None:
            seen += 1
        cache.block(cell, h_in, compute=lambda: h_in + R)
        if cache.refreshing and seen:
            shown.append((seen, cache.beta(cell, 1)))
    print("  refreshes -> weight at lag 1: "
          + "  ".join(f"{n}:{b:.3f}" for n, b in shown))


if __name__ == "__main__":
    print("\nthe estimate converges on the right weight, using only refresh cycles\n")
    demo()
    print()
    warmup()
    print("""
Cost: two floats per cell, updated on cycles the cache already computes.
No extra forward passes, and it re-estimates if the scene changes.

One caveat, measured rather than assumed. The beta_k = beta_tau**(k/tau)
correction assumes the correlation decays multiplicatively. That holds on both
robot policies tested (it recovers 96-97% of a full calibration pass) and FAILS
on a video diffusion model, where the correlation barely decays and the
correction is worse than using beta_tau uncorrected (68% vs 90%). If you are
unsure which regime you are in, spend one extra full-compute cycle occasionally:
two consecutive refreshes give you beta_1 directly, and comparing it against
beta_tau**(1/tau) tells you whether the decay model holds on your model.
""")

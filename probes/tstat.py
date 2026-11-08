"""Student-t confidence intervals for the small samples this study actually has.

Every closed-loop headline here is a mean over a handful of runs -- five seeds on
the frontier, nine runs at tau=8, two on the oracle ceiling. An earlier version
of the analysis put 1.96 standard errors around those means. That is the normal
quantile, which assumes the standard deviation is known rather than estimated
from the same five numbers, and at n=5 it understates the interval by 42%.

The difference is not cosmetic at these sample sizes:

    n=5   1.96 -> 2.776    interval 42% wider
    n=9   1.96 -> 2.306    interval 18% wider

No scipy in this tier -- the analysis runs on numpy alone so a reviewer needs one
install -- so the quantile is computed here by inverting the t CDF. For t > 0,

    P(|T| > t) = I_{v/(v+t^2)}(v/2, 1/2)

with I the regularised incomplete beta, evaluated by the standard continued
fraction and inverted by bisection. self_test() checks the result against
published table values; verify.sh runs it.
"""
from __future__ import annotations

import math

# published two-sided 95% critical values, for the self-test
_TABLE = {1: 12.7062, 2: 4.30265, 3: 3.18245, 4: 2.77645, 5: 2.57058,
          8: 2.30600, 10: 2.22814, 20: 2.08596, 30: 2.04227, 100: 1.98397}


def _betacf(a, b, x, itmax=300, eps=3e-16):
    """Continued fraction for the incomplete beta, by the modified Lentz method."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny: d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny: d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny: c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny: d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny: c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps: break
    return h


def betainc(a, b, x):
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                          + b * math.log1p(-x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def sf2(t, df):
    """Two-sided tail P(|T| > t) for t >= 0."""
    if t <= 0: return 1.0
    return betainc(df / 2.0, 0.5, df / (df + t * t))


def tcrit(df, alpha=0.05):
    """Two-sided critical value: t with P(|T| > t) = alpha."""
    if df <= 0: return float("inf")
    lo, hi = 0.0, 2.0
    while sf2(hi, df) > alpha:
        hi *= 2.0
        if hi > 1e6: return float("inf")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if sf2(mid, df) > alpha: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)


def mean_ci(v, alpha=0.05):
    """Mean and two-sided t interval. n=1 has no interval and says so."""
    n = len(v)
    if n == 0: return None
    m = sum(v) / n
    if n == 1: return m, float("-inf"), float("inf"), 0.0, float("inf")
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    se = sd / math.sqrt(n)
    t = tcrit(n - 1, alpha)
    return m, m - t * se, m + t * se, sd, t


def self_test():
    bad = []
    for df, ref in _TABLE.items():
        got = tcrit(df)
        if abs(got - ref) > 5e-4: bad.append((df, ref, got))
    if bad:
        for df, ref, got in bad:
            print(f"  df={df}: table {ref}, computed {got}")
        return 1
    print(f"  tcrit matches published table at {len(_TABLE)} degrees of freedom "
          f"(df=1..100), max error < 5e-4")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(self_test())

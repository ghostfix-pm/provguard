"""Exact binomial confidence intervals, so small samples are reported honestly.

Pure Python (no SciPy): Clopper-Pearson bounds found by bisection on the
binomial CDF. With 0 successes in 48 trials the one-sided 95% upper bound is
about 6.1%, the figure quoted in the README.
"""

from __future__ import annotations

from math import comb


def _cdf(k: int, n: int, p: float) -> float:
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def _bisect(f, lo: float = 0.0, hi: float = 1.0, iters: int = 60) -> float:
    for _ in range(iters):
        mid = (lo + hi) / 2
        if f(mid):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def clopper_pearson(k: int, n: int, confidence: float = 0.95, one_sided: bool = False) -> tuple[float, float]:
    """Exact interval for a binomial proportion ``k / n``."""
    if n <= 0 or not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n and n > 0")
    alpha = 1 - confidence
    tail = alpha if one_sided else alpha / 2
    lower = 0.0 if k == 0 else _bisect(lambda p: 1 - _cdf(k - 1, n, p) < tail)
    upper = 1.0 if k == n else _bisect(lambda p: _cdf(k, n, p) > tail)
    return lower, upper


def fmt_rate(k: int, n: int) -> str:
    """``"97/105 (92.4%, 95% CI 85.5-96.7%)"``."""
    if n == 0:
        return "0/0"
    lo, hi = clopper_pearson(k, n)
    return f"{k}/{n} ({k / n:.1%}, 95% CI {lo:.1%}-{hi:.1%})"

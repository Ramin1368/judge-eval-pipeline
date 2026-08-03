from __future__ import annotations

"""Statistical primitives with a defensible primary CI method.

Design decisions:

* Bootstrap CI is BCa (bias-corrected and accelerated) by default. Percentile
  bootstrap is retained as a cross-check because reviewers expect to see both
  and disagreement between the two is itself diagnostic. The plain percentile
  method under-covers when the sampling distribution is skewed, which happens
  with binary preference data at small n, so it must not be the primary CI.
* Bootstrap CIs are also produced for Cohen's kappa. A point estimate of
  kappa on ten items is essentially uninformative, and the trust gate at 0.4
  is only defensible if the CI is reported alongside it. The pipeline flags
  "trustworthy_with_reserve" when the point estimate clears the threshold but
  the lower CI bound does not.
* Percentile indices use ``ceil`` for the lower bound and ``floor`` for the
  upper bound, so the returned interval is at least the nominal (1-alpha)
  wide. The previous implementation had an off-by-one on both ends.
* The seed is a function argument for reproducibility. Every downstream
  consumer threads it through.
"""

import math
import random
from typing import Callable, Sequence


def win_rate(scores: Sequence[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.5


def _percentile_indices(n_boot: int, alpha: float) -> tuple[int, int]:
    lo = max(0, int(math.ceil((alpha / 2.0) * n_boot)) - 1)
    hi = min(n_boot - 1, int(math.floor((1.0 - alpha / 2.0) * n_boot)) - 1)
    if hi < lo:
        hi = lo
    return lo, hi


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    # Beasley-Springer-Moro approximation, sufficient for CI endpoints.
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    a = [-39.6968302866538, 220.946098424521, -275.928510446969,
         138.357751867269, -30.6647980661472, 2.50662827745924]
    b = [-54.4760987982241, 161.585836858041, -155.698979859887,
         66.8013118877197, -13.2806815528857]
    c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184,
         -2.54973253934373, 4.37466414146497, 2.93816398269878]
    d = [0.00778469570904146, 0.32246712907004, 2.445134137143,
         3.75440866190742]
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


def percentile_bootstrap_ci(
    scores: Sequence[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 12345,
    statistic: Callable[[Sequence[float]], float] = win_rate,
) -> tuple[float, float]:
    if not scores:
        return (0.0, 1.0)
    rng = random.Random(seed)
    n = len(scores)
    replicates: list[float] = []
    for _ in range(n_boot):
        resample = [scores[rng.randrange(n)] for _ in range(n)]
        replicates.append(statistic(resample))
    replicates.sort()
    lo_i, hi_i = _percentile_indices(n_boot, alpha)
    return replicates[lo_i], replicates[hi_i]


def bca_bootstrap_ci(
    values: Sequence,
    statistic: Callable[[Sequence], float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 12345,
) -> tuple[float, float]:
    """BCa (bias-corrected and accelerated) bootstrap CI.

    Falls back to the percentile CI if the observed statistic is degenerate
    (all replicates equal) or if the acceleration term diverges, which happens
    when the jackknife influence values are all zero. Falling back is
    correct behavior, not a workaround: BCa's correction is undefined in
    those cases and the percentile interval is the appropriate limit.
    """
    if not values:
        return (0.0, 1.0)
    n = len(values)
    rng = random.Random(seed)
    theta_hat = statistic(values)

    replicates: list[float] = []
    for _ in range(n_boot):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        replicates.append(statistic(resample))
    replicates_sorted = sorted(replicates)

    less = sum(1 for r in replicates if r < theta_hat)
    equal = sum(1 for r in replicates if r == theta_hat)
    prop = (less + 0.5 * equal) / n_boot
    if prop <= 0.0 or prop >= 1.0:
        lo_i, hi_i = _percentile_indices(n_boot, alpha)
        return replicates_sorted[lo_i], replicates_sorted[hi_i]
    z0 = _norm_ppf(prop)

    # Jackknife acceleration
    jack: list[float] = []
    for i in range(n):
        loo = list(values[:i]) + list(values[i + 1:])
        jack.append(statistic(loo))
    jack_mean = sum(jack) / n
    num = sum((jack_mean - j) ** 3 for j in jack)
    den = 6.0 * (sum((jack_mean - j) ** 2 for j in jack) ** 1.5)
    if den == 0.0:
        lo_i, hi_i = _percentile_indices(n_boot, alpha)
        return replicates_sorted[lo_i], replicates_sorted[hi_i]
    a_hat = num / den

    z_lo = _norm_ppf(alpha / 2.0)
    z_hi = _norm_ppf(1.0 - alpha / 2.0)
    alpha_lo = _norm_cdf(z0 + (z0 + z_lo) / (1.0 - a_hat * (z0 + z_lo)))
    alpha_hi = _norm_cdf(z0 + (z0 + z_hi) / (1.0 - a_hat * (z0 + z_hi)))
    lo_i = min(n_boot - 1, max(0, int(math.floor(alpha_lo * n_boot))))
    hi_i = min(n_boot - 1, max(0, int(math.floor(alpha_hi * n_boot))))
    return replicates_sorted[lo_i], replicates_sorted[hi_i]


# Backwards-compatible name; now uses BCa by default with percentile fallback.
def bootstrap_ci(
    scores: Sequence[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 12345,
) -> tuple[float, float]:
    return bca_bootstrap_ci(scores, win_rate, n_boot=n_boot, alpha=alpha, seed=seed)


def wilson_ci(wins: int, decisive: int, z: float = 1.96) -> tuple[float, float]:
    if decisive == 0:
        return (0.0, 1.0)
    p = wins / decisive
    denom = 1 + z ** 2 / decisive
    center = (p + z ** 2 / (2 * decisive)) / denom
    half = (z * math.sqrt(p * (1 - p) / decisive + z ** 2 / (4 * decisive ** 2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _log_binom_coeff(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def sign_test_p(wins_b: int, wins_a: int) -> float:
    n = wins_a + wins_b
    if n == 0:
        return 1.0
    k = min(wins_a, wins_b)

    def pmf(i: int) -> float:
        return math.exp(_log_binom_coeff(n, i) + n * math.log(0.5))

    tail = sum(pmf(i) for i in range(0, k + 1))
    return min(1.0, 2 * tail)


def minimum_detectable_effect(n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Two-sided MDE for a proportion vs 0.5, expressed as an absolute shift."""
    if n <= 0:
        return 1.0
    z_alpha = _norm_ppf(1.0 - alpha / 2.0)
    z_power = _norm_ppf(power)
    return (z_alpha + z_power) * math.sqrt(0.25 / n)


def kappa_bootstrap_ci(
    labels_true: Sequence[str],
    labels_pred: Sequence[str],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 12345,
) -> tuple[float, float]:
    """Bootstrap CI on Cohen's kappa by resampling item indices with replacement."""
    from .calibration import cohen_kappa

    n = len(labels_true)
    if n == 0:
        return (0.0, 1.0)
    idxs = list(range(n))
    rng = random.Random(seed)
    reps: list[float] = []
    for _ in range(n_boot):
        pick = [idxs[rng.randrange(n)] for _ in range(n)]
        t = [labels_true[i] for i in pick]
        p = [labels_pred[i] for i in pick]
        reps.append(cohen_kappa(t, p))
    reps.sort()
    lo_i, hi_i = _percentile_indices(n_boot, alpha)
    return reps[lo_i], reps[hi_i]


def summarize_scores(scores: Sequence[float], seed: int = 12345) -> dict:
    n = len(scores)
    wins_b = sum(1 for s in scores if s == 1.0)
    wins_a = sum(1 for s in scores if s == 0.0)
    ties = sum(1 for s in scores if s == 0.5)
    bca = bca_bootstrap_ci(list(scores), win_rate, seed=seed)
    pct = percentile_bootstrap_ci(list(scores), seed=seed)
    return {
        "n": n,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "win_rate_b": win_rate(scores),
        "bootstrap_ci": bca,
        "bca_ci": bca,
        "percentile_ci": pct,
        "wilson_ci": wilson_ci(wins_b, wins_a + wins_b),
        "p_value": sign_test_p(wins_b, wins_a),
        "mde": minimum_detectable_effect(n),
    }

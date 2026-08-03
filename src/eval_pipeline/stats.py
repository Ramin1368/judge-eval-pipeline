from __future__ import annotations

import math
import random


def win_rate(scores: list[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.5


def bootstrap_ci(scores: list[float], n_boot: int = 10000, alpha: float = 0.05, seed: int = 12345) -> tuple[float, float]:
    if not scores:
        return (0.0, 1.0)
    rng = random.Random(seed)
    n = len(scores)
    means = []
    for _ in range(n_boot):
        resample = [scores[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return (lo, hi)


def wilson_ci(wins: int, decisive: int, z: float = 1.96) -> tuple[float, float]:
    if decisive == 0:
        return (0.0, 1.0)
    p = wins / decisive
    denom = 1 + z**2 / decisive
    center = (p + z**2 / (2 * decisive)) / denom
    half = (z * math.sqrt(p * (1 - p) / decisive + z**2 / (4 * decisive**2))) / denom
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
    if n <= 0:
        return 1.0
    z_alpha = 1.96
    z_power = 0.84
    return (z_alpha + z_power) / (2 * math.sqrt(n))


def summarize_scores(scores: list[float], seed: int = 12345) -> dict:
    n = len(scores)
    wins_b = sum(1 for s in scores if s == 1.0)
    wins_a = sum(1 for s in scores if s == 0.0)
    ties = sum(1 for s in scores if s == 0.5)
    return {
        "n": n,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "win_rate_b": win_rate(scores),
        "bootstrap_ci": bootstrap_ci(scores, seed=seed),
        "wilson_ci": wilson_ci(wins_b, wins_a + wins_b),
        "p_value": sign_test_p(wins_b, wins_a),
        "mde": minimum_detectable_effect(n),
    }

from __future__ import annotations
import math
import random
from eval_pipeline.stats import bca_bootstrap_ci, kappa_bootstrap_ci, percentile_bootstrap_ci, win_rate

def test_bca_ci_covers_true_mean_at_nominal_rate():
    rng = random.Random(1)
    true_p = 0.65
    covers = 0
    trials = 60
    for _ in range(trials):
        sample = [1.0 if rng.random() < true_p else 0.0 for _ in range(100)]
        lo, hi = bca_bootstrap_ci(sample, win_rate, n_boot=800, seed=rng.randrange(10 ** 9))
        if lo <= true_p <= hi:
            covers += 1
    coverage = covers / trials
    assert coverage >= 0.8, f'BCa coverage {coverage} at n=100 was too low'

def test_bca_and_percentile_agree_on_symmetric_data():
    rng = random.Random(2)
    sample = [rng.gauss(0, 1) for _ in range(200)]

    def mean_stat(xs):
        return sum(xs) / len(xs)
    lo_b, hi_b = bca_bootstrap_ci(sample, mean_stat, n_boot=800, seed=42)
    lo_p, hi_p = percentile_bootstrap_ci(sample, n_boot=800, seed=42, statistic=mean_stat)
    assert abs(lo_b - lo_p) < 0.15
    assert abs(hi_b - hi_p) < 0.15

def test_kappa_bootstrap_ci_wider_at_small_n():
    rng = random.Random(3)

    def sample_labels(n: int):
        true_labels = [rng.choice(['A', 'B', 'tie']) for _ in range(n)]
        pred = [t if rng.random() < 0.75 else rng.choice(['A', 'B', 'tie']) for t in true_labels]
        return (true_labels, pred)
    t_small, p_small = sample_labels(20)
    t_large, p_large = sample_labels(400)
    lo_s, hi_s = kappa_bootstrap_ci(t_small, p_small, n_boot=600, seed=1)
    lo_l, hi_l = kappa_bootstrap_ci(t_large, p_large, n_boot=600, seed=1)
    width_small = hi_s - lo_s
    width_large = hi_l - lo_l
    assert width_small > width_large, f'kappa CI did not shrink with n: small={width_small}, large={width_large}'

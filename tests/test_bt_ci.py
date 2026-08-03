from __future__ import annotations
from eval_pipeline.bradley_terry import bt_bootstrap_ci

def test_bt_bootstrap_ci_narrows_with_n():
    scores_small = [1.0] * 6 + [0.0] * 4
    scores_large = [1.0] * 60 + [0.0] * 40
    small = bt_bootstrap_ci(scores_small, 'a', 'b', n_boot=500, seed=1)
    large = bt_bootstrap_ci(scores_large, 'a', 'b', n_boot=500, seed=1)
    w_small = small['win_prob_b_ci'][1] - small['win_prob_b_ci'][0]
    w_large = large['win_prob_b_ci'][1] - large['win_prob_b_ci'][0]
    assert w_small > w_large

def test_bt_bootstrap_covers_point_estimate():
    scores = [1.0] * 65 + [0.0] * 35
    ci = bt_bootstrap_ci(scores, 'a', 'b', n_boot=800, seed=2)
    lo, hi = ci['win_prob_b_ci']
    assert lo <= 0.65 <= hi

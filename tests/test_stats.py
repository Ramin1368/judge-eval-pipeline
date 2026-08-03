from eval_pipeline.stats import win_rate, bootstrap_ci, wilson_ci, sign_test_p, minimum_detectable_effect, summarize_scores

def test_win_rate_counts_ties_as_half():
    assert win_rate([1.0, 0.0]) == 0.5
    assert win_rate([1.0, 1.0, 0.5, 0.0]) == 0.625

def test_bootstrap_ci_is_deterministic_and_ordered():
    scores = [1.0] * 70 + [0.0] * 30
    lo, hi = bootstrap_ci(scores, seed=1)
    lo2, hi2 = bootstrap_ci(scores, seed=1)
    assert (lo, hi) == (lo2, hi2)
    assert lo < 0.7 < hi
    assert 0.0 <= lo <= hi <= 1.0

def test_bootstrap_ci_excludes_half_for_strong_effect():
    scores = [1.0] * 90 + [0.0] * 10
    lo, hi = bootstrap_ci(scores, seed=3)
    assert lo > 0.5

def test_wilson_ci_matches_known_value():
    lo, hi = wilson_ci(70, 100)
    assert abs(lo - 0.603) < 0.01
    assert abs(hi - 0.782) < 0.01

def test_sign_test_symmetric_is_one():
    assert sign_test_p(50, 50) == 1.0

def test_sign_test_extreme_is_significant():
    assert sign_test_p(95, 5) < 0.001

def test_mde_shrinks_with_n():
    assert minimum_detectable_effect(25) > minimum_detectable_effect(400)

def test_summarize_scores_bundle():
    scores = [1.0] * 60 + [0.0] * 30 + [0.5] * 10
    s = summarize_scores(scores, seed=1)
    assert s['n'] == 100
    assert s['wins_b'] == 60 and s['wins_a'] == 30 and (s['ties'] == 10)
    assert abs(s['win_rate_b'] - 0.65) < 1e-09
    assert s['bootstrap_ci'][0] <= s['win_rate_b'] <= s['bootstrap_ci'][1]

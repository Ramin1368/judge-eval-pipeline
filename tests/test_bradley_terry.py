from eval_pipeline.bradley_terry import fit_bradley_terry, win_probability, rank_policies


def test_recovers_transitive_ranking():
    win_counts = {
        ("strong", "mid"): 8, ("mid", "strong"): 2,
        ("mid", "weak"): 8, ("weak", "mid"): 2,
        ("strong", "weak"): 9, ("weak", "strong"): 1,
    }
    strengths = fit_bradley_terry(win_counts)
    order = [p for p, _ in rank_policies(strengths)]
    assert order == ["strong", "mid", "weak"]


def test_win_probability_tracks_win_rate_for_two_players():
    win_counts = {("b", "a"): 70, ("a", "b"): 30}
    strengths = fit_bradley_terry(win_counts)
    p = win_probability(strengths, "b", "a")
    assert abs(p - 0.70) < 0.03


def test_equal_records_give_even_probability():
    win_counts = {("a", "b"): 25, ("b", "a"): 25}
    strengths = fit_bradley_terry(win_counts)
    assert abs(win_probability(strengths, "a", "b") - 0.5) < 1e-6

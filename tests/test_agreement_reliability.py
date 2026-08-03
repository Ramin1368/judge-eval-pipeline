from eval_pipeline.schemas import Preference, PreferenceExample
from eval_pipeline.agreement import fleiss_kappa
from eval_pipeline.reliability import reliability_curve

def _ex(prompt, a, b, pref, ann):
    return PreferenceExample(prompt, a, b, Preference.parse(pref), annotator_id=ann)

def test_fleiss_perfect_agreement():
    exs = [_ex('p1', 'aaa', 'bbb', 'A', 'r1'), _ex('p1', 'aaa', 'bbb', 'A', 'r2'), _ex('p2', 'ccc', 'ddd', 'B', 'r1'), _ex('p2', 'ccc', 'ddd', 'B', 'r2')]
    k = fleiss_kappa(exs)
    assert k is not None
    assert k > 0.99

def test_fleiss_none_without_repeats():
    exs = [_ex('p1', 'aaa', 'bbb', 'A', 'r1')]
    assert fleiss_kappa(exs) is None

def test_reliability_perfectly_calibrated_has_low_ece():
    confidences = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1]
    correct = [True, True, True, True, True, True, True, True, True, False]
    bins, ece = reliability_curve(confidences, correct, n_bins=5)
    assert ece < 0.15

def test_reliability_overconfident_has_high_ece():
    confidences = [1.0] * 10
    correct = [True] * 5 + [False] * 5
    bins, ece = reliability_curve(confidences, correct, n_bins=5)
    assert ece > 0.4

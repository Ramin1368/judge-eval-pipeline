from __future__ import annotations

from eval_pipeline.agreement import fleiss_kappa
from eval_pipeline.schemas import Preference, PreferenceExample


def _mk(prompt, a, b, pref, aid):
    return PreferenceExample(prompt=prompt, response_a=a, response_b=b, preferred=pref, annotator_id=aid)


def test_fleiss_handles_variable_rater_counts_per_item():
    # Item 1: 3 raters, all agree on A
    # Item 2: 5 raters, 4 say B, 1 tie
    # Item 3: 2 raters, both A
    examples = [
        _mk("p1", "x", "y", Preference.A, "r1"),
        _mk("p1", "x", "y", Preference.A, "r2"),
        _mk("p1", "x", "y", Preference.A, "r3"),
        _mk("p2", "u", "v", Preference.B, "r1"),
        _mk("p2", "u", "v", Preference.B, "r2"),
        _mk("p2", "u", "v", Preference.B, "r3"),
        _mk("p2", "u", "v", Preference.B, "r4"),
        _mk("p2", "u", "v", Preference.TIE, "r5"),
        _mk("p3", "m", "n", Preference.A, "r1"),
        _mk("p3", "m", "n", Preference.A, "r2"),
    ]
    kappa = fleiss_kappa(examples)
    # Substantial agreement, positive kappa
    assert kappa is not None
    assert kappa > 0.3


def test_fleiss_returns_none_when_no_multi_rater_items():
    examples = [
        _mk("p1", "x", "y", Preference.A, "r1"),
        _mk("p2", "u", "v", Preference.B, "r1"),
    ]
    assert fleiss_kappa(examples) is None

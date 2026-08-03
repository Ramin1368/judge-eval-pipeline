from eval_pipeline.schemas import Preference, PreferenceExample
from eval_pipeline.validation import validate_and_dedupe


def _ex(prompt, a, b, pref, ann=None):
    return PreferenceExample(prompt, a, b, Preference.parse(pref), annotator_id=ann)


def test_exact_duplicates_collapse_to_one():
    exs = [_ex("p", "aaa", "bbb", "A"), _ex("p", "aaa", "bbb", "A")]
    clean, rep = validate_and_dedupe(exs)
    assert len(clean) == 1
    assert rep.duplicate_groups == 1


def test_contradiction_resolved_by_majority():
    exs = [
        _ex("p", "aaa", "bbb", "A"),
        _ex("p", "aaa", "bbb", "A"),
        _ex("p", "aaa", "bbb", "B"),
    ]
    clean, rep = validate_and_dedupe(exs)
    assert len(clean) == 1
    assert clean[0].preferred is Preference.A
    assert rep.contradictory_groups == 1


def test_order_swap_is_recognized_as_same_comparison():
    exs = [
        _ex("p", "aaa", "bbb", "A"),
        _ex("p", "bbb", "aaa", "B"),
    ]
    clean, rep = validate_and_dedupe(exs)
    assert len(clean) == 1
    assert rep.order_swapped_groups == 1
    assert rep.contradictory_groups == 0


def test_unresolvable_split_is_quarantined():
    exs = [
        _ex("p", "aaa", "bbb", "A"),
        _ex("p", "aaa", "bbb", "B"),
    ]
    clean, rep = validate_and_dedupe(exs)
    assert rep.n_quarantined == 1
    assert len(clean) == 0
    assert any("quarantined" in m for m in rep.messages)


def test_self_consistency_computed():
    exs = [
        _ex("p", "aaa", "bbb", "A"),
        _ex("p", "aaa", "bbb", "A"),
        _ex("p", "aaa", "bbb", "B"),
    ]
    _, rep = validate_and_dedupe(exs)
    assert rep.human_self_consistency is not None
    assert abs(rep.human_self_consistency - (2 / 3)) < 1e-9

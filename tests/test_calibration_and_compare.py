from eval_pipeline.schemas import Preference, PreferenceExample, JudgeVerdict
from eval_pipeline.calibration import cohen_kappa, calibrate_judge
from eval_pipeline.judges.base import Judge
from eval_pipeline.judges import HeuristicJudge
from eval_pipeline.policy_compare import compare_policies


def test_cohen_kappa_perfect_and_chance():
    assert abs(cohen_kappa(["A", "B", "A"], ["A", "B", "A"]) - 1.0) < 1e-9
    k = cohen_kappa(["A", "B", "A", "B"], ["A", "A", "A", "A"])
    assert k <= 0.0 + 1e-9


class _PositionBiasedJudge(Judge):
    name = "biased"

    def _decide(self, prompt, response_a, response_b):
        return JudgeVerdict(Preference.A, confidence=1.0)


def test_position_bias_detected_and_neutralized():
    j = _PositionBiasedJudge()
    v = j.judge("p", "x", "y")
    assert v.preferred is Preference.TIE
    assert v.position_unstable is True


def test_heuristic_prefers_relevant_over_filler():
    j = HeuristicJudge()
    good = "A load balancer distributes incoming traffic across servers."
    bad = "It depends and there are many factors to consider."
    v = j.judge("Explain what a load balancer does.", good, bad)
    assert v.preferred is Preference.A


def test_calibration_report_shapes():
    j = HeuristicJudge()
    exs = [
        PreferenceExample("Explain load balancing.",
                          "A load balancer spreads traffic across servers.",
                          "It depends on many factors.", Preference.A),
        PreferenceExample("What is Kubernetes?",
                          "Kubernetes orchestrates containers across nodes.",
                          "Hard to say really.", Preference.A),
    ]
    rep = calibrate_judge(j, exs)
    assert rep.n == 2
    assert 0.0 <= rep.accuracy <= 1.0
    assert -1.0 <= rep.cohen_kappa <= 1.0
    assert 0.0 <= rep.position_bias_rate <= 1.0
    assert rep.expected_calibration_error is not None


def test_compare_policies_recovers_better_policy():
    j = HeuristicJudge()
    rows = []
    for i in range(20):
        rows.append({
            "prompt": f"Explain topic {i} clearly.",
            "policy_a": "It depends and there are many factors to consider.",
            "policy_b": f"Topic {i} works by doing X, with a concrete relevant example about topic {i}.",
        })
    res = compare_policies(j, rows, "policy_a", "policy_b", seed=1)
    assert res.winner == "policy_b"
    assert res.win_rate_b > 0.5
    assert res.ci_low <= res.win_rate_b <= res.ci_high

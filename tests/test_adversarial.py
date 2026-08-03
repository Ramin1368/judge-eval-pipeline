from __future__ import annotations
from eval_pipeline.adversarial import apply_policy, sycophancy_policy, verbosity_padding_policy
from eval_pipeline.judges import HeuristicJudge
from eval_pipeline.policy_compare import compare_policies

def _synth_rows(n: int=20) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append({'prompt': f'Explain how load balancers manage traffic (case {i}).', 'policy_a': 'A load balancer distributes incoming traffic across healthy backends and removes failed hosts based on health checks so users see steady latency.', 'policy_b': 'A load balancer distributes incoming traffic across healthy backends and removes failed hosts based on health checks so users see steady latency.'})
    return rows

def test_verbosity_padding_is_caught_by_length_control():
    rows = _synth_rows()
    attacked = apply_policy(rows, 'policy_b', verbosity_padding_policy())
    judge = HeuristicJudge()
    result = compare_policies(judge, attacked, 'policy_a', 'policy_b')
    length_gap_warning = any(('length' in n.lower() for n in result.notes))
    length_control_dropped = result.length_controlled_win_rate_b is None
    assert length_gap_warning or length_control_dropped, f'verbosity padding was not caught: raw={result.win_rate_b}, length_controlled={result.length_controlled_win_rate_b}, notes={result.notes}'

def test_sycophancy_exploits_heuristic_but_length_control_is_available():
    rows = _synth_rows()
    attacked = apply_policy(rows, 'policy_b', sycophancy_policy())
    judge = HeuristicJudge()
    result = compare_policies(judge, attacked, 'policy_a', 'policy_b')
    assert result.win_rate_b >= 0.7, f'expected heuristic to be exploited by sycophancy prefix, got {result.win_rate_b}'
    assert result.length_controlled_win_rate_b is not None, 'length-controlled win rate should be defined even under sycophancy attack'

def test_sycophancy_prefix_has_low_position_bias_because_of_both_order_averaging():
    from eval_pipeline.calibration import calibrate_judge
    from eval_pipeline.schemas import Preference, PreferenceExample
    rows = _synth_rows(n=10)
    attacker = sycophancy_policy()
    examples: list[PreferenceExample] = []
    for i, r in enumerate(rows):
        a = r['policy_a']
        b = attacker(r['prompt'], r['policy_b'])
        examples.append(PreferenceExample(prompt=r['prompt'], response_a=a, response_b=b, preferred=Preference.TIE, example_id=f'a_{i}'))
    judge = HeuristicJudge()
    report = calibrate_judge(judge, examples, kappa_ci_boot=200)
    assert report.position_bias_rate <= 0.15, f'both-order averaging failed under sycophancy: bias={report.position_bias_rate}'

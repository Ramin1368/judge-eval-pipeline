from __future__ import annotations
from typing import Any
from fastapi import FastAPI
from pydantic import BaseModel
from eval_pipeline.schemas import Preference, PreferenceExample
from eval_pipeline.validation import validate_and_dedupe
from eval_pipeline.calibration import calibrate_judge
from eval_pipeline.policy_compare import compare_policies
from eval_pipeline.report import build_report
from eval_pipeline.judges import build_judge
app = FastAPI(title='Judge Calibration and Policy Eval', version='1.0.0')

class LabeledItem(BaseModel):
    prompt: str
    response_a: str
    response_b: str
    preferred: str
    annotator_id: str | None = None

class CalibrateRequest(BaseModel):
    judge: str = 'heuristic'
    examples: list[LabeledItem]

class CompareRequest(BaseModel):
    judge: str = 'heuristic'
    policy_a: str = 'policy_a'
    policy_b: str = 'policy_b'
    labeled: list[LabeledItem]
    policy_outputs: list[dict[str, Any]]

@app.get('/health')
def health() -> dict:
    return {'status': 'ok'}

def _to_examples(items: list[LabeledItem]) -> list[PreferenceExample]:
    out = []
    for it in items:
        out.append(PreferenceExample(prompt=it.prompt, response_a=it.response_a, response_b=it.response_b, preferred=Preference.parse(it.preferred), annotator_id=it.annotator_id))
    return out

@app.post('/calibrate')
def calibrate(req: CalibrateRequest) -> dict:
    clean, validation = validate_and_dedupe(_to_examples(req.examples))
    judge = build_judge(req.judge)
    rep = calibrate_judge(judge, clean)
    return {'judge': judge.name, 'n': rep.n, 'accuracy': rep.accuracy, 'cohen_kappa': rep.cohen_kappa, 'position_bias_rate': rep.position_bias_rate, 'expected_calibration_error': rep.expected_calibration_error, 'trustworthy': rep.trustworthy, 'notes': rep.notes, 'validation': {'n_input': validation.n_input, 'n_clean': validation.n_clean, 'n_quarantined': validation.n_quarantined, 'human_self_consistency': validation.human_self_consistency, 'fleiss_kappa': validation.fleiss_kappa}}

@app.post('/compare')
def compare(req: CompareRequest) -> dict:
    clean, validation = validate_and_dedupe(_to_examples(req.labeled))
    judge = build_judge(req.judge)
    calibration = calibrate_judge(judge, clean)
    comparison = compare_policies(judge, req.policy_outputs, req.policy_a, req.policy_b)
    report_md = build_report(calibration, comparison, validation, judge_name=judge.name)
    return {'winner': comparison.winner, 'win_rate_b': comparison.win_rate_b, 'ci': [comparison.ci_low, comparison.ci_high], 'p_value': comparison.p_value, 'judge_trustworthy': calibration.trustworthy, 'cohen_kappa': calibration.cohen_kappa, 'report_markdown': report_md}

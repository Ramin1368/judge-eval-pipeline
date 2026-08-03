from __future__ import annotations
from .calibration import calibrate_judge
from .judges import build_judge
from .schemas import PreferenceExample

def compare_judges(examples: list[PreferenceExample], kinds: list[str]) -> list[dict]:
    rows = []
    for kind in kinds:
        judge = build_judge(kind)
        rep = calibrate_judge(judge, examples)
        rows.append({'judge': judge.name, 'n': rep.n, 'accuracy': rep.accuracy, 'cohen_kappa': rep.cohen_kappa, 'position_bias_rate': rep.position_bias_rate, 'expected_calibration_error': rep.expected_calibration_error, 'fallback_rate': rep.fallback_rate, 'trustworthy': rep.trustworthy})
    return rows

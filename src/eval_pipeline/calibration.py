from __future__ import annotations
from collections import Counter
from statistics import mean
from .reliability import reliability_curve
from .schemas import CalibrationReport, Preference, PreferenceExample
from .judges.base import Judge
KAPPA_TRUST_THRESHOLD = 0.4
POSITION_BIAS_MAX = 0.15

def cohen_kappa(labels_true: list[str], labels_pred: list[str]) -> float:
    n = len(labels_true)
    if n == 0:
        return 0.0
    cats = sorted(set(labels_true) | set(labels_pred))
    observed = sum((1 for t, p in zip(labels_true, labels_pred) if t == p)) / n
    ct = Counter(labels_true)
    cp = Counter(labels_pred)
    expected = sum((ct[c] / n * (cp[c] / n) for c in cats))
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)

def calibrate_judge(judge: Judge, examples: list[PreferenceExample], slice_fn=None, kappa_ci_boot: int=2000, seed: int=12345) -> CalibrationReport:
    human: list[str] = []
    pred: list[str] = []
    position_flags: list[bool] = []
    confidences: list[float] = []
    correct_flags: list[bool] = []
    slices: dict[str, list[bool]] = {}
    for ex in examples:
        verdict = judge.judge(ex.prompt, ex.response_a, ex.response_b)
        human.append(ex.preferred.value)
        pred.append(verdict.preferred.value)
        position_flags.append(verdict.position_unstable)
        is_correct = verdict.preferred.value == ex.preferred.value
        confidences.append(verdict.confidence)
        correct_flags.append(is_correct)
        bucket = slice_fn(ex) if slice_fn else _default_slice(ex)
        slices.setdefault(bucket, []).append(is_correct)
    n = len(examples)
    accuracy = mean((1.0 if h == p else 0.0 for h, p in zip(human, pred))) if n else 0.0
    kappa = cohen_kappa(human, pred)
    position_bias = mean((1.0 if f else 0.0 for f in position_flags)) if n else 0.0
    tie_human = human.count(Preference.TIE.value) / n if n else 0.0
    tie_judge = pred.count(Preference.TIE.value) / n if n else 0.0
    per_slice = {k: {'n': len(v), 'accuracy': mean(v) if v else 0.0} for k, v in slices.items()}
    bins, ece = reliability_curve(confidences, correct_flags)
    fallback_rate = getattr(judge, 'fallback_rate', lambda: None)()
    kappa_lo = kappa_hi = None
    if n >= 2:
        from .stats import kappa_bootstrap_ci
        kappa_lo, kappa_hi = kappa_bootstrap_ci(human, pred, n_boot=kappa_ci_boot, seed=seed)
    notes: list[str] = []
    trustworthy = kappa >= KAPPA_TRUST_THRESHOLD and position_bias <= POSITION_BIAS_MAX
    trustworthy_with_reserve = trustworthy and kappa_lo is not None and (kappa_lo < KAPPA_TRUST_THRESHOLD)
    if kappa < KAPPA_TRUST_THRESHOLD:
        notes.append(f'kappa {kappa:.3f} below {KAPPA_TRUST_THRESHOLD}, judge agreement with humans is weak, treat policy verdicts as low confidence')
    elif trustworthy_with_reserve:
        notes.append(f"kappa point estimate {kappa:.3f} clears the {KAPPA_TRUST_THRESHOLD} gate, but its 95 percent lower bound is {kappa_lo:.3f}, so agreement may fall below the gate on a different sample; the trust flag is 'trustworthy with reserve'")
    if position_bias > POSITION_BIAS_MAX:
        notes.append(f'position bias rate {position_bias:.3f} above {POSITION_BIAS_MAX}, judge is order sensitive, results rely on both order averaging')
    if tie_judge > 0.6:
        notes.append('judge ties on a majority of items, it may be under discriminating')
    return CalibrationReport(n=n, accuracy=accuracy, cohen_kappa=kappa, position_bias_rate=position_bias, tie_rate_human=tie_human, tie_rate_judge=tie_judge, per_slice=per_slice, expected_calibration_error=ece, reliability_bins=[b.__dict__ for b in bins], fallback_rate=fallback_rate, trustworthy=trustworthy, kappa_ci_low=kappa_lo, kappa_ci_high=kappa_hi, trustworthy_with_reserve=trustworthy_with_reserve, notes=notes)

def _default_slice(ex: PreferenceExample) -> str:
    return 'human_tie' if ex.preferred is Preference.TIE else 'human_decisive'

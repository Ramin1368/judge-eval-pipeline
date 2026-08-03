from __future__ import annotations
from .bradley_terry import bt_bootstrap_ci, fit_bradley_terry, win_probability
from .schemas import Preference, PolicyComparisonResult
from .judges.base import Judge
from .stats import minimum_detectable_effect, summarize_scores

def _pair_outputs(rows: list[dict], policy_a: str, policy_b: str) -> list[tuple[str, str, str]]:
    if rows and policy_a in rows[0] and (policy_b in rows[0]):
        return [(r['prompt'], r[policy_a], r[policy_b]) for r in rows]
    by_prompt: dict[str, dict[str, str]] = {}
    for r in rows:
        by_prompt.setdefault(r['prompt'], {})[r['policy']] = r['response']
    triples = []
    for prompt, d in by_prompt.items():
        if policy_a in d and policy_b in d:
            triples.append((prompt, d[policy_a], d[policy_b]))
    return triples

def _score(verdict_pref: Preference) -> float:
    return {Preference.A: 0.0, Preference.TIE: 0.5, Preference.B: 1.0}[verdict_pref]

def compare_policies(judge: Judge, rows: list[dict], policy_a: str, policy_b: str, seed: int=12345, length_parity_ratio: float=1.5, bt_boot: int=2000) -> PolicyComparisonResult:
    triples = _pair_outputs(rows, policy_a, policy_b)
    if not triples:
        raise ValueError(f'no shared prompts found for policies {policy_a!r} and {policy_b!r}')
    scores: list[float] = []
    parity_scores: list[float] = []
    for prompt, ra, rb in triples:
        verdict = judge.judge(prompt, ra, rb)
        s = _score(verdict.preferred)
        scores.append(s)
        la, lb = (max(1, len(ra.split())), max(1, len(rb.split())))
        if max(la, lb) / min(la, lb) <= length_parity_ratio:
            parity_scores.append(s)
    win_counts = {(policy_b, policy_a): sum((1.0 if s == 1.0 else 0.5 if s == 0.5 else 0.0 for s in scores)), (policy_a, policy_b): sum((1.0 if s == 0.0 else 0.5 if s == 0.5 else 0.0 for s in scores))}
    bt = fit_bradley_terry(win_counts)
    bt_prob_b = win_probability(bt, policy_b, policy_a)
    bt_ci = bt_bootstrap_ci(scores, policy_a, policy_b, n_boot=bt_boot, seed=seed)
    summ = summarize_scores(scores, seed=seed)
    lo, hi = summ['bootstrap_ci']
    wr = summ['win_rate_b']
    if lo > 0.5:
        winner = policy_b
    elif hi < 0.5:
        winner = policy_a
    else:
        winner = 'inconclusive (CI spans 0.5)'
    length_controlled = sum(parity_scores) / len(parity_scores) if parity_scores else None
    notes: list[str] = []
    if length_controlled is not None and abs(length_controlled - wr) >= 0.08:
        notes.append(f'raw win rate {wr:.3f} vs length controlled {length_controlled:.3f}, gap suggests the result is partly driven by response length, prefer the length controlled figure')
    if summ['ties'] / summ['n'] > 0.4:
        notes.append(f"{summ['ties']} of {summ['n']} prompts were ties or order unstable")
    return PolicyComparisonResult(policy_a=policy_a, policy_b=policy_b, n_prompts=summ['n'], wins_a=summ['wins_a'], wins_b=summ['wins_b'], ties=summ['ties'], win_rate_b=wr, ci_low=lo, ci_high=hi, ci_method='BCa bootstrap over prompts, 10k resamples, 95 percent', p_value=summ['p_value'], significance_test='two sided exact binomial sign test vs 0.5', winner=winner, length_controlled_win_rate_b=length_controlled, bt_strength_a=bt[policy_a], bt_strength_b=bt[policy_b], bt_win_prob_b=bt_prob_b, bt_strength_a_ci=bt_ci['strength_a_ci'], bt_strength_b_ci=bt_ci['strength_b_ci'], bt_win_prob_b_ci=bt_ci['win_prob_b_ci'], percentile_ci=summ['percentile_ci'], mde=minimum_detectable_effect(summ['n']), notes=notes)

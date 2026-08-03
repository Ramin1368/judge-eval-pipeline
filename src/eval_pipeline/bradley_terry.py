from __future__ import annotations
import math
import random

def fit_bradley_terry(win_counts: dict, smoothing: float=0.5, max_iter: int=2000, tol: float=1e-10) -> dict:
    players = set()
    for a, b in win_counts:
        players.add(a)
        players.add(b)
    players = sorted(players)
    if len(players) < 2:
        return {p: 1.0 for p in players}
    counts = dict(win_counts)
    pairs = set()
    for a, b in win_counts:
        pairs.add(tuple(sorted((a, b))))
    for a, b in pairs:
        counts[a, b] = counts.get((a, b), 0.0) + smoothing
        counts[b, a] = counts.get((b, a), 0.0) + smoothing
    total_wins = {p: 0.0 for p in players}
    for (a, b), c in counts.items():
        total_wins[a] += c
    strengths = {p: 1.0 for p in players}
    for _ in range(max_iter):
        new = {}
        for p in players:
            denom = 0.0
            for q in players:
                if q == p:
                    continue
                n_pq = counts.get((p, q), 0.0) + counts.get((q, p), 0.0)
                if n_pq == 0:
                    continue
                denom += n_pq / (strengths[p] + strengths[q])
            new[p] = total_wins[p] / denom if denom > 0 else strengths[p]
        gm = math.exp(sum((math.log(v) for v in new.values())) / len(new))
        new = {p: v / gm for p, v in new.items()}
        diff = max((abs(new[p] - strengths[p]) for p in players))
        strengths = new
        if diff < tol:
            break
    return strengths

def win_probability(strengths: dict, a: str, b: str) -> float:
    sa, sb = (strengths[a], strengths[b])
    return sa / (sa + sb) if sa + sb > 0 else 0.5

def rank_policies(strengths: dict) -> list:
    return sorted(strengths.items(), key=lambda kv: kv[1], reverse=True)

def bt_bootstrap_ci(scores_per_prompt: list, policy_a: str, policy_b: str, n_boot: int=2000, alpha: float=0.05, seed: int=12345) -> dict:
    n = len(scores_per_prompt)
    if n == 0:
        return {'strength_a_ci': (1.0, 1.0), 'strength_b_ci': (1.0, 1.0), 'win_prob_b_ci': (0.5, 0.5)}
    rng = random.Random(seed)
    a_reps: list[float] = []
    b_reps: list[float] = []
    prob_reps: list[float] = []
    for _ in range(n_boot):
        resample = [scores_per_prompt[rng.randrange(n)] for _ in range(n)]
        wb = sum((1.0 if s == 1.0 else 0.5 if s == 0.5 else 0.0 for s in resample))
        wa = sum((1.0 if s == 0.0 else 0.5 if s == 0.5 else 0.0 for s in resample))
        wc = {(policy_b, policy_a): wb, (policy_a, policy_b): wa}
        st = fit_bradley_terry(wc)
        a_reps.append(st[policy_a])
        b_reps.append(st[policy_b])
        prob_reps.append(win_probability(st, policy_b, policy_a))
    a_reps.sort()
    b_reps.sort()
    prob_reps.sort()
    lo = max(0, int(math.ceil(alpha / 2.0 * n_boot)) - 1)
    hi = min(n_boot - 1, int(math.floor((1.0 - alpha / 2.0) * n_boot)) - 1)
    return {'strength_a_ci': (a_reps[lo], a_reps[hi]), 'strength_b_ci': (b_reps[lo], b_reps[hi]), 'win_prob_b_ci': (prob_reps[lo], prob_reps[hi])}

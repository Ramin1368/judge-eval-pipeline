from __future__ import annotations

import math


def fit_bradley_terry(win_counts: dict, smoothing: float = 0.5, max_iter: int = 2000, tol: float = 1e-10) -> dict:
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
        counts[(a, b)] = counts.get((a, b), 0.0) + smoothing
        counts[(b, a)] = counts.get((b, a), 0.0) + smoothing

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
        gm = math.exp(sum(math.log(v) for v in new.values()) / len(new))
        new = {p: v / gm for p, v in new.items()}
        diff = max(abs(new[p] - strengths[p]) for p in players)
        strengths = new
        if diff < tol:
            break
    return strengths


def win_probability(strengths: dict, a: str, b: str) -> float:
    sa, sb = strengths[a], strengths[b]
    return sa / (sa + sb) if (sa + sb) > 0 else 0.5


def rank_policies(strengths: dict) -> list:
    return sorted(strengths.items(), key=lambda kv: kv[1], reverse=True)

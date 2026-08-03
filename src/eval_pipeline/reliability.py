from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ReliabilityBin:
    lower: float
    upper: float
    n: int
    mean_confidence: float
    accuracy: float

def reliability_curve(confidences: list[float], correct: list[bool], n_bins: int=5) -> tuple[list[ReliabilityBin], float]:
    if not confidences:
        return ([], 0.0)
    edges = [i / n_bins for i in range(n_bins + 1)]
    bins: list[ReliabilityBin] = []
    total = len(confidences)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = (edges[i], edges[i + 1])
        idx = [j for j, c in enumerate(confidences) if c >= lo and c < hi or (i == n_bins - 1 and c == hi)]
        if not idx:
            bins.append(ReliabilityBin(lo, hi, 0, 0.0, 0.0))
            continue
        mean_conf = sum((confidences[j] for j in idx)) / len(idx)
        acc = sum((1 for j in idx if correct[j])) / len(idx)
        bins.append(ReliabilityBin(lo, hi, len(idx), mean_conf, acc))
        ece += len(idx) / total * abs(acc - mean_conf)
    return (bins, ece)

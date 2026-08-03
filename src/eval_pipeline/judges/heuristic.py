from __future__ import annotations

import math
import re
from dataclasses import dataclass

from ..schemas import JudgeVerdict, Preference
from .base import Judge

_TOKEN = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class HeuristicWeights:
    relevance: float = 1.0
    informativeness: float = 0.6
    non_degeneracy: float = 0.8
    length_prior: float = 0.15
    margin: float = 0.02


class HeuristicJudge(Judge):
    name = "heuristic_v1"

    def __init__(self, weights: HeuristicWeights | None = None):
        self.w = weights or HeuristicWeights()

    def _score(self, prompt: str, response: str) -> float:
        p_tokens = set(_tokens(prompt))
        r_tokens = _tokens(response)
        r_set = set(r_tokens)
        if not r_tokens:
            return -1.0

        overlap = len(p_tokens & r_set) / len(p_tokens | r_set) if (p_tokens | r_set) else 0.0
        informativeness = math.tanh(len(r_set) / 40.0)
        repetition = 1.0 - (len(r_set) / len(r_tokens))
        non_degeneracy = 1.0 - repetition
        length_prior = math.tanh(len(r_tokens) / 120.0)

        return (
            self.w.relevance * overlap
            + self.w.informativeness * informativeness
            + self.w.non_degeneracy * non_degeneracy
            + self.w.length_prior * length_prior
        )

    def _decide(self, prompt: str, response_a: str, response_b: str) -> JudgeVerdict:
        sa = self._score(prompt, response_a)
        sb = self._score(prompt, response_b)
        diff = sb - sa
        if abs(diff) <= self.w.margin:
            return JudgeVerdict(Preference.TIE, confidence=0.5, rationale=f"a={sa:.3f} b={sb:.3f}")
        pref = Preference.B if diff > 0 else Preference.A
        conf = min(1.0, 0.5 + abs(diff))
        return JudgeVerdict(pref, confidence=conf, rationale=f"a={sa:.3f} b={sb:.3f}")

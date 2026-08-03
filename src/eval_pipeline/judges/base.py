from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import JudgeVerdict, Preference


class Judge(ABC):
    name: str = "abstract"

    @abstractmethod
    def _decide(self, prompt: str, response_a: str, response_b: str) -> JudgeVerdict:
        ...

    def judge(self, prompt: str, response_a: str, response_b: str) -> JudgeVerdict:
        forward = self._decide(prompt, response_a, response_b)
        swapped = self._decide(prompt, response_b, response_a)
        swapped_remapped = _remap_swapped(swapped.preferred)

        if forward.preferred == swapped_remapped:
            return JudgeVerdict(
                preferred=forward.preferred,
                confidence=(forward.confidence + swapped.confidence) / 2,
                rationale=forward.rationale,
                position_unstable=False,
            )
        return JudgeVerdict(
            preferred=Preference.TIE,
            confidence=min(forward.confidence, swapped.confidence),
            rationale="verdict flipped on order swap, treated as tie",
            position_unstable=True,
        )


def _remap_swapped(pref: Preference) -> Preference:
    if pref is Preference.A:
        return Preference.B
    if pref is Preference.B:
        return Preference.A
    return Preference.TIE

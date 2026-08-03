from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Preference(str, Enum):
    A = "A"
    B = "B"
    TIE = "tie"

    @classmethod
    def parse(cls, raw: object) -> "Preference":
        s = str(raw).strip().lower()
        if s in {"a", "response_a", "0", "left", "first"}:
            return cls.A
        if s in {"b", "response_b", "1", "right", "second"}:
            return cls.B
        if s in {"tie", "equal", "both", "neither", "same", "0.5"}:
            return cls.TIE
        raise ValueError(f"Unrecognized preference label: {raw!r}")


@dataclass(frozen=True)
class PreferenceExample:
    prompt: str
    response_a: str
    response_b: str
    preferred: Preference
    example_id: str = ""
    annotator_id: Optional[str] = None

    def key(self) -> tuple[str, str, str]:
        a, b = self.response_a, self.response_b
        return (self.prompt, *sorted((a, b)))


@dataclass(frozen=True)
class PolicyOutput:
    prompt: str
    response: str
    prompt_id: str = ""


@dataclass
class JudgeVerdict:
    preferred: Preference
    confidence: float = 1.0
    rationale: str = ""
    position_unstable: bool = False


@dataclass
class CalibrationReport:
    n: int
    accuracy: float
    cohen_kappa: float
    position_bias_rate: float
    tie_rate_human: float
    tie_rate_judge: float
    per_slice: dict = field(default_factory=dict)
    expected_calibration_error: Optional[float] = None
    reliability_bins: list = field(default_factory=list)
    fallback_rate: Optional[float] = None
    trustworthy: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class PolicyComparisonResult:
    policy_a: str
    policy_b: str
    n_prompts: int
    wins_a: int
    wins_b: int
    ties: int
    win_rate_b: float
    ci_low: float
    ci_high: float
    ci_method: str
    p_value: float
    significance_test: str
    winner: str
    length_controlled_win_rate_b: Optional[float] = None
    bt_strength_a: Optional[float] = None
    bt_strength_b: Optional[float] = None
    bt_win_prob_b: Optional[float] = None
    notes: list[str] = field(default_factory=list)

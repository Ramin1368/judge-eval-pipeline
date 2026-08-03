from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .agreement import fleiss_kappa
from .schemas import Preference, PreferenceExample


@dataclass
class ValidationReport:
    n_input: int
    n_clean: int
    n_quarantined: int
    duplicate_groups: int
    contradictory_groups: int
    order_swapped_groups: int
    human_self_consistency: float | None
    fleiss_kappa: float | None
    messages: list[str] = field(default_factory=list)


def _normalize_to_canonical_order(ex: PreferenceExample) -> Preference:
    a, b = ex.response_a, ex.response_b
    if a <= b:
        return ex.preferred
    if ex.preferred is Preference.A:
        return Preference.B
    if ex.preferred is Preference.B:
        return Preference.A
    return Preference.TIE


def validate_and_dedupe(
    examples: list[PreferenceExample],
) -> tuple[list[PreferenceExample], ValidationReport]:
    groups: dict[tuple, list[PreferenceExample]] = defaultdict(list)
    for ex in examples:
        groups[ex.key()].append(ex)

    clean: list[PreferenceExample] = []
    quarantined = 0
    dup_groups = 0
    contradictory = 0
    order_swapped = 0
    messages: list[str] = []

    consistency_hits = 0
    consistency_total = 0

    for key, members in groups.items():
        canonical_labels = [_normalize_to_canonical_order(m) for m in members]
        orientations = {(m.response_a <= m.response_b) for m in members}
        if len(members) > 1:
            dup_groups += 1
            if len(orientations) > 1:
                order_swapped += 1

        counts = Counter(canonical_labels)
        if len(members) > 1:
            consistency_total += len(members)
            consistency_hits += counts.most_common(1)[0][1]
            if len(set(canonical_labels)) > 1:
                contradictory += 1
                messages.append(
                    f"contradictory labels for prompt={members[0].prompt[:40]!r}: "
                    f"{_fmt(counts)} resolved by majority"
                )

        top = counts.most_common()
        if len(top) >= 2 and top[0][1] == top[1][1] and Preference.TIE not in {top[0][0]}:
            quarantined += 1
            messages.append(
                f"unresolvable split {_fmt(counts)} for prompt="
                f"{members[0].prompt[:40]!r} quarantined from calibration"
            )
            continue

        resolved_canonical = top[0][0]
        first = members[0]
        if first.response_a <= first.response_b:
            resolved = resolved_canonical
        else:
            resolved = {
                Preference.A: Preference.B,
                Preference.B: Preference.A,
                Preference.TIE: Preference.TIE,
            }[resolved_canonical]

        clean.append(
            PreferenceExample(
                prompt=first.prompt,
                response_a=first.response_a,
                response_b=first.response_b,
                preferred=resolved,
                example_id=first.example_id,
                annotator_id=first.annotator_id,
            )
        )

    consistency = (consistency_hits / consistency_total) if consistency_total else None
    report = ValidationReport(
        n_input=len(examples),
        n_clean=len(clean),
        n_quarantined=quarantined,
        duplicate_groups=dup_groups,
        contradictory_groups=contradictory,
        order_swapped_groups=order_swapped,
        human_self_consistency=consistency,
        fleiss_kappa=fleiss_kappa(examples),
        messages=messages,
    )
    return clean, report


def _fmt(counts: Counter) -> str:
    return "{" + ", ".join(f"{k.value}:{v}" for k, v in counts.items()) + "}"

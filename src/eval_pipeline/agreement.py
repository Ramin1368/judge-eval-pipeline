from __future__ import annotations

from collections import Counter, defaultdict

from .schemas import Preference, PreferenceExample


def _canonical_label(ex: PreferenceExample) -> Preference:
    if ex.response_a <= ex.response_b:
        return ex.preferred
    if ex.preferred is Preference.A:
        return Preference.B
    if ex.preferred is Preference.B:
        return Preference.A
    return Preference.TIE


def fleiss_kappa(examples: list[PreferenceExample]) -> float | None:
    categories = [Preference.A.value, Preference.B.value, Preference.TIE.value]
    cat_index = {c: i for i, c in enumerate(categories)}

    groups: dict[tuple, list[PreferenceExample]] = defaultdict(list)
    for ex in examples:
        groups[ex.key()].append(ex)

    rated = [members for members in groups.values() if len(members) >= 2]
    if not rated:
        return None

    n_raters = min(len(m) for m in rated)
    if n_raters < 2:
        return None

    rows = []
    for members in rated:
        counts = Counter(_canonical_label(m).value for m in members[:n_raters])
        row = [counts.get(c, 0) for c in categories]
        rows.append(row)

    n_items = len(rows)
    p_j = [0.0] * len(categories)
    for row in rows:
        for j in range(len(categories)):
            p_j[j] += row[j]
    total = n_items * n_raters
    p_j = [x / total for x in p_j]

    p_i = []
    for row in rows:
        s = sum(c * c for c in row)
        p_i.append((s - n_raters) / (n_raters * (n_raters - 1)))

    p_bar = sum(p_i) / n_items
    p_e = sum(p * p for p in p_j)
    if p_e == 1.0:
        return 1.0
    return (p_bar - p_e) / (1.0 - p_e)

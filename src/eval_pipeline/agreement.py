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
    """Fleiss kappa with support for variable rater counts per item.

    Standard Fleiss assumes a constant number of raters n per item. In real
    preference datasets, some items get 2 annotations and others get 6. The
    variable-rater generalization (Fleiss 1971 as extended in Nichols et al.
    2010) computes agreement per item using n_i * (n_i - 1) in the denominator,
    then weights by items rather than by a constant n. This avoids the
    silent-truncation bug of taking min(n_i) across items, which discards real
    signal.
    """
    categories = [Preference.A.value, Preference.B.value, Preference.TIE.value]

    groups: dict[tuple, list[PreferenceExample]] = defaultdict(list)
    for ex in examples:
        groups[ex.key()].append(ex)

    rated = [members for members in groups.values() if len(members) >= 2]
    if not rated:
        return None

    rows: list[tuple[list[int], int]] = []
    for members in rated:
        counts = Counter(_canonical_label(m).value for m in members)
        row = [counts.get(c, 0) for c in categories]
        n_i = sum(row)
        if n_i < 2:
            continue
        rows.append((row, n_i))
    if not rows:
        return None

    total_ratings = sum(n_i for _, n_i in rows)
    p_j = [0.0] * len(categories)
    for row, _ in rows:
        for j in range(len(categories)):
            p_j[j] += row[j]
    p_j = [x / total_ratings for x in p_j]

    p_i_vals: list[float] = []
    for row, n_i in rows:
        s = sum(c * c for c in row)
        p_i_vals.append((s - n_i) / (n_i * (n_i - 1)))

    p_bar = sum(p_i_vals) / len(p_i_vals)
    p_e = sum(p * p for p in p_j)
    if p_e >= 1.0:
        return 1.0
    return (p_bar - p_e) / (1.0 - p_e)

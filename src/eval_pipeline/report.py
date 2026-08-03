from __future__ import annotations

from .schemas import CalibrationReport, PolicyComparisonResult
from .validation import ValidationReport


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def build_report(
    calibration: CalibrationReport,
    comparison: PolicyComparisonResult,
    validation: ValidationReport | None = None,
    judge_name: str = "judge",
    judge_table: list[dict] | None = None,
) -> str:
    c, m = calibration, comparison
    trust = "TRUSTWORTHY" if c.trustworthy else "USE WITH CAUTION"

    if m.winner.startswith("inconclusive"):
        headline = (
            f"**Result: inconclusive.** We cannot say policy '{m.policy_b}' and "
            f"'{m.policy_a}' differ. The estimated win rate for {m.policy_b} is "
            f"{_pct(m.win_rate_b)}, but the 95 percent confidence interval "
            f"[{_pct(m.ci_low)}, {_pct(m.ci_high)}] includes 50 percent, so the "
            f"difference is not statistically distinguishable from a coin flip."
        )
    else:
        headline = (
            f"**Result: '{m.winner}' is the better policy.** On the held out set, "
            f"'{m.policy_b}' wins {_pct(m.win_rate_b)} of head to head comparisons "
            f"against '{m.policy_a}' (95 percent CI [{_pct(m.ci_low)}, {_pct(m.ci_high)}], "
            f"p = {m.p_value:.4f}). Because the whole interval sits on one side of 50 "
            f"percent, the result is statistically meaningful, not noise."
        )

    lines: list[str] = []
    lines.append("# Policy Evaluation Report\n")
    lines.append("## For a non-research stakeholder\n")
    lines.append(headline + "\n")
    lines.append(
        f"How much should you trust this? The automated judge was checked against "
        f"real human preferences before we used it, and it rated **{trust}** "
        f"(chance corrected agreement, Cohen's kappa, of {c.cohen_kappa:.2f}). "
        + ("Treat the verdict above as reliable.\n" if c.trustworthy else
           "Treat the verdict above as directional, not definitive.\n")
    )
    if m.length_controlled_win_rate_b is not None:
        gap = abs(m.length_controlled_win_rate_b - m.win_rate_b)
        lines.append(
            f"Gaming check: when we restrict to prompts where both answers were "
            f"similar in length, the win rate is {_pct(m.length_controlled_win_rate_b)} "
            f"(vs {_pct(m.win_rate_b)} overall), so the result is "
            f"{'not merely' if gap < 0.08 else 'partly'} a length effect.\n"
        )

    lines.append("## Judge calibration\n")
    lines.append(f"Judge: {judge_name}")
    lines.append(f"Labeled items evaluated: {c.n}")
    lines.append(f"Raw agreement with humans (accuracy): {_pct(c.accuracy)}")
    lines.append(f"Cohen's kappa (chance corrected): {c.cohen_kappa:.3f} (trust gate at least 0.40)")
    lines.append(f"Position bias rate (verdict flips on A/B swap): {_pct(c.position_bias_rate)}")
    if c.expected_calibration_error is not None:
        lines.append(f"Expected calibration error (confidence vs correctness): {c.expected_calibration_error:.3f}")
    if c.fallback_rate is not None:
        lines.append(f"Heuristic fallback rate (LLM unavailable): {_pct(c.fallback_rate)}")
    lines.append(f"Tie rate, human vs judge: {_pct(c.tie_rate_human)} vs {_pct(c.tie_rate_judge)}")
    if c.per_slice:
        lines.append("Per slice accuracy:")
        for name, s in c.per_slice.items():
            lines.append(f"    {name}: {_pct(s['accuracy'])} (n={s['n']})")
    for note in c.notes:
        lines.append(f"Note: {note}")
    lines.append("")

    if judge_table:
        lines.append("## Judge comparison\n")
        lines.append("| judge | n | accuracy | kappa | position bias | ECE | trustworthy |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in judge_table:
            ece = f"{r['expected_calibration_error']:.3f}" if r['expected_calibration_error'] is not None else "n/a"
            lines.append(
                f"| {r['judge']} | {r['n']} | {_pct(r['accuracy'])} | {r['cohen_kappa']:.3f} | "
                f"{_pct(r['position_bias_rate'])} | {ece} | {r['trustworthy']} |"
            )
        lines.append("")

    lines.append("## Policy comparison\n")
    lines.append(f"Policies: A = {m.policy_a}, B = {m.policy_b}")
    lines.append(f"Held out prompts: {m.n_prompts}")
    lines.append(f"Wins: A={m.wins_a}, B={m.wins_b}, ties or unstable={m.ties}")
    lines.append(f"Win rate for B (ties count as 0.5): {_pct(m.win_rate_b)}")
    lines.append(f"95 percent CI: [{_pct(m.ci_low)}, {_pct(m.ci_high)}] ({m.ci_method})")
    lines.append(f"Significance: p = {m.p_value:.4f} ({m.significance_test})")
    if m.length_controlled_win_rate_b is not None:
        lines.append(f"Length controlled win rate for B: {_pct(m.length_controlled_win_rate_b)}")
    if m.bt_win_prob_b is not None:
        lines.append(
            f"Bradley-Terry strengths: {m.policy_a}={m.bt_strength_a:.3f}, {m.policy_b}={m.bt_strength_b:.3f}, "
            f"implied P(B preferred)={_pct(m.bt_win_prob_b)}"
        )
    for note in m.notes:
        lines.append(f"Note: {note}")
    lines.append("")

    if validation is not None:
        v = validation
        lines.append("## Data quality\n")
        lines.append(f"Input rows: {v.n_input}, unique comparisons kept: {v.n_clean}, quarantined: {v.n_quarantined}")
        lines.append(f"Duplicate comparison groups: {v.duplicate_groups}")
        lines.append(f"Order swapped groups: {v.order_swapped_groups}")
        lines.append(f"Contradictory groups resolved by majority: {v.contradictory_groups}")
        if v.human_self_consistency is not None:
            lines.append(f"Human self consistency: {_pct(v.human_self_consistency)}")
        if v.fleiss_kappa is not None:
            lines.append(f"Inter annotator agreement (Fleiss kappa): {v.fleiss_kappa:.3f}")
        for msg in v.messages[:8]:
            lines.append(f"    {msg}")
        if len(v.messages) > 8:
            lines.append(f"    and {len(v.messages) - 8} more")
        lines.append("")

    return "\n".join(lines)


def build_html_report(
    calibration: CalibrationReport,
    comparison: PolicyComparisonResult,
    validation: ValidationReport | None = None,
    judge_name: str = "judge",
) -> str:
    m = comparison
    chart = _ci_chart_svg(m.win_rate_b, m.ci_low, m.ci_high)
    rel = _reliability_svg(calibration.reliability_bins)
    verdict = m.winner if not m.winner.startswith("inconclusive") else "Inconclusive"
    trust = "Trustworthy" if calibration.trustworthy else "Use with caution"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Policy Evaluation Report</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#16202a;max-width:820px;margin:32px auto;padding:0 20px;line-height:1.5}}
h1{{font-size:24px}} h2{{font-size:16px;color:#0b3d63;margin-top:28px}}
.verdict{{background:#0b3d63;color:#fff;padding:16px 20px;border-radius:10px;font-size:18px}}
.kpis{{display:flex;gap:14px;flex-wrap:wrap;margin:16px 0}}
.kpi{{flex:1;min-width:150px;border:1px solid #dfe6ec;border-radius:10px;padding:12px 14px}}
.kpi .v{{font-size:22px;font-weight:700}} .kpi .l{{font-size:12px;color:#5b6b78}}
table{{border-collapse:collapse;width:100%;font-size:13px}} td,th{{border:1px solid #dfe6ec;padding:6px 8px;text-align:left}}
.muted{{color:#5b6b78;font-size:13px}}
</style></head><body>
<h1>Policy Evaluation Report</h1>
<div class="verdict">Winner: {verdict} &nbsp; ({_pct(m.win_rate_b)} win rate for {m.policy_b})</div>
<div class="kpis">
  <div class="kpi"><div class="v">{_pct(m.win_rate_b)}</div><div class="l">Win rate B</div></div>
  <div class="kpi"><div class="v">[{_pct(m.ci_low)}, {_pct(m.ci_high)}]</div><div class="l">95% CI</div></div>
  <div class="kpi"><div class="v">{m.p_value:.4f}</div><div class="l">p value</div></div>
  <div class="kpi"><div class="v">{calibration.cohen_kappa:.2f}</div><div class="l">Judge kappa ({trust})</div></div>
</div>
<h2>Win rate with 95% confidence interval</h2>
{chart}
<p class="muted">The bar is the point estimate. The whisker is the 95% bootstrap interval. If it crosses the 50% line, the result is inconclusive.</p>
<h2>Judge confidence calibration</h2>
{rel}
<p class="muted">Points on the diagonal mean the judge's stated confidence matches its actual accuracy. ECE = {calibration.expected_calibration_error:.3f}.</p>
<h2>Data quality</h2>
<p class="muted">{_html_dq(validation)}</p>
</body></html>"""


def _ci_chart_svg(point: float, lo: float, hi: float) -> str:
    w, h = 720, 90
    x = lambda p: 40 + p * (w - 80)
    half = x(0.5)
    return (
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
        f'<line x1="{half}" y1="10" x2="{half}" y2="{h-20}" stroke="#c33" stroke-dasharray="4 4"/>'
        f'<text x="{half}" y="{h-4}" font-size="11" fill="#c33" text-anchor="middle">50%</text>'
        f'<line x1="{x(lo)}" y1="40" x2="{x(hi)}" y2="40" stroke="#0b3d63" stroke-width="3"/>'
        f'<line x1="{x(lo)}" y1="30" x2="{x(lo)}" y2="50" stroke="#0b3d63" stroke-width="3"/>'
        f'<line x1="{x(hi)}" y1="30" x2="{x(hi)}" y2="50" stroke="#0b3d63" stroke-width="3"/>'
        f'<circle cx="{x(point)}" cy="40" r="6" fill="#0b3d63"/>'
        f'<text x="{x(point)}" y="26" font-size="11" fill="#0b3d63" text-anchor="middle">{_pct(point)}</text>'
        f'</svg>'
    )


def _reliability_svg(bins: list[dict]) -> str:
    w, h = 300, 300
    pad = 40
    x = lambda v: pad + v * (w - 2 * pad)
    y = lambda v: (h - pad) - v * (h - 2 * pad)
    parts = [
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'<line x1="{x(0)}" y1="{y(0)}" x2="{x(1)}" y2="{y(1)}" stroke="#aab" stroke-dasharray="4 4"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="#333"/>',
        f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#333"/>',
        f'<text x="{w/2}" y="{h-8}" font-size="11" text-anchor="middle">confidence</text>',
    ]
    pts = [(b["mean_confidence"], b["accuracy"]) for b in bins if b["n"] > 0]
    for i in range(1, len(pts)):
        parts.append(
            f'<line x1="{x(pts[i-1][0])}" y1="{y(pts[i-1][1])}" x2="{x(pts[i][0])}" y2="{y(pts[i][1])}" stroke="#0b3d63" stroke-width="2"/>'
        )
    for cx, cy in pts:
        parts.append(f'<circle cx="{x(cx)}" cy="{y(cy)}" r="4" fill="#0b3d63"/>')
    parts.append('</svg>')
    return "".join(parts)


def _html_dq(v: ValidationReport | None) -> str:
    if v is None:
        return "No validation data."
    fk = f", Fleiss kappa {v.fleiss_kappa:.3f}" if v.fleiss_kappa is not None else ""
    sc = f", human self consistency {_pct(v.human_self_consistency)}" if v.human_self_consistency is not None else ""
    return (
        f"{v.n_input} input rows collapsed to {v.n_clean} unique comparisons, "
        f"{v.contradictory_groups} contradictions resolved by majority, "
        f"{v.n_quarantined} quarantined{sc}{fk}."
    )

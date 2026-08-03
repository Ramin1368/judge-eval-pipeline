from __future__ import annotations

import argparse
import sys

from .ingestion import load_preferences, load_policy_outputs
from .validation import validate_and_dedupe
from .calibration import calibrate_judge
from .compare_judges import compare_judges
from .policy_compare import compare_policies
from .report import build_report, build_html_report
from .judges import build_judge


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Offline evaluation and judge calibration pipeline")
    ap.add_argument("--preferences", required=True)
    ap.add_argument("--policies", required=True)
    ap.add_argument("--policy-a", required=True)
    ap.add_argument("--policy-b", required=True)
    ap.add_argument("--judge", default="heuristic")
    ap.add_argument("--out", default="report.md")
    ap.add_argument("--html", default=None)
    ap.add_argument("--compare-judges", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args(argv)

    examples, ingest_errors = load_preferences(args.preferences)
    if ingest_errors:
        print(f"[ingestion] {len(ingest_errors)} rows rejected", file=sys.stderr)
        for e in ingest_errors[:10]:
            print(f"  {e}", file=sys.stderr)
    if not examples:
        print("[ingestion] no valid preference examples, aborting", file=sys.stderr)
        return 2

    clean, validation = validate_and_dedupe(examples)
    print(f"[validation] {validation.n_input} to {validation.n_clean} clean, "
          f"{validation.n_quarantined} quarantined, "
          f"{validation.contradictory_groups} contradictions resolved, "
          f"fleiss={validation.fleiss_kappa}")

    judge = build_judge(args.judge)
    calibration = calibrate_judge(judge, clean)
    print(f"[calibration] judge={judge.name} accuracy={calibration.accuracy:.3f} "
          f"kappa={calibration.cohen_kappa:.3f} position_bias={calibration.position_bias_rate:.3f} "
          f"ece={calibration.expected_calibration_error:.3f} trustworthy={calibration.trustworthy}")

    judge_table = None
    if args.compare_judges:
        judge_table = compare_judges(clean, args.compare_judges)
        for r in judge_table:
            print(f"[compare] {r['judge']}: kappa={r['cohen_kappa']:.3f} acc={r['accuracy']:.3f}")

    rows = load_policy_outputs(args.policies)
    comparison = compare_policies(judge, rows, args.policy_a, args.policy_b, seed=args.seed)
    print(f"[comparison] winner={comparison.winner} win_rate_b={comparison.win_rate_b:.3f} "
          f"CI=[{comparison.ci_low:.3f}, {comparison.ci_high:.3f}] p={comparison.p_value:.4f}")

    report = build_report(calibration, comparison, validation, judge_name=judge.name, judge_table=judge_table)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"[report] written to {args.out}")

    if args.html:
        with open(args.html, "w", encoding="utf-8") as fh:
            fh.write(build_html_report(calibration, comparison, validation, judge_name=judge.name))
        print(f"[report] html written to {args.html}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

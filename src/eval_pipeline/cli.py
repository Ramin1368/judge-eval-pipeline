from __future__ import annotations
import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from .cache import JSONFileCache
from .calibration import calibrate_judge
from .compare_judges import compare_judges
from .config import load_config
from .ingestion import load_preferences, load_policy_outputs
from .judges import build_judge
from .judges.llm_judge import DigitalOceanLLMJudge
from .policy_compare import compare_policies
from .report import build_html_report, build_report
from .validation import validate_and_dedupe

def _asdict_safe(obj) -> dict:
    if dataclasses.is_dataclass(obj):
        d = dataclasses.asdict(obj)
        return {k: v for k, v in d.items()}
    return {}

def main(argv: list[str] | None=None) -> int:
    ap = argparse.ArgumentParser(description='Offline evaluation and judge calibration pipeline')
    ap.add_argument('--preferences', required=False, default=None)
    ap.add_argument('--policies', required=False, default=None)
    ap.add_argument('--policy-a', required=False, default='policy_a')
    ap.add_argument('--policy-b', required=False, default='policy_b')
    ap.add_argument('--judge', default=None, help='Overrides config judge.kind')
    ap.add_argument('--config', default='config/config.yaml')
    ap.add_argument('--live-llm', action='store_true', help='Use the DigitalOcean LLM judge; requires DO_INFERENCE_API_KEY')
    ap.add_argument('--out', default='report.md')
    ap.add_argument('--out-json', default=None, help='Also write machine readable results here')
    ap.add_argument('--html', default=None)
    ap.add_argument('--cache', default=None, help='Path to a JSON cache file for LLM verdicts (overrides config)')
    ap.add_argument('--no-cache', action='store_true', help='Disable the LLM verdict cache')
    ap.add_argument('--compare-judges', nargs='*', default=None)
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if not args.preferences or not args.policies:
        print('--preferences and --policies are required', file=sys.stderr)
        return 2
    seed = args.seed if args.seed is not None else cfg.seed
    examples, ingest_errors = load_preferences(args.preferences)
    if ingest_errors:
        print(f'[ingestion] {len(ingest_errors)} rows rejected', file=sys.stderr)
        for e in ingest_errors[:10]:
            print(f'  {e}', file=sys.stderr)
    if not examples:
        print('[ingestion] no valid preference examples, aborting', file=sys.stderr)
        return 2
    clean, validation = validate_and_dedupe(examples)
    print(f'[validation] {validation.n_input} to {validation.n_clean} clean, {validation.n_quarantined} quarantined, {validation.contradictory_groups} contradictions resolved, fleiss={validation.fleiss_kappa}')
    kind = 'llm' if args.live_llm else args.judge or cfg.judge_kind
    cache = None
    if kind in {'llm', 'do', 'do_llm', 'digitalocean'} and (not args.no_cache):
        cache_path = args.cache or cfg.cache_path
        cache = JSONFileCache(cache_path)
    if kind in {'llm', 'do', 'do_llm', 'digitalocean'}:
        judge = DigitalOceanLLMJudge(cache=cache, seed=seed)
    else:
        judge = build_judge(kind)
    calibration = calibrate_judge(judge, clean, seed=seed)
    print(f'[calibration] judge={judge.name} accuracy={calibration.accuracy:.3f} kappa={calibration.cohen_kappa:.3f} kappa_ci=[{calibration.kappa_ci_low or 0:.3f},{calibration.kappa_ci_high or 0:.3f}] position_bias={calibration.position_bias_rate:.3f} ece={calibration.expected_calibration_error:.3f} trust={calibration.trustworthy} reserve={calibration.trustworthy_with_reserve}')
    judge_table = None
    if args.compare_judges:
        judge_table = compare_judges(clean, args.compare_judges)
        for r in judge_table:
            print(f"[compare] {r['judge']}: kappa={r['cohen_kappa']:.3f} acc={r['accuracy']:.3f}")
    rows = load_policy_outputs(args.policies)
    comparison = compare_policies(judge, rows, args.policy_a, args.policy_b, seed=seed)
    print(f'[comparison] winner={comparison.winner} win_rate_b={comparison.win_rate_b:.3f} CI=[{comparison.ci_low:.3f}, {comparison.ci_high:.3f}] p={comparison.p_value:.4f} MDE={comparison.mde:.3f}')
    cache_stats = None
    if cache is not None:
        cache_stats = {'hits': cache.stats.hits, 'misses': cache.stats.misses, 'writes': cache.stats.writes, 'hit_rate': cache.stats.hit_rate()}
        print(f'[cache] hits={cache.stats.hits} misses={cache.stats.misses} writes={cache.stats.writes} hit_rate={cache.stats.hit_rate():.2%}')
    report = build_report(calibration, comparison, validation, judge_name=judge.name, judge_table=judge_table, cache_stats=cache_stats)
    with open(args.out, 'w', encoding='utf-8') as fh:
        fh.write(report)
    print(f'[report] written to {args.out}')
    if args.html:
        with open(args.html, 'w', encoding='utf-8') as fh:
            fh.write(build_html_report(calibration, comparison, validation, judge_name=judge.name))
        print(f'[report] html written to {args.html}')
    if args.out_json:
        payload = {'judge': judge.name, 'calibration': _asdict_safe(calibration), 'comparison': _asdict_safe(comparison), 'validation': _asdict_safe(validation), 'cache': cache_stats}
        with open(args.out_json, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, indent=2, default=str)
        print(f'[report] json written to {args.out_json}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

@dataclass
class PipelineConfig:
    judge_kind: str = 'heuristic'
    kappa_trust_threshold: float = 0.4
    position_bias_max: float = 0.15
    bootstrap_resamples: int = 10000
    alpha: float = 0.05
    seed: int = 12345
    cache_path: str = '.cache/judge_verdicts.json'
    inference_env: dict = field(default_factory=lambda: {'base_url_env': 'DO_INFERENCE_BASE_URL', 'api_key_env': 'DO_INFERENCE_API_KEY', 'model_env': 'DO_INFERENCE_MODEL'})
    raw: dict = field(default_factory=dict)

def _fallback_parse(text: str) -> dict:
    root: dict[str, Any] = {}
    current: dict[str, Any] | None = root
    stack: list[tuple[int, dict]] = [(0, root)]
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith('#'):
            continue
        indent = len(line) - len(line.lstrip())
        while stack and indent < stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        key, _, value = line.strip().partition(':')
        key = key.strip()
        value = value.strip()
        if not value:
            new: dict[str, Any] = {}
            current[key] = new
            stack.append((indent + 2, new))
        elif value.lower() in {'true', 'false'}:
            current[key] = value.lower() == 'true'
        else:
            try:
                current[key] = int(value)
            except ValueError:
                try:
                    current[key] = float(value)
                except ValueError:
                    current[key] = value.strip('"').strip("'")
    return root

def load_config(path: str | Path | None) -> PipelineConfig:
    if not path:
        return PipelineConfig()
    p = Path(path)
    if not p.exists():
        return PipelineConfig()
    text = p.read_text(encoding='utf-8')
    try:
        import yaml
        data = yaml.safe_load(text) or {}
    except ImportError:
        data = _fallback_parse(text)
    return _from_dict(data)

def _from_dict(data: dict) -> PipelineConfig:
    cfg = PipelineConfig(raw=data)
    if 'judge' in data and isinstance(data['judge'], dict):
        cfg.judge_kind = str(data['judge'].get('kind', cfg.judge_kind))
    if 'calibration' in data and isinstance(data['calibration'], dict):
        c = data['calibration']
        cfg.kappa_trust_threshold = float(c.get('kappa_trust_threshold', cfg.kappa_trust_threshold))
        cfg.position_bias_max = float(c.get('position_bias_max', cfg.position_bias_max))
    if 'stats' in data and isinstance(data['stats'], dict):
        s = data['stats']
        cfg.bootstrap_resamples = int(s.get('bootstrap_resamples', cfg.bootstrap_resamples))
        cfg.alpha = float(s.get('alpha', cfg.alpha))
        cfg.seed = int(s.get('seed', cfg.seed))
    if 'cache' in data and isinstance(data['cache'], dict):
        cfg.cache_path = str(data['cache'].get('path', cfg.cache_path))
    if 'inference' in data and isinstance(data['inference'], dict):
        cfg.inference_env.update({k: str(v) for k, v in data['inference'].items()})
    return cfg

from .base import Judge
from .heuristic import HeuristicJudge, HeuristicWeights
from .llm_judge import DigitalOceanLLMJudge
__all__ = ['Judge', 'HeuristicJudge', 'HeuristicWeights', 'DigitalOceanLLMJudge', 'build_judge']

def build_judge(kind: str, **kwargs) -> Judge:
    kind = (kind or 'heuristic').lower()
    if kind in {'heuristic', 'baseline'}:
        return HeuristicJudge()
    if kind in {'llm', 'do', 'digitalocean', 'do_llm'}:
        return DigitalOceanLLMJudge(**kwargs)
    raise ValueError(f'unknown judge kind: {kind!r}')

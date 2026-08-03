from __future__ import annotations
import json
import os
import tempfile
from eval_pipeline.cache import InMemoryCache, JSONFileCache, make_cache_key

def test_cache_key_stable_and_input_sensitive():
    k1 = make_cache_key('m', 'p', 'a', 'b', 0.0, 12345)
    k2 = make_cache_key('m', 'p', 'a', 'b', 0.0, 12345)
    assert k1 == k2
    k3 = make_cache_key('m', 'p', 'a', 'b*', 0.0, 12345)
    assert k1 != k3
    k4 = make_cache_key('m', 'p', 'a', 'b', 0.7, 12345)
    assert k1 != k4

def test_json_cache_roundtrip_and_stats(tmp_path):
    p = tmp_path / 'cache.json'
    c = JSONFileCache(p)
    assert c.get('x') is None
    assert c.stats.misses == 1
    c.set('x', {'preferred': 'A', 'confidence': 0.9})
    assert c.stats.writes == 1
    assert c.get('x') == {'preferred': 'A', 'confidence': 0.9}
    assert c.stats.hits == 1
    c2 = JSONFileCache(p)
    assert c2.get('x') == {'preferred': 'A', 'confidence': 0.9}
    with open(p) as fh:
        assert json.load(fh) == {'x': {'preferred': 'A', 'confidence': 0.9}}

def test_in_memory_cache_matches_protocol():
    c = InMemoryCache()
    assert c.get('k') is None
    c.set('k', {'v': 1})
    assert c.get('k') == {'v': 1}

def test_llm_judge_uses_cache_on_hit():
    from eval_pipeline.judges.llm_judge import DigitalOceanLLMJudge
    from eval_pipeline.schemas import JudgeVerdict, Preference
    cache = InMemoryCache()
    j = DigitalOceanLLMJudge(api_key='', cache=cache)
    v1 = j._decide('p', 'a', 'b')
    assert isinstance(v1, JudgeVerdict)
    assert cache.stats.writes == 0
    k = make_cache_key(j.model, 'p2', 'a2', 'b2', j.temperature, j.seed)
    cache.set(k, {'preferred': 'A', 'confidence': 0.42, 'rationale': 'cached'})
    v2 = j._decide('p2', 'a2', 'b2')
    assert v2.preferred is Preference.A
    assert v2.confidence == 0.42

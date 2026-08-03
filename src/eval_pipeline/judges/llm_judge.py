from __future__ import annotations
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from typing import Optional
from ..cache import Cache, make_cache_key
from ..schemas import JudgeVerdict, Preference
from .base import Judge
from .heuristic import HeuristicJudge
_SYSTEM = 'You are a careful evaluation judge. Compare two responses to the same prompt and decide which better satisfies the user\'s request. Judge on correctness, relevance, and helpfulness. Do not reward length or confident tone for their own sake. Respond with strict JSON only: {"winner": "A" | "B" | "tie", "confidence": 0.0-1.0, "reason": "<short>"}'
_USER_TMPL = 'PROMPT:\n{prompt}\n\nRESPONSE A:\n{a}\n\nRESPONSE B:\n{b}\n\nReturn the JSON verdict now.'

class DigitalOceanLLMJudge(Judge):
    name = 'do_llm_judge'

    def __init__(self, model: str | None=None, base_url: str | None=None, api_key: str | None=None, timeout: float=30.0, fallback: Judge | None=None, cache: Optional[Cache]=None, temperature: float=0.0, seed: int=12345, max_retries: int=3, backoff_base: float=0.5):
        self.base_url = (base_url or os.getenv('DO_INFERENCE_BASE_URL', 'https://inference.do-ai.run/v1')).rstrip('/')
        self.api_key = api_key or os.getenv('DO_INFERENCE_API_KEY', '')
        self.model = model or os.getenv('DO_INFERENCE_MODEL', 'llama3.3-70b-instruct')
        self.timeout = timeout
        self.fallback = fallback or HeuristicJudge()
        self.cache = cache
        self.temperature = temperature
        self.seed = seed
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.fallback_count = 0
        self.call_count = 0

    def _decide(self, prompt: str, response_a: str, response_b: str) -> JudgeVerdict:
        self.call_count += 1
        cache_key = None
        if self.cache is not None:
            cache_key = make_cache_key(self.model, prompt, response_a, response_b, self.temperature, self.seed)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return _verdict_from_dict(cached)
        if not self.api_key:
            self.fallback_count += 1
            return self.fallback._decide(prompt, response_a, response_b)
        try:
            raw = self._call_api_with_retry(prompt, response_a, response_b)
            verdict = self._parse(raw)
            if self.cache is not None and cache_key is not None:
                self.cache.set(cache_key, _verdict_to_dict(verdict))
            return verdict
        except Exception:
            self.fallback_count += 1
            return self.fallback._decide(prompt, response_a, response_b)

    def fallback_rate(self) -> float:
        return self.fallback_count / self.call_count if self.call_count else 0.0

    def _call_api_with_retry(self, prompt: str, a: str, b: str) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return self._call_api(prompt, a, b)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last_err = e
                sleep = self.backoff_base * 2 ** attempt + random.random() * 0.1
                time.sleep(sleep)
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504):
                    last_err = e
                    sleep = self.backoff_base * 2 ** attempt + random.random() * 0.1
                    time.sleep(sleep)
                    continue
                raise
        assert last_err is not None
        raise last_err

    def _call_api(self, prompt: str, a: str, b: str) -> str:
        payload = {'model': self.model, 'temperature': self.temperature, 'max_tokens': 200, 'messages': [{'role': 'system', 'content': _SYSTEM}, {'role': 'user', 'content': _USER_TMPL.format(prompt=prompt, a=a, b=b)}]}
        req = urllib.request.Request(f'{self.base_url}/chat/completions', data=json.dumps(payload).encode(), headers={'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode())
        return body['choices'][0]['message']['content']

    @staticmethod
    def _parse(content: str) -> JudgeVerdict:
        match = re.search('\\{.*\\}', content, re.DOTALL)
        if not match:
            raise ValueError('no JSON in judge response')
        obj = json.loads(match.group(0))
        winner = str(obj.get('winner', '')).strip().lower()
        pref = {'a': Preference.A, 'b': Preference.B, 'tie': Preference.TIE}.get(winner)
        if pref is None:
            raise ValueError(f'bad winner field: {winner!r}')
        conf = float(obj.get('confidence', 0.7))
        return JudgeVerdict(pref, confidence=max(0.0, min(1.0, conf)), rationale=str(obj.get('reason', '')))

def _verdict_to_dict(v: JudgeVerdict) -> dict:
    return {'preferred': v.preferred.value, 'confidence': v.confidence, 'rationale': v.rationale, 'position_unstable': v.position_unstable}

def _verdict_from_dict(d: dict) -> JudgeVerdict:
    return JudgeVerdict(preferred=Preference(d['preferred']), confidence=float(d.get('confidence', 1.0)), rationale=str(d.get('rationale', '')), position_unstable=bool(d.get('position_unstable', False)))

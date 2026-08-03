from __future__ import annotations

import json
import os
import re
import urllib.request

from ..schemas import JudgeVerdict, Preference
from .base import Judge
from .heuristic import HeuristicJudge

_SYSTEM = (
    "You are a careful evaluation judge. Compare two responses to the same "
    "prompt and decide which better satisfies the user's request. Judge on "
    "correctness, relevance, and helpfulness. Do not reward length or "
    "confident tone for their own sake. Respond with strict JSON only: "
    '{"winner": "A" | "B" | "tie", "confidence": 0.0-1.0, "reason": "<short>"}'
)

_USER_TMPL = (
    "PROMPT:\n{prompt}\n\n"
    "RESPONSE A:\n{a}\n\n"
    "RESPONSE B:\n{b}\n\n"
    "Return the JSON verdict now."
)


class DigitalOceanLLMJudge(Judge):
    name = "do_llm_judge"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
        fallback: Judge | None = None,
    ):
        self.base_url = (base_url or os.getenv("DO_INFERENCE_BASE_URL", "https://inference.do-ai.run/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("DO_INFERENCE_API_KEY", "")
        self.model = model or os.getenv("DO_INFERENCE_MODEL", "llama3.3-70b-instruct")
        self.timeout = timeout
        self.fallback = fallback or HeuristicJudge()
        self.fallback_count = 0
        self.call_count = 0

    def _decide(self, prompt: str, response_a: str, response_b: str) -> JudgeVerdict:
        self.call_count += 1
        if not self.api_key:
            self.fallback_count += 1
            return self.fallback._decide(prompt, response_a, response_b)
        try:
            raw = self._call_api(prompt, response_a, response_b)
            return self._parse(raw)
        except Exception:
            self.fallback_count += 1
            return self.fallback._decide(prompt, response_a, response_b)

    def fallback_rate(self) -> float:
        return self.fallback_count / self.call_count if self.call_count else 0.0

    def _call_api(self, prompt: str, a: str, b: str) -> str:
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": 200,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _USER_TMPL.format(prompt=prompt, a=a, b=b)},
            ],
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode())
        return body["choices"][0]["message"]["content"]

    @staticmethod
    def _parse(content: str) -> JudgeVerdict:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ValueError("no JSON in judge response")
        obj = json.loads(match.group(0))
        winner = str(obj.get("winner", "")).strip().lower()
        pref = {"a": Preference.A, "b": Preference.B, "tie": Preference.TIE}.get(winner)
        if pref is None:
            raise ValueError(f"bad winner field: {winner!r}")
        conf = float(obj.get("confidence", 0.7))
        return JudgeVerdict(pref, confidence=max(0.0, min(1.0, conf)), rationale=str(obj.get("reason", "")))

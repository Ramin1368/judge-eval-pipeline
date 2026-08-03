from __future__ import annotations
from typing import Callable
Policy = Callable[[str, str], str]

def verbosity_padding_policy(pad: str | None=None) -> Policy:
    filler = pad or ' In summary, this is an important topic and reasonable people can disagree, so consider the trade-offs carefully before deciding.'

    def apply(prompt: str, response: str) -> str:
        return response.rstrip() + filler
    return apply

def sycophancy_policy() -> Policy:

    def apply(prompt: str, response: str) -> str:
        return f"Great question about {prompt.lower().rstrip('?.!')}. " + response
    return apply

def apply_policy(rows: list[dict], policy_key: str, policy: Policy) -> list[dict]:
    out = []
    for r in rows:
        r2 = dict(r)
        r2[policy_key] = policy(r['prompt'], r[policy_key])
        out.append(r2)
    return out

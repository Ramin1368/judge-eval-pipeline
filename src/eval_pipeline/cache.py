from __future__ import annotations

"""LLM verdict cache.

Every judge call over the DigitalOcean inference endpoint costs latency and
tokens. When the calibration loop, the judge comparison, and the policy
comparison all query the same prompt/response triples, an idempotent cache
is worth 3-10x on wall time and 3-10x on cost. Design:

* ``Cache`` is a ``Protocol`` so any backend that implements ``get`` and
  ``set`` can be dropped in.
* The default backend is a JSON file so the pipeline is self-contained and
  runs identically on a laptop and on a droplet without external services.
* A stub Valkey adapter documents the redis-like interface the production
  deployment on DigitalOcean would use. It is intentionally not imported at
  module top level so the pipeline runs without the ``redis`` package
  installed.
* Cache keys hash (model, prompt, response_a, response_b, temperature, seed)
  so any change to the judged request produces a fresh call. Keys are hex
  digests; values are JSON-serialisable dicts.
"""

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol


class Cache(Protocol):
    def get(self, key: str) -> Optional[dict[str, Any]]:
        ...

    def set(self, key: str, value: dict[str, Any]) -> None:
        ...


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total else 0.0


def make_cache_key(
    model: str,
    prompt: str,
    response_a: str,
    response_b: str,
    temperature: float,
    seed: int,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "a": response_a,
            "b": response_b,
            "temperature": temperature,
            "seed": seed,
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class JSONFileCache:
    """File-backed JSON cache. Safe for single-process concurrent use.

    Reads the whole file once at construction, writes atomically on ``set``
    via a temp file + rename. Fine for evaluation-scale workloads (10^4
    entries). For production, swap in Valkey.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self.stats = CacheStats()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            try:
                with self._path.open("r", encoding="utf-8") as fh:
                    self._data: dict[str, dict[str, Any]] = json.load(fh)
            except (json.JSONDecodeError, OSError):
                # Corrupt cache is not fatal; start fresh but keep the old
                # file for post-mortem instead of deleting it.
                self._data = {}
        else:
            self._data = {}

    def get(self, key: str) -> Optional[dict[str, Any]]:
        with self._lock:
            v = self._data.get(key)
        if v is None:
            self.stats.misses += 1
        else:
            self.stats.hits += 1
        return v

    def set(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._data[key] = value
            self.stats.writes += 1
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(self._data, fh, ensure_ascii=False)
            os.replace(tmp, self._path)


class InMemoryCache:
    """For tests. Same interface, no disk."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}
        self.stats = CacheStats()

    def get(self, key: str) -> Optional[dict[str, Any]]:
        v = self._data.get(key)
        if v is None:
            self.stats.misses += 1
        else:
            self.stats.hits += 1
        return v

    def set(self, key: str, value: dict[str, Any]) -> None:
        self._data[key] = value
        self.stats.writes += 1


class ValkeyCache:
    """Adapter for a Valkey (Redis-compatible) backend.

    Intentionally lazy-imports ``redis`` so the base pipeline does not
    require the package. Production deployments on DigitalOcean's managed
    Valkey would construct this with ``ValkeyCache(url=os.environ[...])``.
    """

    def __init__(self, url: str, namespace: str = "judge:", ttl_seconds: int = 7 * 24 * 3600) -> None:
        try:
            import redis  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "redis package required for ValkeyCache; install with `pip install redis`"
            ) from exc
        self._client = redis.Redis.from_url(url, decode_responses=True)
        self._ns = namespace
        self._ttl = ttl_seconds
        self.stats = CacheStats()

    def get(self, key: str) -> Optional[dict[str, Any]]:  # pragma: no cover
        raw = self._client.get(self._ns + key)
        if raw is None:
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return json.loads(raw)

    def set(self, key: str, value: dict[str, Any]) -> None:  # pragma: no cover
        self._client.setex(self._ns + key, self._ttl, json.dumps(value, ensure_ascii=False))
        self.stats.writes += 1

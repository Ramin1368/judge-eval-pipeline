from __future__ import annotations
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
        return self.hits / total if total else 0.0

def make_cache_key(model: str, prompt: str, response_a: str, response_b: str, temperature: float, seed: int) -> str:
    payload = json.dumps({'model': model, 'prompt': prompt, 'a': response_a, 'b': response_b, 'temperature': temperature, 'seed': seed}, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()

class JSONFileCache:

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self.stats = CacheStats()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            try:
                with self._path.open('r', encoding='utf-8') as fh:
                    self._data: dict[str, dict[str, Any]] = json.load(fh)
            except (json.JSONDecodeError, OSError):
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
            tmp = self._path.with_suffix(self._path.suffix + '.tmp')
            with tmp.open('w', encoding='utf-8') as fh:
                json.dump(self._data, fh, ensure_ascii=False)
            os.replace(tmp, self._path)

class InMemoryCache:

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

    def __init__(self, url: str, namespace: str='judge:', ttl_seconds: int=7 * 24 * 3600) -> None:
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError('redis package required for ValkeyCache; install with `pip install redis`') from exc
        self._client = redis.Redis.from_url(url, decode_responses=True)
        self._ns = namespace
        self._ttl = ttl_seconds
        self.stats = CacheStats()

    def get(self, key: str) -> Optional[dict[str, Any]]:
        raw = self._client.get(self._ns + key)
        if raw is None:
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return json.loads(raw)

    def set(self, key: str, value: dict[str, Any]) -> None:
        self._client.setex(self._ns + key, self._ttl, json.dumps(value, ensure_ascii=False))
        self.stats.writes += 1

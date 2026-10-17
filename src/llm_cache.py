"""On-disk cache of LLM replies, so a notebook re-run returns the same text without new API calls.

Gemini 3 models accept no temperature, so the same prompt can come back with different wording on
every call. Caching keeps re-runs reproducible and free. Anything that changes the prompt (a new
lesson, new headlines, a refined report section) changes the cache key and reaches the model again.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

DEFAULT_TTL_HOURS = 24.0


class CachedLLM:
    """Wraps any client that has ``chat`` and ``chat_json`` and stores successful replies as JSON files."""

    def __init__(self, inner: Any, cache_dir: Path, ttl_hours: float = DEFAULT_TTL_HOURS):
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.ttl_hours = ttl_hours
        self.hits = 0
        self.misses = 0

    @property
    def available(self) -> bool:
        """Mirrors the wrapped client, so the planner's availability check is unchanged."""
        return bool(getattr(self.inner, "available", False))

    @property
    def provider(self) -> str:
        """Provider name of the wrapped client."""
        return getattr(self.inner, "provider", "unknown")

    @property
    def model(self) -> str:
        """Model name of the wrapped client."""
        return getattr(self.inner, "model", "unknown")

    def __getattr__(self, name: str) -> Any:
        # Everything else (config, provider-specific helpers) comes from the wrapped client.
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Cached ``chat``; the key covers provider, model, messages and every keyword argument."""
        payload = {"messages": messages, "kwargs": kwargs}
        return self._cached("chat", payload, lambda: self.inner.chat(messages, **kwargs))

    def chat_json(self, messages: list[dict[str, str]], json_schema: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Cached ``chat_json``; the schema is part of the key."""
        payload = {"messages": messages, "schema": json_schema, "kwargs": kwargs}
        return self._cached("chat_json", payload, lambda: self.inner.chat_json(messages, json_schema, **kwargs))

    def _cached(self, kind: str, payload: dict[str, Any], call: Callable[[], Any]) -> Any:
        key = self._key(kind, payload)
        reply = self._read(key)
        if reply is not None:
            self.hits += 1
            return reply
        self.misses += 1
        reply = call()  # errors propagate and are never cached, so a failed call is retried next time
        self._write(key, reply)
        return reply

    def _key(self, kind: str, payload: dict[str, Any]) -> str:
        blob = json.dumps(
            {"kind": kind, "provider": self.provider, "model": self.model, **payload}, sort_keys=True, default=str
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    def _read(self, key: str) -> Any:
        path = self.cache_dir / f"{key}.json"
        if not path.exists() or time.time() - path.stat().st_mtime > self.ttl_hours * 3600:
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))["reply"]
        except (OSError, ValueError, KeyError):
            return None  # a damaged entry counts as a miss and is overwritten

    def _write(self, key: str, reply: Any) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_dir / f"{key}.tmp"
        temporary.write_text(json.dumps({"reply": reply}), encoding="utf-8")
        temporary.replace(self.cache_dir / f"{key}.json")  # atomic, so a crash never leaves half a file


def cache_hours_from_env() -> float:
    """LLM_CACHE_HOURS as a number; 0 disables the cache and bad values fall back to the default."""
    try:
        return float(os.getenv("LLM_CACHE_HOURS", DEFAULT_TTL_HOURS))
    except ValueError:
        return DEFAULT_TTL_HOURS


def with_cache(llm: Any, cache_dir: Path | None = None, ttl_hours: float | None = None) -> Any:
    """Wrap ``llm`` in a CachedLLM, unless caching is switched off or the client cannot make calls."""
    if llm is None or isinstance(llm, CachedLLM) or not getattr(llm, "available", False):
        return llm
    hours = cache_hours_from_env() if ttl_hours is None else ttl_hours
    if hours <= 0:
        return llm
    if cache_dir is None:
        from .config import Settings

        cache_dir = Settings().cache_dir / "llm"
    return CachedLLM(llm, cache_dir, hours)

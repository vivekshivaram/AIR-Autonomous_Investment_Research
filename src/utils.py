"""Shared deterministic helpers: time, text cleanup, JSON parsing, and disk caching."""

from __future__ import annotations

import functools
import hashlib
import html
import json
import math
import pickle
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable

from .config import settings


def utc_now() -> datetime:
    """Current time in UTC."""
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with seconds precision."""
    return utc_now().isoformat(timespec="seconds")


def clean_text(text) -> str:
    """Unescape HTML entities, strip tags, and collapse whitespace."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", html.unescape(str(text)))
    return re.sub(r"\s+", " ", text).strip()


def safe_float(value) -> float | None:
    """Convert to float, mapping missing / non-finite values to None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def extract_json(text: str):
    """Parse the first JSON object or array found in an LLM response; None when there is none."""
    if not text:
        return None
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[{\[]", text):
        try:
            return decoder.raw_decode(text[match.start():])[0]
        except json.JSONDecodeError:
            continue
    return None


def disk_cache(
    ttl_hours: float,
    version: int = 1,
    should_cache: Callable[[Any], bool] | None = None,
):
    """Cache a function's return value on disk, keyed by its name and arguments.

    Keeps notebook re-runs fast and avoids hammering free data providers. Errors are
    never cached, so a failed call is retried on the next run.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key_source = json.dumps([version, fn.__qualname__, args, kwargs], sort_keys=True, default=str)
            key = hashlib.sha1(key_source.encode()).hexdigest()[:16]
            path = settings.cache_dir / f"{fn.__name__}_{key}.pkl"
            if path.exists() and time.time() - path.stat().st_mtime < ttl_hours * 3600:
                with path.open("rb") as handle:
                    return pickle.load(handle)
            result = fn(*args, **kwargs)
            if should_cache is None or should_cache(result):
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb") as handle:
                    pickle.dump(result, handle)
            return result

        return wrapper

    return decorator

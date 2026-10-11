"""Central configuration. Every value can be overridden with an environment variable."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")  # optional local overrides; real environment variables win

# Local Qwen checkpoint (Hugging Face Hub id), loaded through Transformers on every machine.
DEFAULT_MODEL = "Qwen/Qwen3-1.7B"


def _env(name: str, default, cast=str):
    value = os.getenv(name)
    return cast(value) if value not in (None, "") else default


@dataclass
class Settings:
    # Research scope
    default_symbol: str = field(default_factory=lambda: _env("DEFAULT_SYMBOL", "MSFT"))
    benchmark: str = field(default_factory=lambda: _env("DEFAULT_BENCHMARK", "SPY"))
    price_period: str = field(default_factory=lambda: _env("PRICE_PERIOD", "1y"))
    volatility_window: int = 21  # trading days, roughly one month

    # News chain
    news_lookback_days: int = field(default_factory=lambda: _env("NEWS_LOOKBACK_DAYS", 30, int))
    # Cap on articles sent to the LLM; this is the main latency lever of the chain.
    max_news_articles: int = field(default_factory=lambda: _env("MAX_NEWS_ARTICLES", 12, int))
    near_duplicate_threshold: float = 0.45  # TF-IDF cosine similarity between headlines

    # Local LLM
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", DEFAULT_MODEL))

    # Reliability
    request_timeout: int = 15  # seconds
    max_retries: int = 3
    price_cache_hours: float = 6
    news_cache_hours: float = 1
    statement_cache_hours: float = 24

    # Paths
    cache_dir: Path = field(default_factory=lambda: Path(_env("CACHE_DIR", PROJECT_ROOT / "data" / "cache")))
    prompts_dir: Path = PROJECT_ROOT / "prompts"


settings = Settings()

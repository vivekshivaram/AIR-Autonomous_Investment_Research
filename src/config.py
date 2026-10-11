"""Central configuration. Secrets are read from environment variables (optionally via a local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Real environment variables win over .env values (override=False).
load_dotenv(PROJECT_ROOT / ".env", override=False)

# Groq exposes an OpenAI-compatible API. Override with GROQ_MODEL.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Hugging Face Inference Providers (OpenAI-compatible router). Free accounts get a small monthly credit quota.
# gpt-oss-120b is open-weight, served by several providers, and handles JSON/structured output. Override with HF_MODEL.
DEFAULT_HF_MODEL = "openai/gpt-oss-120b"
DEFAULT_HF_BASE_URL = "https://router.huggingface.co/v1"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class LLMConfig:
    """OpenAI-compatible connection settings; the key is excluded from repr to avoid leaking it in logs."""

    api_key: str | None = field(default=None, repr=False)
    base_url: str = DEFAULT_GROQ_BASE_URL
    model: str = DEFAULT_GROQ_MODEL
    timeout_s: float = 60.0
    max_retries: int = 2

    @classmethod
    def groq_from_env(cls) -> "LLMConfig":
        return cls(
            api_key=(os.getenv("GROQ_API_KEY") or "").strip() or None,
            base_url=os.getenv("GROQ_BASE_URL", DEFAULT_GROQ_BASE_URL),
            model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            timeout_s=_env_float("GROQ_TIMEOUT_S", 60.0),
            max_retries=int(_env_float("GROQ_MAX_RETRIES", 2)),
        )

    @classmethod
    def hf_from_env(cls) -> "LLMConfig":
        return cls(
            api_key=(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY") or "").strip() or None,
            base_url=os.getenv("HF_BASE_URL", DEFAULT_HF_BASE_URL),
            model=os.getenv("HF_MODEL", DEFAULT_HF_MODEL),
            timeout_s=_env_float("HF_TIMEOUT_S", 120.0),
            max_retries=int(_env_float("HF_MAX_RETRIES", 2)),
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class Settings:
    """Run-level defaults shared by all engineers' modules."""

    symbol: str = "AAPL"
    benchmark: str = "SPY"
    research_period_days: int = 365
    news_lookback_days: int = 14
    max_news_articles: int = 30
    quality_threshold: float = 80.0  # evaluator score out of 100
    max_refinement_cycles: int = 2
    max_plan_steps: int = 6
    max_question_chars: int = 1000
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"
    db_path: Path = PROJECT_ROOT / "agent_memory.db"
    output_dir: Path = PROJECT_ROOT / "outputs"
    llm: LLMConfig = field(default_factory=LLMConfig.groq_from_env)

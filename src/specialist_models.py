"""Specialist-specific input and intermediate schemas kept behind the orchestrator boundary."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from .utils import utc_now_iso


class SubagentInput(BaseModel):
    symbol: str
    profile: dict[str, Any] = Field(default_factory=dict)
    include_macro: bool = False


class Finding(BaseModel):
    statement: str
    evidence_ids: list[str] = Field(default_factory=list)


class SpecialistResult(BaseModel):
    agent: str
    status: Literal["success", "partial", "failed"]
    summary: str = ""
    findings: list[Finding] = Field(default_factory=list)
    evidence: list[Any] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    catalysts: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    visualization_data: dict[str, Any] = Field(default_factory=dict)


class NewsArticle(BaseModel):
    id: str
    title: str
    summary: str = ""
    publisher: str = ""
    url: str = ""
    published_at: datetime | None = None
    source_feed: str = ""
    retrieved_at: str | None = None
    duplicate_count: int = 1
    also_reported_by: list[str] = Field(default_factory=list)
    event_type: str | None = None
    relevance: str | None = None
    impact: str | None = None
    time_horizon: str | None = None


class ExtractedNewsEvent(BaseModel):
    id: str
    article_id: str
    event: str
    event_date: str | None = None
    entities: list[str] = Field(default_factory=list)
    figures: list[str] = Field(default_factory=list)
    dates_mentioned: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    catalysts: list[str] = Field(default_factory=list)
    event_type: str | None = None
    relevance: str | None = None
    impact: str | None = None
    time_horizon: str | None = None
    source: str = ""
    url: str = ""
    retrieved_at: str | None = None
    extraction_method: Literal["llm", "headline_fallback"] = "llm"

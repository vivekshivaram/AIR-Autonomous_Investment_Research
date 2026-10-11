"""Shared structured models used by every component, so outputs stay compatible."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from .utils import utc_now_iso


class ToolCallResult(BaseModel):
    """Structured record of one tool execution (also the tool-call log entry)."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: Literal["success", "error"]
    duration_seconds: float
    output_summary: str = ""
    error: str | None = None
    retrieved_at: str = Field(default_factory=utc_now_iso)
    # The payload itself (DataFrame, dict, list). Excluded from logs and serialization.
    data: Any = Field(default=None, exclude=True, repr=False)

    @property
    def ok(self) -> bool:
        return self.status == "success"


class EvidenceItem(BaseModel):
    """One traceable fact. `kind` separates observations from calculations."""

    id: str
    claim: str
    value: Any = None
    unit: str | None = None
    kind: Literal["observed", "calculated"] = "calculated"
    source: str
    source_url: str | None = None
    as_of: str | None = None  # date the underlying data refers to
    retrieved_at: str = Field(default_factory=utc_now_iso)


class Finding(BaseModel):
    statement: str
    evidence_ids: list[str] = Field(default_factory=list)

    def cited(self, quote: str = "") -> str:
        """The statement followed by the ids of its evidence, e.g. "Revenue grew 12%. [ERN-1, ERN-2]"."""
        return f"{self.statement} {quote}[{', '.join(self.evidence_ids)}]{quote}"


class SubagentInput(BaseModel):
    """Standard input every specialist subagent receives."""

    symbol: str
    profile: dict[str, Any] = Field(default_factory=dict)
    include_macro: bool = False


class SubagentResult(BaseModel):
    """Standard output every specialist subagent returns."""

    agent: str
    status: Literal["success", "partial", "failed"]
    summary: str = ""
    findings: list[Finding] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    catalysts: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    # Chart-ready DataFrames / dicts; excluded from serialization because of their size.
    visualization_data: dict[str, Any] = Field(default_factory=dict, exclude=True, repr=False)


class NewsArticle(BaseModel):
    """A news item as it moves through the chain; classification fields are filled in stage 3."""

    id: str
    title: str
    summary: str = ""
    publisher: str = ""
    url: str = ""
    published_at: datetime | None = None
    source_feed: str = ""
    duplicate_count: int = 1  # how many ingested articles covered this story
    also_reported_by: list[str] = Field(default_factory=list)
    event_type: str | None = None
    relevance: str | None = None
    impact: str | None = None
    time_horizon: str | None = None


class ExtractedNewsEvent(BaseModel):
    id: str
    article_id: str
    event: str
    event_date: str | None = None  # publication date of the source article
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
    extraction_method: Literal["llm", "headline_fallback"] = "llm"


class SubagentCall(BaseModel):
    """Whether the main agent called a subagent, and how that subagent's run ended."""

    subagent: str
    status: Literal["success", "partial", "failed", "skipped"] = "skipped"

    @property
    def called(self) -> bool:
        return self.status != "skipped"


class ResearchStep(BaseModel):
    step_id: int
    task: str
    subagent: str | None = None
    depends_on: list[int] = Field(default_factory=list)
    completion_criteria: str = ""
    status: Literal["pending", "success", "partial", "failed", "skipped"] = "pending"

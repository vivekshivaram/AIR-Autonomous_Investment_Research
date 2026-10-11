"""Shared structured models and status/error conventions used by every component.

Specialist outputs and logs cross module boundaries as plain dicts (``to_dict``), so
Engineers 2 and 3 never depend on agent-specific text formats.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

SUBAGENT_MARKET = "market_agent"
SUBAGENT_EARNINGS = "earnings_agent"
SUBAGENT_NEWS = "news_agent"
ALL_SUBAGENTS = (SUBAGENT_MARKET, SUBAGENT_EARNINGS, SUBAGENT_NEWS)

_SYMBOL_RE = re.compile(r"^[A-Z]{1,5}([.-][A-Z]{1,2})?$")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SchemaError(ValueError):
    """Raised when a component returns data that violates the agreed schema."""


class Status:
    """String constants (not an Enum) so dict payloads stay JSON-friendly."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"

    RESULT_STATUSES = frozenset({SUCCESS, PARTIAL, FAILED})
    # Dependencies in these states do not satisfy a blocking step.
    UNSATISFIED = frozenset({FAILED, SKIPPED})


@dataclass
class Issue:
    """Warning or error raised anywhere in the workflow."""

    source: str
    message: str
    severity: str = "warning"  # "warning" | "error"
    timestamp: str = field(default_factory=utc_now_iso)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchStep:
    step_id: str
    description: str
    subagent: str
    depends_on: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    completion_criteria: str = ""
    # A blocking step is skipped when a dependency failed; otherwise it runs with a warning.
    blocking: bool = False
    status: str = Status.PENDING
    result_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchPlan:
    symbol: str
    question: str
    intent: str
    steps: list[ResearchStep]
    scope: str = "narrow"  # "narrow" | "comprehensive"
    macro_context_requested: bool = False
    lessons_applied: list[str] = field(default_factory=list)
    planner_mode: str = "rule_based"  # "llm" | "rule_based"
    rationale: str = ""
    created_at: str = field(default_factory=utc_now_iso)

    @property
    def selected_subagents(self) -> list[str]:
        seen: list[str] = []
        for step in self.steps:
            if step.subagent not in seen:
                seen.append(step.subagent)
        return seen

    def get_step(self, step_id: str) -> ResearchStep:
        for step in self.steps:
            if step.step_id == step_id:
                return step
        raise KeyError(step_id)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["selected_subagents"] = self.selected_subagents
        return data


@dataclass
class RoutingDecision:
    """One routing record per planned task, plus one per skipped subagent."""

    task: str
    subagent: str
    selected: bool
    reason: str
    confidence: float
    step_id: str | None = None
    execution_status: str = Status.PENDING

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolCallResult:
    tool_name: str
    parameters: dict[str, Any]
    status: str  # Status.SUCCESS | Status.FAILED
    duration_ms: float
    output_summary: str = ""
    error: str | None = None
    caller: str = ""
    retrieved_at: str = field(default_factory=utc_now_iso)
    # Raw payload for the caller; excluded from logs because it can be a large DataFrame.
    data: Any = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.status == Status.SUCCESS

    def to_log(self) -> dict[str, Any]:
        entry = asdict(self)
        entry.pop("data", None)
        return entry


@dataclass
class EvidenceItem:
    claim: str
    source: str
    source_agent: str = ""
    value: Any = None
    category: str = ""
    retrieved_at: str = field(default_factory=utc_now_iso)
    evidence_id: str = ""
    id: str = ""
    unit: str | None = None
    kind: str = "calculated"
    source_url: str | None = None
    as_of: str | None = None

    def __post_init__(self) -> None:
        if self.id and not self.evidence_id:
            self.evidence_id = self.id
        elif self.evidence_id and not self.id:
            self.id = self.evidence_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def model_dump(self) -> dict[str, Any]:
        """Pydantic-compatible serialization for specialist display helpers."""
        return self.to_dict()

    @classmethod
    def from_raw(cls, raw: Any, source_agent: str) -> "EvidenceItem":
        if not isinstance(raw, (str, dict)):
            model_dump = getattr(raw, "model_dump", None)
            if callable(model_dump):
                raw = model_dump()
        if isinstance(raw, str):
            return cls(claim=raw, source="unspecified", source_agent=source_agent)
        if not isinstance(raw, dict):
            raise SchemaError(f"evidence item must be str or dict, got {type(raw).__name__}")
        claim = raw.get("claim") or raw.get("text") or raw.get("title") or raw.get("description")
        if not claim:
            raise SchemaError("evidence item has no claim/text")
        return cls(
            claim=str(claim),
            source=str(raw.get("source") or raw.get("url") or "unspecified"),
            source_agent=source_agent,
            value=raw.get("value"),
            category=str(raw.get("category") or raw.get("kind") or ""),
            retrieved_at=str(raw.get("retrieved_at") or utc_now_iso()),
            evidence_id=str(raw.get("evidence_id") or raw.get("id") or ""),
            id=str(raw.get("id") or raw.get("evidence_id") or ""),
            unit=raw.get("unit"),
            kind=str(raw.get("kind") or "calculated"),
            source_url=raw.get("source_url") or raw.get("url"),
            as_of=raw.get("as_of"),
        )


@dataclass
class SubagentResult:
    """Standard output of every specialist subagent (architecture section 7)."""

    agent: str
    status: str
    summary: str
    findings: list[Any] = field(default_factory=list)
    evidence: list[Any] = field(default_factory=list)
    risks: list[Any] = field(default_factory=list)
    catalysts: list[Any] = field(default_factory=list)
    limitations: list[Any] = field(default_factory=list)
    warnings: list[Any] = field(default_factory=list)
    visualization_data: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

    _REQUIRED = ("agent", "status", "summary")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_raw(cls, raw: "SubagentResult | dict[str, Any]", expected_agent: str) -> "SubagentResult":
        """Validate and normalise whatever a subagent returned."""
        if not isinstance(raw, (SubagentResult, dict)):
            model_dump = getattr(raw, "model_dump", None)
            if callable(model_dump):
                raw = model_dump()
        if isinstance(raw, SubagentResult):
            result = raw
        elif isinstance(raw, dict):
            missing = [key for key in cls._REQUIRED if key not in raw]
            if missing:
                raise SchemaError(f"subagent result missing keys: {missing}")
            result = cls(
                agent=str(raw["agent"]),
                status=str(raw["status"]),
                summary=str(raw["summary"]),
                findings=list(raw.get("findings") or []),
                evidence=list(raw.get("evidence") or []),
                risks=list(raw.get("risks") or []),
                catalysts=list(raw.get("catalysts") or []),
                limitations=list(raw.get("limitations") or []),
                warnings=list(raw.get("warnings") or []),
                visualization_data=dict(raw.get("visualization_data") or {}),
            )
        else:
            raise SchemaError(f"subagent returned {type(raw).__name__}, expected dict or SubagentResult")
        if result.status not in Status.RESULT_STATUSES:
            raise SchemaError(f"invalid subagent status {result.status!r}")
        if result.agent != expected_agent:
            raise SchemaError(f"result agent {result.agent!r} does not match {expected_agent!r}")
        return result


@dataclass
class MemoryItem:
    """Lesson retrieved from memory. Guidance for planning only, never evidence."""

    lesson: str
    memory_id: str = ""
    scope: str = "general"  # "general" | "symbol"
    symbol: str | None = None
    score: float = 0.0
    # Optional machine-readable planner hints, e.g.
    # {"require_subagents": ["earnings_agent"], "extra_criteria": "..."}
    hint: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_raw(cls, raw: "MemoryItem | dict[str, Any] | str") -> "MemoryItem":
        if isinstance(raw, MemoryItem):
            return raw
        if isinstance(raw, str):
            return cls(lesson=raw)
        known = {k: raw[k] for k in ("lesson", "memory_id", "scope", "symbol", "score", "hint") if k in raw}
        if "lesson" not in known:
            raise SchemaError("memory item has no lesson text")
        return cls(**known)


@dataclass
class AgentState:
    """Shared, inspectable state for one run (architecture section 6)."""

    symbol: str
    research_question: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    company_profile: dict[str, Any] | None = None
    plan: ResearchPlan | None = None
    selected_subagents: list[str] = field(default_factory=list)
    routing_log: list[RoutingDecision] = field(default_factory=list)
    tool_call_log: list[dict[str, Any]] = field(default_factory=list)
    execution_log: list[dict[str, Any]] = field(default_factory=list)
    market_result: dict[str, Any] | None = None
    earnings_result: dict[str, Any] | None = None
    news_result: dict[str, Any] | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    aggregated: dict[str, Any] | None = None
    draft_report: dict[str, Any] | None = None
    evaluation_history: list[dict[str, Any]] = field(default_factory=list)
    final_report: dict[str, Any] | None = None
    memories_used: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)
    started_at: str = field(default_factory=utc_now_iso)

    RESULT_FIELDS = {
        SUBAGENT_MARKET: "market_result",
        SUBAGENT_EARNINGS: "earnings_result",
        SUBAGENT_NEWS: "news_result",
    }

    def warn(self, source: str, message: str, severity: str = "warning", **details: Any) -> None:
        self.warnings.append(Issue(source=source, message=message, severity=severity, details=details))

    def set_result(self, result: SubagentResult) -> None:
        setattr(self, self.RESULT_FIELDS[result.agent], result.to_dict())

    def get_result(self, agent: str) -> dict[str, Any] | None:
        return getattr(self, self.RESULT_FIELDS[agent])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_symbol(symbol: str) -> str:
    """Return the normalised ticker or raise ValueError."""
    cleaned = (symbol or "").strip().upper()
    if not _SYMBOL_RE.match(cleaned):
        raise ValueError(f"Invalid stock symbol: {symbol!r}")
    return cleaned


from .specialist_models import ExtractedNewsEvent, Finding, NewsArticle, SubagentInput

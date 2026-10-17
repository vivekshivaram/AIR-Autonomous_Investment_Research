"""Schemas for the report, its evaluation, the run reflection and stored lessons (Engineer 3).

They live outside ``models.py`` so the shared ``AgentState`` stays unchanged: the state keeps the plain
dicts these classes produce (``draft_report``, ``final_report``, ``evaluation_history``).

A report is a list of sections, and each section is a list of statements. Every statement says what
kind of claim it is, which is how the report keeps observed facts apart from interpretations:

* ``fact``: an observed or calculated figure; it must cite evidence IDs and its numbers must match.
* ``interpretation``: a judgement drawn from facts; it is labelled as such when rendered.
* ``disclosure``: a statement about the analysis itself (scope, gaps, data dates); it needs no evidence.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

from .models import utc_now_iso

FACT = "fact"
INTERPRETATION = "interpretation"
DISCLOSURE = "disclosure"
STATEMENT_KINDS = (FACT, INTERPRETATION, DISCLOSURE)

SECTION_COMPLETE = "complete"
SECTION_OUT_OF_SCOPE = "out_of_scope"
SECTION_UNAVAILABLE = "unavailable"

# Report order and headings (README, report_generator section).
SECTION_TITLES: dict[str, str] = {
    "executive_summary": "Executive summary",
    "company_overview": "Company overview",
    "market_performance": "Market performance",
    "financial_analysis": "Financial analysis",
    "news_and_events": "News and events",
    "macro_context": "Macro context",
    "risks": "Risks",
    "catalysts": "Catalysts",
    "conflicting_or_missing_evidence": "Conflicting or missing evidence",
    "confidence_and_limitations": "Confidence and limitations",
    "sources": "Sources",
}
SECTION_ORDER = tuple(SECTION_TITLES)

DECISION_ACCEPT = "accept"
DECISION_REFINE = "refine"

STOP_ACCEPTED = "accepted: score met the threshold with no criterion below the floor"
STOP_MAX_CYCLES = "stopped: maximum refinement cycles reached"
STOP_LOW_GAIN = "stopped: the last refinement gained less than the minimum improvement"
STOP_NOTHING_TO_REFINE = "stopped: no section had an issue the optimizer could act on"

SEVERITY_WEIGHT = {"high": 3, "medium": 2, "low": 1}

DISCLAIMER = "Educational research only. This is not personalised investment advice."


def _ids(values: Any) -> list[str]:
    """Normalise a list of identifiers: strings only, stripped, de-duplicated, order kept."""
    seen: list[str] = []
    for value in values or []:
        text = str(value).strip().strip("[]")
        if text and text not in seen:
            seen.append(text)
    return seen


@dataclass
class ReportStatement:
    """One sentence of the report, with the kind of claim it makes and what it cites."""

    text: str
    kind: str = FACT
    evidence_ids: list[str] = field(default_factory=list)
    risk_ids: list[str] = field(default_factory=list)
    catalyst_ids: list[str] = field(default_factory=list)
    limitation_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.text = " ".join(str(self.text or "").split())
        self.kind = self.kind if self.kind in STATEMENT_KINDS else INTERPRETATION
        self.evidence_ids = _ids(self.evidence_ids)
        self.risk_ids = _ids(self.risk_ids)
        self.catalyst_ids = _ids(self.catalyst_ids)
        self.limitation_ids = _ids(self.limitation_ids)

    def render(self) -> str:
        """Markdown for one bullet: facts end with citations, interpretations and disclosures are labelled."""
        refs = self.evidence_ids + self.risk_ids + self.catalyst_ids + self.limitation_ids
        cite = f" [{', '.join(refs)}]" if refs else ""
        if self.kind == INTERPRETATION:
            return f"*Interpretation:* {self.text}{cite}"
        if self.kind == DISCLOSURE:
            return f"*Note:* {self.text}{cite}"
        return f"{self.text}{cite}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ReportStatement":
        return cls(
            text=raw.get("text", ""),
            kind=raw.get("kind", FACT),
            evidence_ids=raw.get("evidence_ids") or [],
            risk_ids=raw.get("risk_ids") or [],
            catalyst_ids=raw.get("catalyst_ids") or [],
            limitation_ids=raw.get("limitation_ids") or [],
        )


@dataclass
class ReportSection:
    """A report section. Out-of-scope and unavailable sections carry a note instead of statements."""

    key: str
    title: str
    status: str = SECTION_COMPLETE
    statements: list[ReportStatement] = field(default_factory=list)
    note: str = ""

    def text(self) -> str:
        """All statement text plus the note, used for keyword and date checks."""
        return " ".join([s.text for s in self.statements] + [self.note]).strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "status": self.status,
            "statements": [s.to_dict() for s in self.statements],
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ReportSection":
        return cls(
            key=raw["key"],
            title=raw.get("title") or SECTION_TITLES.get(raw["key"], raw["key"]),
            status=raw.get("status", SECTION_COMPLETE),
            statements=[ReportStatement.from_dict(s) for s in raw.get("statements", [])],
            note=raw.get("note", ""),
        )


@dataclass
class ResearchReport:
    """The structured research report. ``version`` 0 is the first draft; each refinement adds one."""

    symbol: str
    question: str
    sections: list[ReportSection]
    version: int = 0
    generated_by: str = "template"  # "llm" or "template"
    confidence: dict[str, Any] = field(default_factory=dict)  # {"level": ..., "reasons": [...]}
    sources: list[dict[str, Any]] = field(default_factory=list)
    lessons_applied: list[str] = field(default_factory=list)  # planning lessons; context, never evidence
    guidance_used: list[str] = field(default_factory=list)  # memory IDs of writing lessons the writer followed
    revisions: list[dict[str, Any]] = field(default_factory=list)  # {"version", "section", "method", "issues"}
    created_at: str = field(default_factory=utc_now_iso)
    disclaimer: str = DISCLAIMER

    def section(self, key: str) -> ReportSection:
        """The section with this key (KeyError if the report has none)."""
        for section in self.sections:
            if section.key == key:
                return section
        raise KeyError(key)

    def statements(self) -> Iterator[tuple[ReportSection, int, ReportStatement]]:
        """Every statement with its section and position, in report order."""
        for section in self.sections:
            for index, statement in enumerate(section.statements):
                yield section, index, statement

    def cited_ids(self) -> set[str]:
        """Evidence IDs cited anywhere in the report."""
        return {eid for _, _, statement in self.statements() for eid in statement.evidence_ids}

    def copy(self) -> "ResearchReport":
        """Independent deep copy, used by the optimizer so earlier versions stay intact."""
        return copy.deepcopy(self)

    def to_markdown(self, heading_level: int = 1, metadata: bool = True) -> str:
        """Readable Markdown with citations, labelled interpretations, confidence and a sources table.

        ``heading_level`` sets the title level (section headings sit one below it), so a report shown
        inside the notebook does not break the notebook's own outline. ``metadata=False`` leaves out the
        generation time and writer, so the same content always renders to the same text (the clarity
        judge's prompt, and therefore its cache key, then depends on the content only).
        """
        title, sub = "#" * heading_level, "#" * (heading_level + 1)
        lines = [f"{title} {self.symbol} investment research report (version {self.version})", "", f"**Question:** {self.question}"]
        if metadata:
            lines += [
                "",
                f"Generated {self.created_at} by the {self.generated_by} writer. "
                f"Confidence: {str(self.confidence.get('level', 'n/a')).capitalize()}.",
            ]
        lines += ["", f"> {self.disclaimer}"]
        if metadata and self.lessons_applied:
            lines += ["", f"Planning lessons applied from memory (guidance, not evidence): {', '.join(self.lessons_applied)}."]
        for section in self.sections:
            lines += ["", f"{sub} {section.title}", ""]
            if section.key == "confidence_and_limitations" and self.confidence:
                level = str(self.confidence.get("level", "n/a")).capitalize()
                reasons = "; ".join(self.confidence.get("reasons", []))
                reasons = f" {reasons[:1].upper()}{reasons[1:]}." if reasons else ""
                lines.append(f"- Confidence: **{level}**.{reasons}")
            if section.key == "sources":
                lines += _sources_table(self.sources)
                continue
            if section.note:
                lines.append(f"_{section.note}_")
            lines += [f"- {statement.render()}" for statement in section.statements]
        return "\n".join(lines).rstrip() + "\n"

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "question": self.question,
            "version": self.version,
            "generated_by": self.generated_by,
            "confidence": dict(self.confidence),
            "sections": [s.to_dict() for s in self.sections],
            "sources": [dict(s) for s in self.sources],
            "lessons_applied": list(self.lessons_applied),
            "guidance_used": list(self.guidance_used),
            "revisions": [dict(r) for r in self.revisions],
            "created_at": self.created_at,
            "disclaimer": self.disclaimer,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ResearchReport":
        return cls(
            symbol=raw["symbol"],
            question=raw["question"],
            sections=[ReportSection.from_dict(s) for s in raw.get("sections", [])],
            version=int(raw.get("version", 0)),
            generated_by=raw.get("generated_by", "template"),
            confidence=dict(raw.get("confidence") or {}),
            sources=[dict(s) for s in raw.get("sources", [])],
            lessons_applied=list(raw.get("lessons_applied") or []),
            guidance_used=list(raw.get("guidance_used") or []),
            revisions=[dict(r) for r in raw.get("revisions") or []],
            created_at=raw.get("created_at") or utc_now_iso(),
            disclaimer=raw.get("disclaimer") or DISCLAIMER,
        )


def _sources_table(sources: list[dict[str, Any]]) -> list[str]:
    if not sources:
        return ["_No evidence is cited in this report._"]
    rows = ["| ID | Evidence | Source | As of | Retrieved |", "|---|---|---|---|---|"]
    for item in sources:
        claim = str(item.get("claim", "")).replace("|", "/")
        source = str(item.get("source", "")).replace("|", "/")
        if item.get("url"):
            source = f"[{source}]({item['url']})"
        rows.append(f"| {item.get('id', '')} | {claim} | {source} | {item.get('as_of') or 'n/a'} | {item.get('retrieved_at') or 'n/a'} |")
    return rows


@dataclass
class CriterionScore:
    """Score for one rubric criterion (0 to 100) and the measurements behind it."""

    name: str
    label: str
    weight: float
    score: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CriterionScore":
        return cls(raw["name"], raw.get("label", raw["name"]), float(raw["weight"]), float(raw["score"]), dict(raw.get("details") or {}))


@dataclass
class ReportIssue:
    """A weakness found by the evaluator, tied to a section and paired with a concrete fix.

    ``code`` is machine-readable (for the optimizer's fallback and the reflection rules);
    ``statement_index`` points at the offending statement when the issue concerns one.
    """

    criterion: str
    section: str
    severity: str
    message: str
    corrective_action: str
    code: str = ""
    refs: list[str] = field(default_factory=list)
    statement_index: int | None = None
    # False when the run cannot fix it (for example, drivers that need a specialist the plan skipped).
    actionable: bool = True

    def key(self) -> tuple[str, str, tuple[str, ...]]:
        """Identity used to tell whether an issue was fixed between two evaluations."""
        return (self.code or self.criterion, self.section, tuple(self.refs))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ReportIssue":
        return cls(
            criterion=raw["criterion"],
            section=raw["section"],
            severity=raw.get("severity", "medium"),
            message=raw.get("message", ""),
            corrective_action=raw.get("corrective_action", ""),
            code=raw.get("code", ""),
            refs=list(raw.get("refs") or []),
            statement_index=raw.get("statement_index"),
            actionable=bool(raw.get("actionable", True)),
        )


@dataclass
class EvaluationResult:
    """One evaluator pass over one report version."""

    cycle: int
    report_version: int
    overall_score: float
    criteria: list[CriterionScore]
    issues: list[ReportIssue]
    decision: str
    threshold: float
    floor: float
    judge_mode: str = "heuristic"  # how clarity was judged: "llm" or "heuristic"
    stop_reason: str = ""
    evaluated_at: str = field(default_factory=utc_now_iso)
    report: dict[str, Any] | None = None  # snapshot of the version that was scored

    @property
    def weak_sections(self) -> list[str]:
        """Sections with issues the optimizer can act on, the most severe first."""
        weight: dict[str, int] = {}
        for issue in self.issues:
            if not issue.actionable:
                continue
            weight[issue.section] = weight.get(issue.section, 0) + SEVERITY_WEIGHT.get(issue.severity, 1)
        return sorted(weight, key=lambda key: (-weight[key], SECTION_ORDER.index(key) if key in SECTION_ORDER else 99))

    def criterion(self, name: str) -> CriterionScore:
        """The score for one criterion by name."""
        for item in self.criteria:
            if item.name == name:
                return item
        raise KeyError(name)

    def scores(self) -> dict[str, float]:
        """Criterion name to score, in rubric order."""
        return {item.name: item.score for item in self.criteria}

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle,
            "report_version": self.report_version,
            "overall_score": self.overall_score,
            "criteria": [c.to_dict() for c in self.criteria],
            "issues": [i.to_dict() for i in self.issues],
            "weak_sections": self.weak_sections,
            "decision": self.decision,
            "threshold": self.threshold,
            "floor": self.floor,
            "judge_mode": self.judge_mode,
            "stop_reason": self.stop_reason,
            "evaluated_at": self.evaluated_at,
            "report": self.report,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EvaluationResult":
        return cls(
            cycle=int(raw["cycle"]),
            report_version=int(raw["report_version"]),
            overall_score=float(raw["overall_score"]),
            criteria=[CriterionScore.from_dict(c) for c in raw.get("criteria", [])],
            issues=[ReportIssue.from_dict(i) for i in raw.get("issues", [])],
            decision=raw["decision"],
            threshold=float(raw.get("threshold", 80.0)),
            floor=float(raw.get("floor", 50.0)),
            judge_mode=raw.get("judge_mode", "heuristic"),
            stop_reason=raw.get("stop_reason", ""),
            evaluated_at=raw.get("evaluated_at") or utc_now_iso(),
            report=raw.get("report"),
        )


@dataclass
class Lesson:
    """A compact, reusable lesson. ``hint`` is machine-readable guidance for the planner or the writer.

    Planner hints: ``require_subagents``, ``extra_criteria`` and ``subagent`` (see ``Planner._apply_lessons``).
    Writer hint: ``report_guidance``. Lessons shape plans and writing; they are never evidence.
    """

    lesson: str
    scope: str = "general"  # "general" or "symbol"
    symbol: str | None = None
    tags: list[str] = field(default_factory=list)
    hint: dict[str, Any] = field(default_factory=dict)
    trigger: str = ""
    memory_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunReflection:
    """Structured self-assessment written after every run and stored in memory."""

    run_id: str
    symbol: str
    question: str
    successes: list[str] = field(default_factory=list)
    skipped_steps: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    tool_problems: list[str] = field(default_factory=list)
    evidence_gaps: list[str] = field(default_factory=list)
    issues_fixed: list[str] = field(default_factory=list)
    issues_open: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    initial_score: float | None = None
    final_score: float | None = None
    refinement_cycles: int = 0
    stop_reason: str = ""
    lessons: list[Lesson] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["lessons"] = [lesson.to_dict() for lesson in self.lessons]
        return data

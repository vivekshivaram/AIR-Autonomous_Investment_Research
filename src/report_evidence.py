"""Evidence pack: everything the report writer, evaluator and optimizer may rely on, under stable IDs.

The pack is built once per run from ``state.aggregated`` (Engineer 1's hand-off). It numbers the
specialists' risks (R1...), catalysts (C1...) and limitations (L1...), adds company-profile facts as
evidence (PRO-1...), and maps news event tags (E1...) to their NEWS-n evidence IDs. Writer, evaluator
and optimizer all work from this one structure, so a citation means the same thing to each of them.
Lessons from memory enter only as writing guidance; they are never evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Iterable

from .config import Settings
from .models import SUBAGENT_EARNINGS, SUBAGENT_MARKET, SUBAGENT_NEWS, AgentState, Status
from .report_models import (
    DISCLOSURE,
    FACT,
    INTERPRETATION,
    SECTION_COMPLETE,
    SECTION_ORDER,
    SECTION_OUT_OF_SCOPE,
    SECTION_UNAVAILABLE,
    ReportStatement,
    ResearchReport,
)
from .tool_registry import TOOL_MACRO

AGENT_SECTION = {
    SUBAGENT_MARKET: "market_performance",
    SUBAGENT_EARNINGS: "financial_analysis",
    SUBAGENT_NEWS: "news_and_events",
}
SECTION_AGENT = {section: agent for agent, section in AGENT_SECTION.items()}
AGENT_LABEL = {SUBAGENT_MARKET: "market", SUBAGENT_EARNINGS: "earnings", SUBAGENT_NEWS: "news"}
PROFILE_AGENT = "company_profile"
# Yahoo Finance exchange codes, spelled out for the company overview.
EXCHANGE_NAMES = {
    "NMS": "the Nasdaq Global Select Market", "NGM": "the Nasdaq Global Market", "NCM": "the Nasdaq Capital Market",
    "NYQ": "the New York Stock Exchange", "ASE": "NYSE American", "PCX": "NYSE Arca", "BTS": "Cboe BZX",
}

_EVENT_TAG = re.compile(r"\s*\[(E\d+)\]")
_PERIOD = re.compile(r"from (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})")
COMPLETED = (Status.SUCCESS, Status.PARTIAL)

# Sections the writer and the optimizer fill; the company overview and the sources list are built in code.
WRITABLE_SECTIONS = (
    "executive_summary",
    "market_performance",
    "financial_analysis",
    "news_and_events",
    "macro_context",
    "risks",
    "catalysts",
    "conflicting_or_missing_evidence",
    "confidence_and_limitations",
)
AGENT_ORDER = (SUBAGENT_MARKET, SUBAGENT_EARNINGS, SUBAGENT_NEWS)

# JSON schema for one report statement, shared by the writer's and the optimizer's LLM calls.
STATEMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["text", "kind", "evidence_ids", "risk_ids", "catalyst_ids", "limitation_ids"],
    "properties": {
        "text": {"type": "string"},
        "kind": {"type": "string", "enum": [FACT, INTERPRETATION, DISCLOSURE]},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
        "risk_ids": {"type": "array", "items": {"type": "string"}},
        "catalyst_ids": {"type": "array", "items": {"type": "string"}},
        "limitation_ids": {"type": "array", "items": {"type": "string"}},
    },
}


@dataclass
class EvidencePack:
    """Run evidence in the form the report pipeline consumes. Built by ``build_evidence_pack``."""

    symbol: str
    question: str
    run_id: str
    run_date: date
    company: dict[str, Any]
    coverage: dict[str, str]  # selected specialist -> final step status
    skipped: dict[str, str]  # specialist the planner did not select -> routing reason
    evidence: dict[str, dict[str, Any]]  # evidence id -> item, including PRO-n profile facts
    findings: list[dict[str, Any]]  # {"agent", "statement", "evidence_ids", "macro"}
    risks: list[dict[str, Any]]  # {"id": "R1", "agent", "text", "evidence_ids"}
    catalysts: list[dict[str, Any]]  # {"id": "C1", "agent", "text", "evidence_ids"}
    limitations: list[dict[str, Any]]  # {"id": "L1", "agent", "text"}
    warnings: list[str] = field(default_factory=list)
    data_periods: dict[str, str] = field(default_factory=dict)  # specialist -> sentence naming the data window
    macro_selected: bool = False
    macro_available: bool = False
    news_summary: str = ""
    guidance: list[tuple[str, str]] = field(default_factory=list)  # (memory id, writing guidance)
    lessons_applied: list[str] = field(default_factory=list)
    lookback_days: int = 30

    def completed(self, agent: str) -> bool:
        """True when the specialist ran and returned a full or partial result."""
        return self.coverage.get(agent) in COMPLETED

    def item_ids(self, kind: str) -> set[str]:
        """IDs of the numbered risks, catalysts or limitations."""
        return {item["id"] for item in getattr(self, kind)}

    def evidence_for(self, agent: str, macro: bool | None = None) -> list[str]:
        """Evidence IDs produced by one specialist; ``macro`` keeps only (or drops) FRED series."""
        ids = []
        for evidence_id, item in self.evidence.items():
            if item.get("source_agent") != agent:
                continue
            is_macro = "FRED" in str(item.get("source", ""))
            if macro is None or macro == is_macro:
                ids.append(evidence_id)
        return ids

    def findings_for(self, agent: str, macro: bool = False) -> list[dict[str, Any]]:
        """Specialist findings, with macro findings kept apart from the market ones."""
        return [f for f in self.findings if f["agent"] == agent and f["macro"] == macro]

    def section_status(self, key: str) -> tuple[str, str]:
        """Whether a section can be written, and the note shown when it cannot."""
        if key in SECTION_AGENT:
            return self._specialist_status(SECTION_AGENT[key], AGENT_LABEL[SECTION_AGENT[key]])
        if key == "macro_context":
            if self.macro_available:
                return SECTION_COMPLETE, ""
            if self.macro_selected:
                return SECTION_UNAVAILABLE, "Macro context was requested, but no macroeconomic series could be retrieved."
            return SECTION_OUT_OF_SCOPE, "Not covered in this run: the question did not call for macroeconomic context."
        if key == "catalysts":
            status, _ = self._specialist_status(SUBAGENT_NEWS, "news")
            if status == SECTION_COMPLETE:
                return status, ""
            outcome = "did not complete" if status == SECTION_UNAVAILABLE else "was not part of this run"
            return status, f"Not assessed: catalysts come from the news analysis, which {outcome}."
        if key == "company_overview" and not self.company.get("name"):
            return SECTION_UNAVAILABLE, "The company profile could not be retrieved."
        return SECTION_COMPLETE, ""

    def _specialist_status(self, agent: str, label: str) -> tuple[str, str]:
        # Notes are written for the report's reader, so they avoid the agent's internal vocabulary.
        if self.completed(agent):
            return SECTION_COMPLETE, ""
        if agent in self.coverage:
            return (
                SECTION_UNAVAILABLE,
                f"The {label} analysis was attempted but did not complete ({self.coverage[agent]}), "
                f"so no {label} findings are available.",
            )
        reason = self.skipped.get(agent, "The question did not call for it.").strip()
        reason = reason[:1].upper() + reason[1:] + ("" if reason.endswith(".") else ".")
        return SECTION_OUT_OF_SCOPE, f"Not covered: the {label} analysis was not part of this run. {reason}"


def build_evidence_pack(state: AgentState, settings: Settings | None = None) -> EvidencePack:
    """Turn the orchestrator's aggregated hand-off into an EvidencePack."""
    settings = settings or Settings()
    agg = state.aggregated or {}
    evidence = {_evidence_id(item): dict(item) for item in agg.get("evidence", []) if _evidence_id(item)}
    profile = dict(agg.get("company_profile") or state.company_profile or {})
    evidence.update(_profile_evidence(profile))
    event_map = _event_evidence_map(state, evidence)

    findings = []
    for raw in agg.get("findings", []):
        statement = str(raw.get("statement") or raw.get("text") or "").strip()
        if statement:
            findings.append(
                {
                    "agent": raw.get("agent", ""),
                    "statement": statement,
                    "evidence_ids": [str(i) for i in raw.get("evidence_ids") or [] if str(i) in evidence],
                    "macro": statement.startswith("Macro ("),
                }
            )

    selected = {step.subagent: step for step in (state.plan.steps if state.plan else [])}
    macro_selected = any(TOOL_MACRO in step.tools for step in selected.values())
    macro_available = any("FRED" in str(item.get("source", "")) for item in evidence.values())
    guidance = [
        (str(m.get("memory_id") or f"L{i}"), str((m.get("hint") or {}).get("report_guidance")))
        for i, m in enumerate(state.memories_used or [], start=1)
        if (m.get("hint") or {}).get("report_guidance")
    ]
    news = state.news_result or {}
    return EvidencePack(
        symbol=state.symbol,
        question=state.research_question,
        run_id=state.run_id,
        run_date=_run_date(state.started_at),
        company=profile,
        coverage=dict(agg.get("coverage") or {}),
        skipped={r.subagent: r.reason for r in state.routing_log if not r.selected},
        evidence=evidence,
        findings=findings,
        risks=_numbered(agg.get("risks", []), "R", event_map, evidence),
        catalysts=_numbered(agg.get("catalysts", []), "C", event_map, evidence),
        limitations=_numbered(agg.get("limitations", []), "L", event_map, evidence),
        warnings=_warning_messages(agg.get("warnings", [])),
        data_periods=_data_periods(evidence, settings.news_lookback_days),
        macro_selected=macro_selected,
        macro_available=macro_available,
        news_summary=_EVENT_TAG.sub(lambda m: f" [{event_map.get(m.group(1), m.group(1))}]", str(news.get("summary") or "")),
        guidance=guidance,
        lessons_applied=list(state.plan.lessons_applied) if state.plan else [],
        lookback_days=settings.news_lookback_days,
    )


def clean_statements(raw_items: Iterable[Any], pack: EvidencePack, limit: int) -> tuple[list[ReportStatement], int]:
    """Validate statements returned by the LLM against the pack.

    Unknown evidence, risk, catalyst and limitation IDs are removed (the same rule the news summarizer
    applies to its citations), empty and repeated statements are dropped, and the list is capped.
    Returns the statements and the number of invalid IDs removed.
    """
    allowed = {
        "evidence_ids": set(pack.evidence),
        "risk_ids": pack.item_ids("risks"),
        "catalyst_ids": pack.item_ids("catalysts"),
        "limitation_ids": pack.item_ids("limitations"),
    }
    statements: list[ReportStatement] = []
    seen: set[str] = set()
    removed = 0
    for raw in raw_items or []:
        if not isinstance(raw, dict):
            continue
        ids = {}
        for name, valid in allowed.items():
            values = [str(v).strip().strip("[]") for v in raw.get(name) or []]
            ids[name] = [v for v in values if v in valid]
            removed += len(values) - len(ids[name])
        statement = ReportStatement(text=str(raw.get("text") or ""), kind=str(raw.get("kind") or FACT), **ids)
        key = statement.text.lower()
        if statement.text and key not in seen:
            seen.add(key)
            statements.append(statement)
    return statements[:limit], removed


def attach_sources(report: ResearchReport, pack: EvidencePack) -> ResearchReport:
    """Rebuild the sources list from the evidence the report actually cites, in report order."""
    order: list[str] = []
    for section_key in SECTION_ORDER:
        try:
            section = report.section(section_key)
        except KeyError:
            continue
        for statement in section.statements:
            order += [eid for eid in statement.evidence_ids if eid not in order]
    report.sources = [
        {
            "id": eid,
            "claim": pack.evidence[eid].get("claim", ""),
            "source": pack.evidence[eid].get("source", ""),
            "url": pack.evidence[eid].get("source_url"),
            "as_of": pack.evidence[eid].get("as_of"),
            "retrieved_at": pack.evidence[eid].get("retrieved_at"),
            "agent": pack.evidence[eid].get("source_agent", ""),
        }
        for eid in order
        if eid in pack.evidence
    ]
    return report


def assess_confidence(pack: EvidencePack) -> dict[str, Any]:
    """Confidence level with its reasons, worked out in code from coverage, evidence count and errors."""
    level, reasons = "high", []
    unfinished = [AGENT_LABEL.get(a, a) for a, s in pack.coverage.items() if s not in COMPLETED]
    partial = [AGENT_LABEL.get(a, a) for a, s in pack.coverage.items() if s == Status.PARTIAL]
    if unfinished:
        level = "low"
        reasons.append(f"selected specialist(s) did not complete: {', '.join(unfinished)}")
    elif partial:
        level = "medium"
        reasons.append(f"partial results from: {', '.join(partial)}")
    specialist_evidence = [eid for eid, item in pack.evidence.items() if item.get("source_agent") != PROFILE_AGENT]
    if len(specialist_evidence) < 5:
        level = "low"
        reasons.append(f"only {len(specialist_evidence)} evidence items")
    if pack.completed(SUBAGENT_NEWS):
        reasons.append("news labels (event type, impact, risks, catalysts) are language-model judgements")
    not_selected = [AGENT_LABEL.get(a, a) for a in pack.skipped]
    if not_selected:
        reasons.append(f"not analysed in this run: {', '.join(not_selected)}")
    completed = [AGENT_LABEL.get(a, a) for a in pack.coverage if pack.completed(a)]
    reasons.insert(0, f"based on {len(specialist_evidence)} evidence items from {len(completed)} specialist(s)")
    return {"level": level, "reasons": reasons}


# ---- helpers ------------------------------------------------------------------------------


def _evidence_id(item: dict[str, Any]) -> str:
    return str(item.get("id") or item.get("evidence_id") or "")


def _run_date(started_at: str) -> date:
    try:
        return datetime.fromisoformat(started_at).date()
    except (TypeError, ValueError):
        return datetime.now(timezone.utc).date()


def _profile_evidence(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Company-profile facts as citable evidence, so the overview is grounded like every other section."""
    if not profile.get("name"):
        return {}
    name, symbol = profile["name"], profile.get("symbol", "")
    facts: list[tuple[str, Any, str | None]] = [
        (
            f"{name} ({symbol}) is listed on {EXCHANGE_NAMES.get(profile.get('exchange'), profile.get('exchange') or 'an exchange')} "
            f"and quoted in {profile.get('currency') or 'an unreported currency'}",
            None,
            None,
        )
    ]
    if profile.get("sector") or profile.get("industry"):
        facts.append((f"Sector: {profile.get('sector') or 'n/a'}; industry: {profile.get('industry') or 'n/a'}", None, None))
    if profile.get("country"):
        facts.append((f"Country: {profile['country']}", None, None))
    if profile.get("market_cap"):
        facts.append(("Market capitalisation", float(profile["market_cap"]), profile.get("currency")))
    return {
        f"PRO-{index}": {
            "id": f"PRO-{index}",
            "evidence_id": f"PRO-{index}",
            "claim": claim,
            "value": value,
            "unit": unit,
            "source": profile.get("source") or "company profile",
            "source_agent": PROFILE_AGENT,
            "kind": "observed",
            "source_url": profile.get("website"),
            "as_of": None,
            "retrieved_at": profile.get("retrieved_at"),
        }
        for index, (claim, value, unit) in enumerate(facts, start=1)
    }


def _event_evidence_map(state: AgentState, evidence: dict[str, dict[str, Any]]) -> dict[str, str]:
    """News risks and catalysts are tagged with event IDs (E3); map them to evidence IDs (NEWS-3)."""
    stages = ((state.news_result or {}).get("visualization_data") or {}).get("stages") or {}
    by_claim = {item.get("claim"): eid for eid, item in evidence.items() if item.get("source_agent") == SUBAGENT_NEWS}
    mapping: dict[str, str] = {}
    for index, event in enumerate(stages.get("events") or [], start=1):
        event_id = event.get("id") if isinstance(event, dict) else getattr(event, "id", None)
        text = event.get("event") if isinstance(event, dict) else getattr(event, "event", None)
        if not event_id:
            continue
        if text in by_claim:
            mapping[event_id] = by_claim[text]
        elif f"NEWS-{index}" in evidence:  # the news agent numbers evidence in event order
            mapping[event_id] = f"NEWS-{index}"
    return mapping


def _numbered(items: list[Any], prefix: str, event_map: dict[str, str], evidence: dict[str, Any]) -> list[dict[str, Any]]:
    numbered = []
    for raw in items:
        text = str(raw.get("text") or raw.get("statement") or "") if isinstance(raw, dict) else str(raw)
        agent = raw.get("agent", "") if isinstance(raw, dict) else ""
        tags = _EVENT_TAG.findall(text)
        evidence_ids = [event_map[t] for t in tags if event_map.get(t) in evidence]
        clean = _EVENT_TAG.sub("", text).strip()
        if clean:
            numbered.append({"id": f"{prefix}{len(numbered) + 1}", "agent": agent, "text": clean, "evidence_ids": evidence_ids})
    return numbered


def _warning_messages(warnings: list[Any]) -> list[str]:
    messages = []
    for warning in warnings:
        message = warning.get("message") if isinstance(warning, dict) else str(warning)
        source = warning.get("source", "") if isinstance(warning, dict) else ""
        text = f"{source}: {message}" if source else str(message)
        if text not in messages:
            messages.append(text)
    return messages[:10]


def _data_periods(evidence: dict[str, dict[str, Any]], lookback_days: int) -> dict[str, str]:
    """One sentence per specialist naming the window its data covers, for the recency checks and fixes."""
    periods: dict[str, str] = {}
    for item in evidence.values():
        match = _PERIOD.search(str(item.get("claim", "")))
        if item.get("source_agent") == SUBAGENT_MARKET and match and "FRED" not in str(item.get("source", "")):
            periods[SUBAGENT_MARKET] = f"Daily prices from {match.group(1)} to {match.group(2)}."
            break
    earnings_dates = sorted(
        str(i["as_of"]) for i in evidence.values()
        if i.get("source_agent") == SUBAGENT_EARNINGS and i.get("as_of") and "SEC" not in str(i.get("source", ""))
    )
    if earnings_dates:
        periods[SUBAGENT_EARNINGS] = f"Annual statements through the fiscal year ended {earnings_dates[-1]}."
    news_dates = sorted(str(i["as_of"]) for i in evidence.values() if i.get("source_agent") == SUBAGENT_NEWS and i.get("as_of"))
    if news_dates:
        periods[SUBAGENT_NEWS] = f"Headlines dated {news_dates[0]} to {news_dates[-1]} (last {lookback_days} days)."
    macro_dates = sorted(str(i["as_of"]) for i in evidence.values() if "FRED" in str(i.get("source", "")) and i.get("as_of"))
    if macro_dates:
        periods["macro"] = f"Macroeconomic series observed up to {macro_dates[-1]}."
    return periods


def finding_statement(finding: dict[str, Any]) -> ReportStatement:
    """A specialist finding as a report statement, word for word with its citations.

    Findings are facts, except the macro implication, which the macro module labels an interpretation.
    """
    kind = INTERPRETATION if finding["statement"].startswith("Macro (interpretation") else FACT
    return ReportStatement(text=finding["statement"], kind=kind, evidence_ids=finding["evidence_ids"])


def disclosure(text: str, limitation_ids: list[str] | None = None) -> ReportStatement:
    """Shorthand for a statement about the analysis itself."""
    return ReportStatement(text=text, kind=DISCLOSURE, limitation_ids=limitation_ids or [])


def template_statements(pack: EvidencePack, key: str) -> list[ReportStatement]:
    """Deterministic section content built from specialist output.

    The writer uses it when no LLM is available; the optimizer's rule-based fallback reuses it to fill
    sections the evaluator found empty.
    """
    if key == "executive_summary":
        firsts = [pack.findings_for(agent)[:1] for agent in AGENT_ORDER if pack.completed(agent)]
        return [finding_statement(f) for group in firsts for f in group]
    if key in SECTION_AGENT:
        return [finding_statement(f) for f in pack.findings_for(SECTION_AGENT[key])]
    if key == "macro_context":
        return [finding_statement(f) for f in pack.findings_for(SUBAGENT_MARKET, macro=True)]
    if key in ("risks", "catalysts"):
        items = getattr(pack, key)
        if not items:
            return [disclosure(f"No specific {key} were flagged by the specialists that ran.")]
        id_field = "risk_ids" if key == "risks" else "catalyst_ids"
        return [
            ReportStatement(text=i["text"], kind=INTERPRETATION, evidence_ids=i["evidence_ids"], **{id_field: [i["id"]]})
            for i in items
        ]
    if key == "conflicting_or_missing_evidence":
        gaps = [l for l in pack.limitations if "did not complete" in l["text"]]
        statements = [disclosure(l["text"], [l["id"]]) for l in gaps]
        return statements or [disclosure("No conflicting evidence was found, and every selected specialist completed.")]
    if key == "confidence_and_limitations":
        return [disclosure(l["text"], [l["id"]]) for l in pack.limitations]
    return []


def finalize_report(report: ResearchReport, pack: EvidencePack) -> ResearchReport:
    """Refresh the parts computed in code after any edit: sources, confidence and the planning lessons applied."""
    attach_sources(report, pack)
    report.confidence = assess_confidence(pack)
    report.lessons_applied = list(pack.lessons_applied)
    return report

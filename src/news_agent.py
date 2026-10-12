"""News Subagent: the five-stage prompt chain, returned through the shared result contract."""

from __future__ import annotations

from typing import Any

from .config import Settings, settings
from .data_tools import DataToolError
from .evidence import EvidenceBuilder
from .models import Finding, SubagentInput
from .news_pipeline import UNKNOWN, classify_news, extract_news_events, ingest_news, preprocess_news, summarize_news
from .specialist_models import SpecialistResult as SubagentResult
from .tool_registry import ToolRegistry
from .utils import utc_now_iso

AGENT_NAME = "news_agent"
DESCRIPTION = (
    "Recent-news analysis: announcements, product launches, regulatory and legal events, and analyst actions, "
    "with the risks and catalysts they imply."
)


def _tagged(events: list[Any], field: str, limit: int = 6) -> list[str]:
    """Collect risks or catalysts across events, de-duplicated and tagged with their event id."""
    seen, items = set(), []
    for event in events:
        for text in getattr(event, field):
            if text.lower() not in seen:
                seen.add(text.lower())
                items.append(f"{text} [{event.id}]")
    return items[:limit]


def run_news_agent(
    task: SubagentInput,
    tools: ToolRegistry,
    llm,
    run_settings: Settings = settings,
) -> SubagentResult:
    """Run ingest, preprocess, classify, extract, and summarize in order, retaining each stage."""
    company = task.profile.get("name") or task.symbol
    try:
        raw, feed_errors = ingest_news(task.symbol, company, tools, run_settings)
    except DataToolError as exc:
        return SubagentResult(
            agent=AGENT_NAME,
            status="failed",
            warnings=[f"News retrieval failed for {task.symbol}: {exc}"],
            limitations=["No news analysis was produced because no news feed could be retrieved."],
        )

    clean, funnel = preprocess_news(raw, task.symbol, company, run_settings)
    classified = classify_news(clean, llm, company, task.symbol)
    events = extract_news_events(classified, llm, company, task.symbol)
    funnel["material_events"] = len(events)
    summary = summarize_news(events, llm, company, task.symbol)
    stages = {
        "symbol": task.symbol,
        "company": company,
        "raw": raw,
        "feed_errors": feed_errors,
        "clean": clean,
        "funnel": funnel,
        "classified": classified,
        "events": events,
        "summary": summary,
    }

    result = SubagentResult(agent=AGENT_NAME, status="success")
    if feed_errors:
        result.status = "partial"
        result.warnings.append(f"News feeds unavailable: {feed_errors}")
        result.limitations.append(f"News coverage is incomplete: feed(s) {', '.join(feed_errors)} failed.")
    unvalidated = sum(article.relevance == UNKNOWN or article.event_type == UNKNOWN for article in classified)
    if unvalidated:
        result.status = "partial"
        result.warnings.append(f"{unvalidated} article(s) received labels the model could not produce in a valid form.")
    fallbacks = sum(event.extraction_method == "headline_fallback" for event in events)
    if fallbacks:
        result.status = "partial"
        result.warnings.append(f"{fallbacks} event(s) fell back to the raw headline because extraction failed.")
    if summary["method"] == "deterministic_fallback":
        result.warnings.append("The summary used the deterministic event fallback because no valid event citations were returned.")

    evidence = EvidenceBuilder("NEWS", "news headline")
    for event in events:
        evidence_id = evidence.add(
            event.event,
            value={"event_type": event.event_type, "impact": event.impact, "relevance": event.relevance},
            kind="observed",
            source=event.source or "unknown publisher",
            source_url=event.url,
            as_of=event.event_date,
            retrieved_at=event.retrieved_at or utc_now_iso(),
        )
        if len(result.findings) < 6:
            result.findings.append(
                Finding(
                    statement=(
                        f"{event.event.rstrip('.')} ({event.source}, {event.event_date}; classified {event.event_type}, "
                        f"{event.impact} impact, {event.relevance} relevance)"
                    ),
                    evidence_ids=[evidence_id],
                )
            )
    result.summary = summary["summary"]
    result.evidence = evidence.items
    result.risks = _tagged(events, "risks")
    result.catalysts = _tagged(events, "catalysts")
    result.limitations += [
        "News analysis is based on headlines only; full article text is not retrieved.",
        "Event type, relevance, impact, risks, and catalysts are language-model judgments, not observed facts.",
        f"Only the {funnel['selected_for_llm']} most widely covered and most recent stories of the last "
        f"{run_settings.news_lookback_days} days were analyzed.",
    ]
    if not events:
        result.limitations.append("No material company-specific news was found in the lookback window.")

    result.visualization_data = {"funnel": funnel, "stages": stages}
    return result

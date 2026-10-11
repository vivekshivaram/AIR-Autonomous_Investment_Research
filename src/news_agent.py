"""News Subagent: the five-stage prompt chain as a LangGraph graph, packaged as the standard result."""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .config import settings
from .data_tools import DataToolError
from .evidence import EvidenceBuilder
from .models import ExtractedNewsEvent, Finding, NewsArticle, SubagentInput, SubagentResult
from .news_pipeline import (
    UNKNOWN, classify_news, extract_news_events, ingest_news, preprocess_news, summarize_news,
)
from .tool_registry import ToolRegistry
from .utils import utc_now_iso

AGENT_NAME = "news_agent"
DESCRIPTION = (
    "Recent-news analysis: announcements, product launches, regulatory and legal events, and analyst actions, "
    "with the risks and catalysts they imply."
)


class NewsState(TypedDict, total=False):
    """State of the news chain. Each stage adds its output, so every intermediate result is kept."""

    symbol: str
    company: str
    raw: list[NewsArticle]
    feed_errors: dict[str, str]
    clean: list[NewsArticle]
    funnel: dict[str, int]
    classified: list[NewsArticle]
    events: list[ExtractedNewsEvent]
    summary: dict


def build_news_graph(tools: ToolRegistry, llm):
    """Compile the chain: ingest -> preprocess -> classify -> extract -> summarize."""

    def ingest(state: NewsState):
        raw, feed_errors = ingest_news(state["symbol"], state["company"], tools)
        return {"raw": raw, "feed_errors": feed_errors}

    def preprocess(state: NewsState):
        clean, funnel = preprocess_news(state["raw"], state["symbol"], state["company"])
        return {"clean": clean, "funnel": funnel}

    def classify(state: NewsState):
        return {"classified": classify_news(state["clean"], llm, state["company"], state["symbol"])}

    def extract(state: NewsState):
        events = extract_news_events(state["classified"], llm, state["company"], state["symbol"])
        return {"events": events, "funnel": {**state["funnel"], "material_events": len(events)}}

    def summarize(state: NewsState):
        return {"summary": summarize_news(state["events"], llm, state["company"], state["symbol"])}

    graph = StateGraph(NewsState)
    graph.add_sequence([ingest, preprocess, classify, extract, summarize])
    graph.add_edge(START, "ingest")
    graph.add_edge("summarize", END)
    return graph.compile()


def _tagged(events, field: str, limit: int = 6) -> list[str]:
    """Collect risks or catalysts across events, de-duplicated and tagged with their event id."""
    seen, items = set(), []
    for event in events:
        for text in getattr(event, field):
            if text.lower() not in seen:
                seen.add(text.lower())
                items.append(f"{text} [{event.id}]")
    return items[:limit]


def run_news_agent(task: SubagentInput, tools: ToolRegistry, llm) -> SubagentResult:
    """Run the news chain and return the standard subagent result."""
    result = SubagentResult(agent=AGENT_NAME, status="success")
    try:
        state = build_news_graph(tools, llm).invoke({"symbol": task.symbol, "company": task.profile.get("name") or task.symbol})
    except DataToolError as exc:
        return SubagentResult(
            agent=AGENT_NAME, status="failed", warnings=[f"News retrieval failed for {task.symbol}: {exc}"],
            limitations=["No news analysis was produced because no news feed could be retrieved."],
        )

    classified, events, funnel = state["classified"], state["events"], state["funnel"]
    if state["feed_errors"]:
        result.status = "partial"
        result.warnings.append(f"News feeds unavailable: {state['feed_errors']}")
        result.limitations.append(f"News coverage is incomplete: feed(s) {', '.join(state['feed_errors'])} failed.")
    unvalidated = sum(a.relevance == UNKNOWN or a.event_type == UNKNOWN for a in classified)
    if unvalidated:
        result.warnings.append(f"{unvalidated} article(s) received labels the model could not produce in a valid form.")
    fallbacks = sum(e.extraction_method == "headline_fallback" for e in events)
    if fallbacks:
        result.warnings.append(f"{fallbacks} event(s) fell back to the raw headline because extraction failed.")

    # One evidence item per event: the headline is observed, its labels are model judgments.
    evidence = EvidenceBuilder("NEWS", "news headline", retrieved_at=utc_now_iso())
    for event in events:
        evidence_id = evidence.add(
            event.event, value={"event_type": event.event_type, "impact": event.impact, "relevance": event.relevance},
            kind="observed", source=event.source or "unknown publisher", source_url=event.url, as_of=event.event_date,
        )
        if len(result.findings) < 6:  # events are already ordered by relevance
            result.findings.append(
                Finding(
                    statement=f"{event.event.rstrip('.')} ({event.source}, {event.event_date}; classified {event.event_type}, "
                    f"{event.impact} impact, {event.relevance} relevance)",
                    evidence_ids=[evidence_id],
                )
            )
    result.summary = state["summary"]["summary"]
    result.evidence = evidence.items
    result.risks = _tagged(events, "risks")
    result.catalysts = _tagged(events, "catalysts")
    result.limitations += [
        "News analysis is based on headlines only; full article text is not retrieved.",
        "Event type, relevance, impact, risks, and catalysts are judgments of a small local language model, not observed facts.",
        f"Only the {funnel['selected_for_llm']} most widely covered and most recent stories of the last "
        f"{settings.news_lookback_days} days were analyzed.",
    ]
    if not events:
        result.limitations.append("No material company-specific news was found in the lookback window.")

    result.visualization_data = {"funnel": funnel, "stages": state}  # stages: the output of every stage of the chain
    return result

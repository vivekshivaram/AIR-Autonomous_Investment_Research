"""Offline fixtures for the Engineer 3 tests: realistic specialist outputs and a scriptable fake LLM.

The fake specialists return the same shapes as the real market, earnings and news agents (evidence
IDs, findings, risks, catalysts, limitations, news stages), and they run through the real Planner,
Orchestrator and aggregation code, so the report pipeline sees exactly what it sees in production.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from src.models import SUBAGENT_EARNINGS, SUBAGENT_MARKET, SUBAGENT_NEWS
from src.orchestrator import Orchestrator
from src.planner import Planner
from src.stubs import register_stub_tools
from src.tool_registry import ToolRegistry, default_subagent_registry

COMPREHENSIVE = "Produce comprehensive investment research"
PRICE_ONLY = "Analyze one-year price performance"
EARNINGS_ONLY = "Analyze revenue and margin trends"


def today():
    return datetime.now(timezone.utc).date()


def _days_ago(days: int) -> str:
    return (today() - timedelta(days=days)).isoformat()


def market_result(symbol: str = "MSFT") -> dict[str, Any]:
    start, end = _days_ago(366), _days_ago(1)
    source, retrieved = "Yahoo Finance (yfinance)", f"{end}T21:00:00+00:00"
    evidence = [
        (f"{symbol} total return from {start} to {end}", 6.02, "%"),
        (f"{symbol} annualized volatility of daily returns", 32.81, "%"),
        (f"SPY total return from {start} to {end}", 18.6, "%"),
        (f"{symbol} excess return versus SPY", -12.58, "percentage points"),
        ("SPY annualized volatility of daily returns", 15.2, "%"),
        (f"{symbol} beta versus SPY (daily returns)", 1.05, None),
        (f"{symbol} maximum drawdown (peak 2026-02-03, trough 2026-04-10)", -15.3, "%"),
    ]
    return {
        "agent": SUBAGENT_MARKET,
        "status": "success",
        "summary": f"{symbol} returned +6.0% from {start} to {end} against +18.6% for SPY.",
        "findings": [
            {
                "statement": f"{symbol} returned +6.0% from {start} to {end} versus +18.6% for SPY, "
                "underperforming by 12.6 percentage points.",
                "evidence_ids": ["MKT-1", "MKT-3", "MKT-4"],
            },
            {"statement": "Annualized volatility was 32.8% versus 15.2% for SPY; beta was 1.05.", "evidence_ids": ["MKT-2", "MKT-5", "MKT-6"]},
            {
                "statement": "Maximum drawdown was -15.3%, from the peak on 2026-02-03 to the trough on 2026-04-10.",
                "evidence_ids": ["MKT-7"],
            },
        ],
        "evidence": [
            {"id": f"MKT-{i}", "claim": claim, "value": value, "unit": unit, "source": source, "as_of": end, "retrieved_at": retrieved}
            for i, (claim, value, unit) in enumerate(evidence, start=1)
        ],
        "risks": ["Volatility is 2.2x the benchmark's, so price swings are larger."],
        "catalysts": [],
        "limitations": [
            f"Based on adjusted daily closes from {start} to {end}; past performance does not predict future returns.",
            "SPY is a broad-market benchmark and may not reflect the company's sector.",
        ],
        "visualization_data": {},
    }


def earnings_result(fiscal_end: str = "2026-06-30") -> dict[str, Any]:
    source, retrieved = "Yahoo Finance (yfinance)", f"{_days_ago(1)}T21:00:00+00:00"
    evidence = [
        (f"Revenue, fiscal year ended {fiscal_end}", 331.8e9, "USD", "observed"),
        ("Revenue growth versus the prior fiscal year", 17.8, "%", "calculated"),
        (f"Net income, fiscal year ended {fiscal_end}", 120.5e9, "USD", "observed"),
        ("Net margin, latest versus prior year", [36.3, 35.8], "%", "calculated"),
    ]
    return {
        "agent": SUBAGENT_EARNINGS,
        "status": "success",
        "summary": f"For the fiscal year ended {fiscal_end}, revenue was USD 331.8 bn.",
        "findings": [
            {"statement": f"Revenue for the fiscal year ended {fiscal_end} was USD 331.8 bn, +17.8% year over year.", "evidence_ids": ["ERN-1", "ERN-2"]},
            {"statement": "Net income was USD 120.5 bn; net margin was 36.3% versus 35.8% a year earlier.", "evidence_ids": ["ERN-3", "ERN-4"]},
        ],
        "evidence": [
            {"id": f"ERN-{i}", "claim": claim, "value": value, "unit": unit, "kind": kind, "source": source, "as_of": fiscal_end, "retrieved_at": retrieved}
            for i, (claim, value, unit, kind) in enumerate(evidence, start=1)
        ],
        "risks": [],
        "catalysts": [],
        "limitations": [f"Based on 4 annual periods through {fiscal_end}; later quarterly results are not reflected."],
        "visualization_data": {},
    }


def news_result() -> dict[str, Any]:
    events = [
        ("E1", "Microsoft unveils the Maia 300 AI chip to reduce reliance on Nvidia.", "Reuters", _days_ago(3)),
        ("E2", "Regulators suspend Microsoft from a green card program.", "AP News", _days_ago(5)),
    ]
    return {
        "agent": SUBAGENT_NEWS,
        "status": "success",
        "summary": "Microsoft launched a custom AI chip [E1] while facing regulatory pressure [E2].",
        "findings": [
            {"statement": f"{text.rstrip('.')} ({publisher}, {day})", "evidence_ids": [f"NEWS-{i}"]}
            for i, (_, text, publisher, day) in enumerate(events, start=1)
        ],
        "evidence": [
            {
                "id": f"NEWS-{i}", "claim": text, "value": {"event_type": "product", "impact": "positive"}, "kind": "observed",
                "source": publisher, "source_url": f"https://example.com/{event_id}", "as_of": day, "retrieved_at": f"{_days_ago(0)}T08:00:00+00:00",
            }
            for i, (event_id, text, publisher, day) in enumerate(events, start=1)
        ],
        "risks": ["Regulatory scrutiny of hiring practices could raise compliance costs [E2]"],
        "catalysts": ["Custom AI chips could lower data-centre costs [E1]"],
        "limitations": ["News analysis is based on headlines only; full article text is not retrieved."],
        "visualization_data": {
            "funnel": {"ingested": 40, "after_preprocessing": 12, "material_events": 2},
            "stages": {"events": [{"id": event_id, "event": text} for event_id, text, _, _ in events]},
        },
    }


def specialist(result_factory: Callable[[], dict[str, Any]]):
    """Wrap a canned result as a subagent that also makes its planned tool calls (so the tool log is realistic)."""

    def run(state, tools, step):
        for tool in step.tools:
            tools.call(tool, symbol=state.symbol)
        return result_factory()

    return run


def failing(state, tools, step):
    raise RuntimeError("simulated news API outage")


def build_orchestrator(report_pipeline=None, memory_retriever=None, news_fn=None, earnings_fiscal_end: str = "2026-06-30"):
    """Planner + orchestrator wired to the fake specialists and stub data tools (rule-based planning, no LLM)."""
    tools, agents = ToolRegistry(), default_subagent_registry()
    register_stub_tools(tools)
    agents.attach(SUBAGENT_MARKET, specialist(market_result))
    agents.attach(SUBAGENT_EARNINGS, specialist(lambda: earnings_result(earnings_fiscal_end)))
    agents.attach(SUBAGENT_NEWS, news_fn or specialist(news_result))
    planner = Planner(agents, tools)
    return Orchestrator(planner, agents, tools, memory_retriever, report_pipeline), tools


class OfflineLLM:
    """An LLM that is never available. Pipeline tests pass it explicitly, because without an ``llm``
    argument the pipeline builds the client configured in .env and would make live API calls."""

    provider, model, available = "offline", "none", False


OFFLINE = OfflineLLM()


class FakeLLM:
    """Replies by schema name. A reply can be a dict, an exception to raise, or a function of (messages, schema)."""

    provider, model, available = "fake", "fake-model", True

    def __init__(self, replies: dict[str, Any] | None = None):
        self.replies = replies or {}
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def chat_json(self, messages, json_schema, *, schema_name="response", **kwargs):
        self.calls.append((schema_name, messages))
        reply = self.replies.get(schema_name)
        if reply is None:
            raise RuntimeError(f"no scripted reply for {schema_name}")
        if isinstance(reply, Exception):
            raise reply
        return reply(messages, json_schema) if callable(reply) else reply

    def chat(self, messages, **kwargs):
        self.calls.append(("chat", messages))
        raise RuntimeError("plain chat is not scripted")

    def count(self, schema_name: str) -> int:
        return sum(1 for name, _ in self.calls if name == schema_name)


def statement(text: str, kind: str = "fact", evidence_ids=(), risk_ids=(), catalyst_ids=(), limitation_ids=()) -> dict[str, Any]:
    """A statement as the LLM returns it."""
    return {
        "text": text, "kind": kind, "evidence_ids": list(evidence_ids), "risk_ids": list(risk_ids),
        "catalyst_ids": list(catalyst_ids), "limitation_ids": list(limitation_ids),
    }

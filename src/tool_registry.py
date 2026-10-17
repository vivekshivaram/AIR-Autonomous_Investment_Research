"""Approved-tool and subagent registries.

Tools are only callable through a ``ToolSession`` scoped to the tools the plan selected for
a step, so tool use is conditional and every call is logged (name, params, status,
duration, summary, error, timestamp).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from .models import (
    SUBAGENT_EARNINGS,
    SUBAGENT_MARKET,
    SUBAGENT_NEWS,
    AgentState,
    Status,
    ToolCallResult,
)

TOOL_COMPANY_PROFILE = "get_company_profile"
TOOL_PRICE_HISTORY = "get_price_history"
TOOL_BENCHMARK_HISTORY = "get_benchmark_history"
TOOL_FINANCIALS = "get_financial_statements"
TOOL_NEWS = "get_company_news"
TOOL_SEC_FILINGS = "get_sec_filings"
TOOL_MACRO = "get_macro_series"
TOOL_MEMORY_RETRIEVAL = "memory_retrieval"
TOOL_MEMORY_STORAGE = "memory_storage"

APPROVED_TOOLS = (
    TOOL_COMPANY_PROFILE,
    TOOL_PRICE_HISTORY,
    TOOL_BENCHMARK_HISTORY,
    TOOL_FINANCIALS,
    TOOL_NEWS,
    TOOL_SEC_FILINGS,
    TOOL_MACRO,
    TOOL_MEMORY_RETRIEVAL,
    TOOL_MEMORY_STORAGE,
)


def summarize_output(data: Any) -> str:
    """Short, log-safe description of a tool payload (never the payload itself)."""
    if data is None:
        return "no data"
    shape = getattr(data, "shape", None)
    if shape is not None:  # DataFrame / ndarray without importing pandas
        return f"{type(data).__name__} shape={tuple(shape)}"
    if isinstance(data, dict):
        return f"dict with {len(data)} keys"
    if isinstance(data, (list, tuple, set)):
        return f"{len(data)} records"
    return f"{type(data).__name__}: {str(data)[:80]}"


def _is_empty(data: Any) -> bool:
    if data is None:
        return True
    empty = getattr(data, "empty", None)
    if isinstance(empty, bool):
        return empty
    try:
        return len(data) == 0
    except TypeError:
        return False


@dataclass
class ToolSpec:
    """A registered tool: approved name, callable, description and whether it is critical."""

    name: str
    fn: Callable[..., Any]
    description: str
    critical: bool = False


class ToolRegistry:
    """Whitelist of callable tools. Registering an unapproved tool name is an error."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, name: str, fn: Callable[..., Any], description: str = "", critical: bool = False) -> None:
        """Register ``fn`` under an approved tool name; anything outside APPROVED_TOOLS raises ValueError."""
        if name not in APPROVED_TOOLS:
            raise ValueError(f"Tool {name!r} is not in the approved tool list")
        self._tools[name] = ToolSpec(name, fn, description, critical)

    def has(self, name: str) -> bool:
        """True when a tool with this name is registered."""
        return name in self._tools

    def names(self) -> list[str]:
        """Registered tool names, in registration order."""
        return list(self._tools)

    def describe(self) -> list[dict[str, Any]]:
        """Name, description and criticality of every registered tool."""
        return [{"name": t.name, "description": t.description, "critical": t.critical} for t in self._tools.values()]

    def session(self, caller: str, allowed: list[str], state: AgentState) -> "ToolSession":
        """A ToolSession that lets ``caller`` use only ``allowed`` tools and logs every call to ``state``."""
        return ToolSession(self, caller, allowed, state)

    def execute(self, name: str, caller: str, params: dict[str, Any]) -> ToolCallResult:
        """Run a tool and return a ToolCallResult; exceptions become failed results."""
        start = time.perf_counter()
        spec = self._tools.get(name)
        if spec is None:
            return ToolCallResult(name, params, Status.FAILED, 0.0, error="tool not registered", caller=caller)
        try:
            data = spec.fn(**params)
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            return ToolCallResult(
                name, params, Status.FAILED, round(elapsed, 2), error=f"{type(exc).__name__}: {exc}", caller=caller
            )
        elapsed = (time.perf_counter() - start) * 1000
        return ToolCallResult(
            name, params, Status.SUCCESS, round(elapsed, 2), output_summary=summarize_output(data), caller=caller, data=data
        )


class ToolSession:
    """Tool access for one step: enforces the plan's tool selection and logs every call to state."""

    def __init__(self, registry: ToolRegistry, caller: str, allowed: list[str], state: AgentState):
        self._registry = registry
        self.caller = caller
        self.allowed = set(allowed)
        self._state = state

    def call(self, tool_name: str, **params: Any) -> ToolCallResult:
        """Run a tool if the plan allowed it. Every call, allowed or refused, is logged; failures and empty results become warnings."""
        if tool_name not in self.allowed:
            result = ToolCallResult(
                tool_name, params, Status.FAILED, 0.0, error="tool not selected by the plan for this step", caller=self.caller
            )
        else:
            result = self._registry.execute(tool_name, self.caller, params)
        self._state.tool_call_log.append(result.to_log())
        if not result.ok:
            self._state.warn(self.caller, f"Tool {tool_name} failed: {result.error}", tool=tool_name)
        elif _is_empty(result.data):
            self._state.warn(self.caller, f"Tool {tool_name} returned no data", tool=tool_name)
        return result


# Signature every subagent implements: fn(state, tools: ToolSession, step) -> SubagentResult | dict
SubagentFn = Callable[[AgentState, ToolSession, Any], Any]


@dataclass
class SubagentSpec:
    """A specialist: description, tools it always needs, optional tools, routing keywords and its implementation."""

    name: str
    description: str
    required_tools: tuple[str, ...]
    optional_tools: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()  # routing hints for the rule-based planner
    fn: SubagentFn | None = None

    def allowed_tools(self) -> tuple[str, ...]:
        """Required plus optional tools; the planner may assign only these to the specialist."""
        return self.required_tools + self.optional_tools


class SubagentRegistry:
    """The specialist subagents the planner can route to."""

    def __init__(self) -> None:
        self._agents: dict[str, SubagentSpec] = {}

    def register(self, spec: SubagentSpec) -> None:
        """Add or replace a subagent spec."""
        self._agents[spec.name] = spec

    def attach(self, name: str, fn: SubagentFn) -> None:
        """Bind the implementation (provided by Engineer 2) to an existing spec."""
        self._agents[name].fn = fn

    def get(self, name: str) -> SubagentSpec:
        """The spec registered under ``name`` (KeyError if there is none)."""
        return self._agents[name]

    def has(self, name: str) -> bool:
        """True when a subagent with this name is registered."""
        return name in self._agents

    def names(self) -> list[str]:
        """Registered subagent names, in registration order."""
        return list(self._agents)

    def specs(self) -> list[SubagentSpec]:
        """Every registered subagent spec."""
        return list(self._agents.values())


def default_subagent_registry() -> SubagentRegistry:
    """Market, Earnings and News specs with their tool allowances and routing keywords."""
    registry = SubagentRegistry()
    registry.register(
        SubagentSpec(
            name=SUBAGENT_MARKET,
            description="Stock and benchmark price performance, returns, rolling volatility, trends; optional macro context.",
            required_tools=(TOOL_PRICE_HISTORY, TOOL_BENCHMARK_HISTORY),
            optional_tools=(TOOL_MACRO,),
            keywords=(
                "price", "stock performance", "share price", "share-price", "return", "volatility", "benchmark",
                "price trend", "rally", "drawdown", "momentum", "outperform", "underperform", "chart", "movement",
                "technical", "s&p", "stock move", "52-week",
            ),
        )
    )
    registry.register(
        SubagentSpec(
            name=SUBAGENT_EARNINGS,
            description="Income statement, balance sheet and cash-flow analysis: revenue, profit, margins, financial risks.",
            required_tools=(TOOL_FINANCIALS,),
            optional_tools=(TOOL_SEC_FILINGS,),
            keywords=(
                "earnings", "revenue", "profit", "margin", "cash flow", "balance sheet", "income statement",
                "financial", "eps", "net income", "debt", "quarterly", "annual", "fundamental", "10-k", "10-q",
                "sales", "balance-sheet", "cash-flow",
            ),
        )
    )
    registry.register(
        SubagentSpec(
            name=SUBAGENT_NEWS,
            description="Recent company news via the five-stage prompt chain: events, catalysts, risks, source-backed summary.",
            required_tools=(TOOL_NEWS,),
            keywords=(
                "news", "announcement", "headline", "event", "launch", "lawsuit", "acquisition",
                "catalyst", "sentiment", "product", "press", "development", "regulat", "partnership", "leadership",
            ),
        )
    )
    return registry

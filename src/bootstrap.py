"""Wires Engineer 2/3 modules into the registries, falling back to stubs for anything not yet merged.

Integration contract (module -> attribute):
  data_tools         one function per tool name, e.g. get_price_history(symbol, ...)
  market_agent       run_market_agent(state, tools, step)    -> SubagentResult | dict
  earnings_agent     run_earnings_agent(state, tools, step)
  news_agent         run_news_agent(state, tools, step)
  memory             retrieve_lessons(symbol, question) -> list[MemoryItem | dict]; store_lesson(...)
  report_generator   make_report_pipeline(llm, settings, tools) -> (state) -> AgentState, or
                     run_report_pipeline(state) -> AgentState   (report, evaluator-optimizer, reflection)
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable

from .config import Settings
from .llm_cache import CachedLLM, with_cache
from .llm_client import create_llm_client
from .models import SUBAGENT_EARNINGS, SUBAGENT_MARKET, SUBAGENT_NEWS
from .orchestrator import Orchestrator
from .planner import Planner
from .specialist_adapters import adapt_specialist
from .stubs import register_stub_subagents, register_stub_tools
from .tool_registry import (
    APPROVED_TOOLS,
    TOOL_MEMORY_RETRIEVAL,
    TOOL_MEMORY_STORAGE,
    SubagentRegistry,
    ToolRegistry,
    default_subagent_registry,
)

_AGENT_ENTRYPOINTS = {
    SUBAGENT_MARKET: ("market_agent", "run_market_agent"),
    SUBAGENT_EARNINGS: ("earnings_agent", "run_earnings_agent"),
    SUBAGENT_NEWS: ("news_agent", "run_news_agent"),
}


@dataclass
class Runtime:
    """Everything one notebook session needs: the planner, orchestrator, registries and LLM client."""

    planner: Planner
    orchestrator: Orchestrator
    tools: ToolRegistry
    subagents: SubagentRegistry
    llm: Any
    # component -> "real" | "stub" | "missing"; shown in the notebook so nothing is silently faked
    integration: dict[str, str] = field(default_factory=dict)


def _load(module: str) -> ModuleType | None:
    """Import a sibling module; absent modules are expected, broken ones are not hidden."""
    try:
        return importlib.import_module(f"{__package__}.{module}")
    except ModuleNotFoundError as exc:
        if exc.name == f"{__package__}.{module}":
            return None
        raise


def build_runtime(
    settings: Settings | None = None,
    llm: Any = None,
    allow_stubs: bool = True,
) -> Runtime:
    """Build the planner and orchestrator with every module that is present.

    Real modules replace stubs automatically; ``integration`` records which components are real, stubbed
    or missing so nothing is silently faked. Without an explicit ``llm`` the configured API client is
    created and wrapped in the on-disk response cache (``LLM_CACHE_HOURS``, 0 disables it).
    """
    settings = settings or Settings()
    llm = llm or with_cache(create_llm_client())
    tools, subagents = ToolRegistry(), default_subagent_registry()
    integration: dict[str, str] = {}

    data_tools = _load("data_tools")
    if data_tools is not None:
        for name in APPROVED_TOOLS:
            fn = getattr(data_tools, name, None)
            if callable(fn):
                tools.register(name, fn, (fn.__doc__ or name).strip().splitlines()[0])

    memory = _load("memory")
    retriever: Callable[..., Any] | None = None
    if memory is not None:
        retriever = getattr(memory, "retrieve_lessons", None)
        if retriever:
            tools.register(TOOL_MEMORY_RETRIEVAL, retriever, "Retrieve lessons from earlier runs")
        if callable(getattr(memory, "store_lesson", None)):
            tools.register(TOOL_MEMORY_STORAGE, memory.store_lesson, "Store a lesson for later runs")
    integration["memory"] = "real" if retriever else "missing"

    for agent, (module, attr) in _AGENT_ENTRYPOINTS.items():
        mod = _load(module)
        fn = getattr(mod, attr, None) if mod else None
        if callable(fn):
            subagents.attach(agent, adapt_specialist(agent, fn, llm, settings))

    if allow_stubs:
        stub_tools = register_stub_tools(tools)
        stub_agents = set(register_stub_subagents(subagents))
    else:
        stub_tools, stub_agents = [], set()
    for agent in subagents.names():
        integration[agent] = "stub" if agent in stub_agents else ("real" if subagents.get(agent).fn else "missing")
    integration["data_tools"] = (
        "missing" if not (data_tools or stub_tools) else "real" if not stub_tools else "partial" if data_tools else "stub"
    )
    integration["llm"] = getattr(llm, "provider", "unknown")
    integration["llm_cache"] = "on" if isinstance(llm, CachedLLM) else "off"

    report_module = _load("report_generator")
    pipeline: Callable[..., Any] | None = None
    if report_module is not None:
        factory = getattr(report_module, "make_report_pipeline", None)
        # The factory binds this runtime's LLM, settings and tool registry, so memory writes are logged tool calls.
        pipeline = factory(llm=llm, settings=settings, tools=tools) if callable(factory) else getattr(report_module, "run_report_pipeline", None)
    integration["report_pipeline"] = "real" if pipeline else "missing"

    planner = Planner(subagents, tools, llm=llm, settings=settings)
    orchestrator = Orchestrator(planner, subagents, tools, memory_retriever=retriever, report_pipeline=pipeline)
    return Runtime(planner, orchestrator, tools, subagents, llm, integration)

"""Placeholder tools and subagents so Engineer 1's flow runs before Engineers 2 and 3 merge.

Stubs never emit market data or findings: they only exercise the tool-call and logging
path and say so in their summary and limitations.
"""

from __future__ import annotations

from typing import Any

from .models import SUBAGENT_EARNINGS, SUBAGENT_MARKET, SUBAGENT_NEWS, AgentState, ResearchStep, Status
from .tool_registry import (
    APPROVED_TOOLS,
    TOOL_MEMORY_RETRIEVAL,
    TOOL_MEMORY_STORAGE,
    SubagentRegistry,
    ToolRegistry,
    ToolSession,
)

_STUB_TOOLS = [t for t in APPROVED_TOOLS if t not in (TOOL_MEMORY_RETRIEVAL, TOOL_MEMORY_STORAGE)]


def _stub_tool(name: str):
    def tool(**params: Any) -> dict[str, Any]:
        return {"stub": True, "tool": name, "params": params}

    return tool


def register_stub_tools(tools: ToolRegistry, only_missing: bool = True) -> list[str]:
    registered = []
    for name in _STUB_TOOLS:
        if only_missing and tools.has(name):
            continue
        tools.register(name, _stub_tool(name), f"[stub] {name}")
        registered.append(name)
    return registered


def make_stub_subagent(agent: str):
    def run(state: AgentState, tools: ToolSession, step: ResearchStep) -> dict[str, Any]:
        calls = [tools.call(tool, symbol=state.symbol) for tool in step.tools]
        ok = sum(c.ok for c in calls)
        return {
            "agent": agent,
            "status": Status.SUCCESS if ok == len(calls) else Status.PARTIAL,
            "summary": f"[STUB] {agent} executed {ok}/{len(calls)} planned tool calls; no analysis performed.",
            "findings": [],
            "evidence": [],
            "risks": [],
            "limitations": ["Stub implementation: no real data was analysed."],
            "visualization_data": {},
        }

    return run


def register_stub_subagents(subagents: SubagentRegistry, only_missing: bool = True) -> list[str]:
    attached = []
    for name in (SUBAGENT_MARKET, SUBAGENT_EARNINGS, SUBAGENT_NEWS):
        if subagents.has(name) and not (only_missing and subagents.get(name).fn is not None):
            subagents.attach(name, make_stub_subagent(name))
            attached.append(name)
    return attached

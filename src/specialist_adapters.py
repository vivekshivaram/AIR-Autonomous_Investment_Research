"""Adapt Engineer 2 specialist call shapes to the orchestrator's shared interface."""

from __future__ import annotations

from typing import Any, Callable

from .config import Settings
from .models import SUBAGENT_EARNINGS, SUBAGENT_MARKET, SUBAGENT_NEWS, AgentState, ResearchStep
from .specialist_models import SubagentInput
from .tool_registry import TOOL_MACRO, ToolSession


def adapt_specialist(
    name: str,
    implementation: Callable[..., Any],
    llm: Any,
    settings: Settings,
) -> Callable[[AgentState, ToolSession, ResearchStep], dict[str, Any]]:
    """Keep specialist-specific inputs private while retaining scoped tool execution."""

    def run(state: AgentState, tools: ToolSession, step: ResearchStep) -> dict[str, Any]:
        task = SubagentInput(
            symbol=state.symbol,
            profile=dict(state.company_profile or {}),
            include_macro=TOOL_MACRO in step.tools,
        )
        if name == SUBAGENT_NEWS:
            result = implementation(task, tools, llm, settings)
        elif name == SUBAGENT_MARKET:
            result = implementation(task, tools, settings)
        elif name == SUBAGENT_EARNINGS:
            result = implementation(task, tools)
        else:
            raise ValueError(f"No specialist adapter is defined for {name!r}")

        model_dump = getattr(result, "model_dump", None)
        return model_dump() if callable(model_dump) else result

    return run

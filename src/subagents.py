"""The specialist subagents, offered to the main agent's model as LangChain tools.

A tool's description is all the model knows about a subagent, and the model decides from these
descriptions whom to call. LangGraph's ToolNode runs the tools the model calls. Each tool runs
its subagent, stores the full result in the graph state, and sends a compact report back to
the model as the tool message.
"""

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from . import earnings_agent, market_agent, news_agent
from .models import SubagentInput, SubagentResult

MACRO_HELP = (
    "Set to true only if the question is about interest rates, inflation, commodity prices, "
    "or the economy, or asks for comprehensive research. Otherwise false."
)

# Section heading of each subagent in the research brief; also the order of the sections.
SECTION_TITLES = {
    market_agent.AGENT_NAME: "Market performance",
    earnings_agent.AGENT_NAME: "Financial analysis",
    news_agent.AGENT_NAME: "News and events",
}


def report_for_main_agent(result: SubagentResult) -> str:
    """The compact report the main agent reads; the full result stays in the graph state."""
    if result.status == "failed":
        return f"{result.agent} failed and produced nothing: {'; '.join(result.warnings)}"
    lines = [result.summary] + [f"- {finding.cited()}" for finding in result.findings[:8]]
    if result.risks:
        lines.append("Risks: " + "; ".join(result.risks[:3]))
    return "\n".join(lines)


def build_subagent_tools(tools, llm) -> list[BaseTool]:
    """One tool per subagent, bound to the data tools and the model the subagents run with."""

    def report(name: str, run, runtime: ToolRuntime, **options) -> Command | str:
        state = runtime.state
        if name in state.get("results", {}):  # the small model sometimes asks twice
            return f"{name} has already reported. Use its report above."
        try:
            result = run(SubagentInput(symbol=state["symbol"], profile=state["company_profile"], **options))
        except Exception as exc:  # an unexpected bug in one specialist must not end the run
            result = SubagentResult(
                agent=name, status="failed", warnings=[f"{name} crashed: {type(exc).__name__}: {exc}"],
                limitations=[f"No output from {name}."],
            )
        message = ToolMessage(report_for_main_agent(result), tool_call_id=runtime.tool_call_id)
        return Command(update={"results": {name: result}, "messages": [message]})

    @tool(market_agent.AGENT_NAME, description=market_agent.DESCRIPTION)
    def market(runtime: ToolRuntime, include_macro: Annotated[bool, MACRO_HELP] = False) -> Command | str:
        return report(market_agent.AGENT_NAME, lambda task: market_agent.run_market_agent(task, tools), runtime, include_macro=include_macro)

    @tool(earnings_agent.AGENT_NAME, description=earnings_agent.DESCRIPTION)
    def earnings(runtime: ToolRuntime) -> Command | str:
        return report(earnings_agent.AGENT_NAME, lambda task: earnings_agent.run_earnings_agent(task, tools), runtime)

    @tool(news_agent.AGENT_NAME, description=news_agent.DESCRIPTION)
    def news(runtime: ToolRuntime) -> Command | str:
        return report(news_agent.AGENT_NAME, lambda task: news_agent.run_news_agent(task, tools, llm), runtime)

    return [market, earnings, news]

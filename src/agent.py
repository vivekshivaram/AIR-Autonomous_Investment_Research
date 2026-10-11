"""The investment research agent: a research question in, an evidence-backed brief out.

    agent = InvestmentResearchAgent()
    state = agent.run("How risky is Tesla right now?")                     # ticker resolved automatically
    state = agent.run("Analyze revenue and margin trends.", symbol="MSFT")  # or given explicitly
    print(state["draft_report"])

From a terminal:  python -m src.agent "How risky is Tesla right now?"

Graph:  resolve -> main_agent <-> subagents (ToolNode: market_agent, earnings_agent, news_agent)
                   main_agent -> brief

The main agent is a tool-calling loop. The specialist subagents are offered to its model as
tools, and the model decides from their descriptions which ones the question needs. LangGraph's
ToolNode runs the subagents the model calls and returns their reports; when the model stops
calling subagents and answers, the brief is assembled.
"""

from __future__ import annotations

import argparse
import operator
import re
from typing import Annotated

from typing_extensions import NotRequired

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .llm import build_chat_model, json_chain, load_prompt
from .models import EvidenceItem, ResearchStep, SubagentCall, SubagentResult, ToolCallResult
from .subagents import SECTION_TITLES, build_subagent_tools
from .tool_registry import ToolRegistry, build_tool_registry

MAX_MODEL_CALLS = 4  # the main agent normally needs two: choose subagents, then answer


class ResearchState(MessagesState):
    """Shared state of one research run. `messages` (from MessagesState) is the main agent's conversation."""

    research_question: str
    symbol: NotRequired[str]
    company_profile: NotRequired[dict]
    results: Annotated[dict[str, SubagentResult], operator.or_]  # subagent name -> result, written by the subagent tools
    warnings: NotRequired[list[str]]
    subagent_calls: NotRequired[list[SubagentCall]]
    plan: NotRequired[list[ResearchStep]]
    evidence: NotRequired[list[EvidenceItem]]
    tool_log_start: NotRequired[int]
    tool_call_log: NotRequired[list[ToolCallResult]]
    answer: NotRequired[str]
    draft_report: NotRequired[str]


def build_research_graph(llm, tools: ToolRegistry | None = None):
    """Compile the research agent around one model and one data-tool registry."""
    tools = tools or build_tool_registry()

    def resolve(state: ResearchState):
        """Work out which company is meant and confirm it exists; nothing else runs without it."""
        log_start = len(tools.log)  # the registry's log is shared by every run on this agent

        def stop(reason: str):
            return {"warnings": [f"Research stopped: {reason}"], "tool_call_log": tools.log[log_start:]}

        symbol = state.get("symbol")
        if not symbol:  # no ticker given: the model names the company, a search maps the name to a ticker
            company = json_chain(llm, load_prompt("resolver")).invoke(state["research_question"]).get("company")
            if not company:
                return stop("no company could be identified in the question.")
            found = tools.call("resolve_symbol", query=str(company))
            if not found.ok:
                return stop(found.error)
            symbol = found.data["symbol"]
        profile = tools.call("get_company_profile", symbol=symbol)
        if not profile.ok:
            return stop(profile.error)
        symbol = profile.data["symbol"]
        request = HumanMessage(f"Company: {profile.data['name']} ({symbol})\nResearch question: {state['research_question']}")
        return {"tool_log_start": log_start, "symbol": symbol, "company_profile": profile.data, "messages": [request]}

    subagents = build_subagent_tools(tools, llm)
    llm_with_subagents = llm.bind_tools(subagents)

    def main_agent(state: ResearchState):
        """One model turn: call subagents, or answer once they have reported."""
        reply = llm_with_subagents.invoke([SystemMessage(load_prompt("main_agent")), *state["messages"]])
        return {"messages": [reply]}

    def route(state: ResearchState) -> str:
        """Conditional edge: run the subagents the model just called, or finish. A model that never answers is cut off."""
        model_calls = sum(isinstance(message, AIMessage) for message in state["messages"])
        return END if model_calls >= MAX_MODEL_CALLS else tools_condition(state)

    def brief(state: ResearchState):
        """Record which subagents ran, collect their evidence and warnings, and assemble the brief."""
        results = [state["results"][name] for name in SECTION_TITLES if name in state["results"]]
        status = {result.agent: result.status for result in results}
        subagent_calls = [SubagentCall(subagent=name, status=status.get(name, "skipped")) for name in SECTION_TITLES]
        plan = [ResearchStep(step_id=1, task="Identify the company and retrieve its profile", completion_criteria="Company identity confirmed", status="success")]
        for decision in subagent_calls:
            if decision.called:
                plan.append(
                    ResearchStep(
                        step_id=len(plan) + 1, task=SECTION_TITLES[decision.subagent], subagent=decision.subagent,
                        depends_on=[1], completion_criteria="Findings linked to evidence", status=decision.status,
                    )
                )
        produced = any(s != "failed" for s in status.values())
        plan.append(
            ResearchStep(
                step_id=len(plan) + 1, task="Answer the question and assemble the evidence-backed brief",
                depends_on=[step.step_id for step in plan[1:]], completion_criteria="Every finding linked to evidence",
                status="success" if produced else "failed",
            )
        )

        answer = next((m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage) and m.content), "")
        warnings = [warning for result in results for warning in result.warnings]
        if not results:
            warnings.append("The main agent called no subagent, so the brief contains no evidence.")
        reports = " ".join(str(m.content) for m in state["messages"] if isinstance(m, ToolMessage))
        unsupported = ungrounded_numbers(answer, reports)
        if unsupported:
            warnings.append(f"The main agent's answer contains figures not found in the subagent reports: {', '.join(unsupported)}.")

        evidence = [item for result in results for item in result.evidence]
        output = {"subagent_calls": subagent_calls, "plan": plan, "evidence": evidence, "answer": answer, "warnings": warnings}
        return {
            **output, "tool_call_log": tools.log[state["tool_log_start"]:],
            "draft_report": build_research_brief({**state, **output}, results),
        }

    graph = StateGraph(ResearchState)
    graph.add_node("resolve", resolve)
    graph.add_node("main_agent", main_agent)
    graph.add_node("brief", brief)
    graph.add_node("subagents", ToolNode(subagents))
    graph.add_edge("subagents", "main_agent")  # every subagent reports back to the main agent
    graph.add_edge(START, "resolve")
    graph.add_conditional_edges("resolve", lambda state: "main_agent" if state.get("company_profile") else END, ["main_agent", END])
    graph.add_conditional_edges("main_agent", route, {"tools": "subagents", END: "brief"})
    graph.add_edge("brief", END)
    return graph.compile()


def ungrounded_numbers(text: str, source: str) -> list[str]:
    """Numbers in `text` that do not occur in `source`; used to check the model-written answer."""
    known = set(re.findall(r"\d+(?:\.\d+)?", source.replace(",", "")))
    return [n for n in dict.fromkeys(re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))) if n not in known]


def build_research_brief(state: ResearchState, results: list[SubagentResult]) -> str:
    """Markdown brief: the main agent's answer, then the subagents' findings with their evidence ids."""
    profile = state["company_profile"]
    lines = [
        f"# Research brief: {profile.get('name', state['symbol'])} ({state['symbol']})",
        f"**Question:** {state['research_question']}",
        f"**Company:** {profile.get('sector')} / {profile.get('industry')}, {profile.get('country')}",
        "",
        "## Answer",
        state["answer"] or "_The main agent gave no answer._",
        "",
        "_Written by the main agent from the subagent reports. The sections below are copied from the subagents' results._",
        "",
        "## Subagents",
    ]
    lines += [f"- `{d.subagent}`: {'called' if d.called else 'not called'}" for d in state["subagent_calls"]]
    for result in results:
        lines += ["", f"## {SECTION_TITLES[result.agent]} ({result.status})"]
        if result.summary:
            lines += [result.summary, ""]
        lines += [f"- {finding.cited('`')}" for finding in result.findings]
    for title, field in (("Risks", "risks"), ("Catalysts", "catalysts")):
        items = [item for result in results for item in getattr(result, field)]
        if items:
            lines += ["", f"## {title}"] + [f"- {item}" for item in items]
    limitations = [limit for result in results for limit in result.limitations]
    lines += ["", "## Limitations and warnings"] + [f"- {text}" for text in limitations + state["warnings"]]
    lines += ["", "## Sources"]
    for source in sorted({item.source for item in state["evidence"]}):
        dates = [item.retrieved_at for item in state["evidence"] if item.source == source]
        lines.append(f"- {source} (retrieved {max(dates)})")
    lines += ["", "_Educational research output, not personalized investment advice._"]
    return "\n".join(lines)


class InvestmentResearchAgent:
    """One main agent with the specialist subagents attached; the compiled graph is the agent."""

    def __init__(self, llm=None, tools: ToolRegistry | None = None):
        self.graph = build_research_graph(llm or build_chat_model(), tools)

    def run(self, query: str, symbol: str | None = None) -> ResearchState:
        """Research one question. Pass `symbol` to name the stock, or let the agent find it in the question."""
        return self.graph.invoke({"research_question": query, **({"symbol": symbol} if symbol else {})})


def result_of(state: ResearchState, agent: str) -> SubagentResult | None:
    """The result of one subagent in a finished run, or None if the main agent did not call it."""
    return state.get("results", {}).get(agent)


def brief_of(state: ResearchState) -> str:
    """The Markdown brief of a run, or the reason the run stopped before producing one."""
    if state.get("draft_report"):
        return state["draft_report"]
    return "# No research produced\n" + "\n".join(f"- {w}" for w in state.get("warnings", []))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the investment research agent for one question.")
    parser.add_argument("query", help='Research question, for example "How risky is Tesla right now?"')
    parser.add_argument("--symbol", help="Ticker symbol; found from the question when omitted")
    arguments = parser.parse_args()
    print(brief_of(InvestmentResearchAgent().run(arguments.query, arguments.symbol)))


if __name__ == "__main__":
    main()

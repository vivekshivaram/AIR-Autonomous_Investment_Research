import numpy as np
from langchain_core.utils.function_calling import convert_to_openai_tool

from src import earnings_agent, market_agent, news_agent
from src.agent import InvestmentResearchAgent, brief_of, result_of
from src.subagents import SECTION_TITLES, build_subagent_tools
from tests.helpers import StubTools, price_frame, scripted_llm

PROFILE = {"symbol": "ACME", "name": "Acme Corporation", "sector": "Technology"}
MARKET = ("market_agent", {})
EARNINGS = ("earnings_agent", {})


def market_tools(**extra):
    return StubTools({
        "get_company_profile": PROFILE,
        "get_price_history": price_frame(np.linspace(100, 150, 260)),
        "get_benchmark_history": price_frame(np.linspace(100, 110, 260)),
        **extra,
    })


def run(llm, query="Analyze one-year price performance", symbol="ACME", tools=None):
    return InvestmentResearchAgent(llm, tools or market_tools()).run(query, symbol)


def test_every_subagent_is_offered_to_the_main_agent_as_a_tool_with_its_description():
    subagent_tools = build_subagent_tools(market_tools(), None)
    shown = {tool.name: convert_to_openai_tool(tool)["function"] for tool in subagent_tools}  # what the model is given
    assert list(shown) == list(SECTION_TITLES)
    assert {name: definition["description"] for name, definition in shown.items()} == {
        "market_agent": market_agent.DESCRIPTION, "earnings_agent": earnings_agent.DESCRIPTION, "news_agent": news_agent.DESCRIPTION,
    }
    # Only the arguments the model fills in are shown; the injected runtime is not.
    assert {name: list(definition["parameters"].get("properties", {})) for name, definition in shown.items()} == {
        "market_agent": ["include_macro"], "earnings_agent": [], "news_agent": [],
    }


def test_only_the_subagent_the_main_agent_calls_is_run_and_everything_is_logged():
    state = run(scripted_llm([MARKET], "ACME returned +50.0% over the period."))
    assert list(state["results"]) == ["market_agent"]
    assert result_of(state, "market_agent").status == "success" and result_of(state, "news_agent") is None
    assert {d.subagent: (d.called, d.status) for d in state["subagent_calls"]} == {
        "market_agent": (True, "success"), "earnings_agent": (False, "skipped"), "news_agent": (False, "skipped"),
    }
    assert [c.tool_name for c in state["tool_call_log"]] == ["get_company_profile", "get_price_history", "get_benchmark_history"]
    assert state["answer"] == "ACME returned +50.0% over the period." and state["evidence"]


def test_subagent_reports_go_back_to_the_main_agent_as_tool_messages():
    state = run(scripted_llm([MARKET], "Done."))
    kinds = [message.type for message in state["messages"]]
    assert kinds == ["human", "ai", "tool", "ai"]  # question, tool call, subagent report, answer
    report = state["messages"][2]
    assert report.name == "market_agent" and report.tool_call_id == state["messages"][1].tool_calls[0]["id"]
    assert "MKT-1" in report.content and "+50.0%" in report.content


def test_subagents_called_in_one_turn_all_run_and_arguments_are_passed_on():
    macro_call = ("market_agent", {"include_macro": True})
    made_up_argument = ("earnings_agent", {"depth": "full"})  # arguments a subagent does not have are ignored
    state = run(scripted_llm([macro_call, made_up_argument], "Done."), query="Analyze earnings, price and rates")
    assert set(state["results"]) == {"market_agent", "earnings_agent"}
    assert "get_macro_series" in [c.tool_name for c in state["tool_call_log"]]  # include_macro reached the market subagent


def test_one_failing_subagent_does_not_stop_the_run():
    state = run(scripted_llm([MARKET, EARNINGS], "Done."), query="Analyze earnings and the share-price reaction")
    assert result_of(state, "market_agent").status == "success" and result_of(state, "earnings_agent").status == "failed"
    assert any("Financial statements unavailable" in w for w in state["warnings"])
    assert "## Market performance (success)" in state["draft_report"] and "## Financial analysis (failed)" in state["draft_report"]


def test_main_agent_can_call_another_subagent_after_seeing_a_report():
    state = run(scripted_llm([MARKET], [EARNINGS], "Done."))
    assert list(state["results"]) == ["market_agent", "earnings_agent"]
    assert [message.type for message in state["messages"]] == ["human", "ai", "tool", "ai", "tool", "ai"]


def test_repeated_and_unknown_tool_calls_do_not_loop():
    repeated = run(scripted_llm([MARKET], [MARKET], [MARKET], [MARKET]))  # keeps asking for the same subagent, never answers
    assert len(repeated["results"]) == 1 and repeated["draft_report"]  # ran once; the cap on model turns ended the run
    assert "already reported" in repeated["messages"][4].content
    unknown = run(scripted_llm([("crystal_ball_agent", {})], "Done."))
    assert not unknown["results"] and unknown["messages"][2].status == "error"  # the model is told the tool does not exist
    assert any("called no subagent" in w for w in unknown["warnings"])


def test_ticker_is_resolved_from_a_free_text_question():
    tools = market_tools(resolve_symbol={"symbol": "ACME", "name": "Acme Corporation"})
    llm = scripted_llm('{"company": "Acme"}', [MARKET], "Done.")
    state = InvestmentResearchAgent(llm, tools).run("How has Acme stock performed this year?")
    assert state["symbol"] == "ACME" and state["company_profile"]["name"] == "Acme Corporation"
    assert [c.tool_name for c in state["tool_call_log"]][:2] == ["resolve_symbol", "get_company_profile"]
    assert "Acme Corporation (ACME)" in state["messages"][0].content


def test_run_stops_when_no_company_can_be_identified_or_found():
    no_company = InvestmentResearchAgent(scripted_llm('{"company": null}'), market_tools()).run("What is inflation?")
    not_found = InvestmentResearchAgent(scripted_llm('{"company": "Nonexistent"}'), market_tools()).run("Analyze Nonexistent")
    unknown_symbol = InvestmentResearchAgent(scripted_llm(), StubTools()).run("Analyze price performance", "ZZZZ")
    for state in (no_company, not_found, unknown_symbol):
        assert "results" not in state or not state["results"]
        assert any("Research stopped" in w for w in state["warnings"])
        assert brief_of(state).startswith("# No research produced")


def test_tool_log_of_a_run_excludes_earlier_runs_on_the_same_agent():
    agent = InvestmentResearchAgent(scripted_llm([MARKET], "Done.", [MARKET], "Done."), market_tools())
    first, second = agent.run("Analyze price performance", "ACME"), agent.run("Analyze price performance", "ACME")
    assert len(first["tool_call_log"]) == len(second["tool_call_log"]) == 3


def test_brief_has_the_answer_and_the_subagent_findings():
    state = run(scripted_llm([MARKET], "ACME returned +50.0% over the period."))
    brief = brief_of(state)
    assert brief.startswith("# Research brief: Acme Corporation (ACME)")
    assert "## Answer\nACME returned +50.0%" in brief and "MKT-1" in brief
    assert "- `market_agent`: called" in brief and "- `news_agent`: not called" in brief

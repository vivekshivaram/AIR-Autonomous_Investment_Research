import pytest

from src.models import SUBAGENT_EARNINGS, SUBAGENT_MARKET, SUBAGENT_NEWS
from src.planner import Planner
from src.stubs import register_stub_tools
from src.tool_registry import TOOL_MACRO, TOOL_SEC_FILINGS, ToolRegistry, default_subagent_registry


@pytest.fixture
def planner():
    tools = ToolRegistry()
    register_stub_tools(tools)
    return Planner(default_subagent_registry(), tools)


def selected(result):
    return set(result.plan.selected_subagents)


@pytest.mark.parametrize(
    "question, expected",
    [
        ("Analyze one-year price performance", {SUBAGENT_MARKET}),
        ("Analyze revenue and margin trends", {SUBAGENT_EARNINGS}),
        ("Summarize recent product announcements", {SUBAGENT_NEWS}),
        ("Analyze earnings and the resulting share-price movement", {SUBAGENT_EARNINGS, SUBAGENT_MARKET}),
        ("Produce comprehensive investment research", {SUBAGENT_MARKET, SUBAGENT_EARNINGS, SUBAGENT_NEWS}),
    ],
)
def test_routing_is_conditional(planner, question, expected):
    assert selected(planner.create_plan("AAPL", question)) == expected


def test_every_subagent_has_decision_with_reason(planner):
    result = planner.create_plan("AAPL", "Analyze one-year price performance")
    by_agent = {r.subagent: r for r in result.routing}
    assert set(by_agent) == {SUBAGENT_MARKET, SUBAGENT_EARNINGS, SUBAGENT_NEWS}
    assert all(r.reason.strip() for r in result.routing)
    assert by_agent[SUBAGENT_MARKET].selected and not by_agent[SUBAGENT_NEWS].selected
    assert all(0 <= r.confidence <= 1 for r in result.routing)


def test_unfocused_question_defaults_to_broad_with_low_confidence(planner):
    result = planner.create_plan("AAPL", "Tell me about this company")
    assert len(selected(result)) == 3
    assert max(r.confidence for r in result.routing) <= 0.5


def test_sequential_question_orders_earnings_before_market(planner):
    plan = planner.create_plan("AAPL", "Analyze earnings and the resulting share-price movement").plan
    assert [s.subagent for s in plan.steps] == [SUBAGENT_EARNINGS, SUBAGENT_MARKET]
    assert plan.steps[1].depends_on == [plan.steps[0].step_id]


def test_macro_tool_only_when_relevant(planner):
    plain = planner.create_plan("AAPL", "Analyze price performance").plan
    macro = planner.create_plan("JPM", "Analyze price performance given interest rates").plan
    assert TOOL_MACRO not in plain.steps[0].tools and not plain.macro_context_requested
    assert TOOL_MACRO in macro.steps[0].tools and macro.macro_context_requested


def test_filings_tool_only_when_relevant(planner):
    plain = planner.create_plan("AAPL", "Analyze revenue").plan
    filing = planner.create_plan("AAPL", "Analyze revenue and recent 10-K filing").plan
    assert TOOL_SEC_FILINGS not in plain.steps[0].tools
    assert TOOL_SEC_FILINGS in filing.steps[0].tools


def test_routing_not_fixed_pipeline(planner):
    questions = ["price performance", "revenue and margins", "latest news announcements"]
    counts = {len(selected(planner.create_plan("AAPL", q))) for q in questions}
    assert counts == {1}

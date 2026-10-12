import pytest

from src.models import ALL_SUBAGENTS, SUBAGENT_EARNINGS, SUBAGENT_MARKET, SUBAGENT_NEWS, ResearchPlan, ResearchStep
from src.planner import PlanValidationError, Planner, validate_plan
from src.stubs import register_stub_tools
from src.tool_registry import ToolRegistry, default_subagent_registry


class FakeLLM:
    available = True

    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def chat_json(self, messages, json_schema, **kwargs):
        self.calls.append((messages, json_schema))
        if self.error:
            raise self.error
        return self.response


def make_planner(llm=None):
    tools = ToolRegistry()
    register_stub_tools(tools)
    return Planner(default_subagent_registry(), tools, llm=llm)


def llm_plan(**overrides):
    plan = {
        "intent": "Price check",
        "scope": "narrow",
        "macro_context_relevant": False,
        "rationale": "Only price data needed.",
        "steps": [
            {
                "step_id": "S1", "description": "Analyze price", "subagent": SUBAGENT_MARKET, "depends_on": [],
                "extra_tools": [], "completion_criteria": "returns computed", "reason": "price question", "confidence": 0.9,
            }
        ],
        "skipped": [{"subagent": SUBAGENT_NEWS, "reason": "no news needed"}],
        "lessons_applied": [],
    }
    plan.update(overrides)
    return plan


def test_rule_based_plan_is_valid_and_structured():
    result = make_planner().create_plan("aapl", "Analyze one-year price performance")
    plan = result.plan
    assert plan.symbol == "AAPL"
    assert plan.planner_mode == "rule_based"
    assert plan.steps and all(s.completion_criteria for s in plan.steps)
    assert any("provider credentials not configured" in i.message for i in result.issues)


def test_llm_plan_used_when_valid():
    llm = FakeLLM(llm_plan())
    result = make_planner(llm).create_plan("AAPL", "price performance")
    assert result.plan.planner_mode == "llm"
    assert result.plan.selected_subagents == [SUBAGENT_MARKET]
    # Tools come from the registry allowance, not from the model.
    assert result.plan.steps[0].tools == ["get_price_history", "get_benchmark_history"]
    messages, schema = llm.calls[0]
    assert schema["properties"]["steps"]["items"]["properties"]["subagent"]["enum"] == list(ALL_SUBAGENTS)


def test_llm_error_falls_back_to_rule_based():
    result = make_planner(FakeLLM(error=RuntimeError("boom"))).create_plan("AAPL", "price performance")
    assert result.plan.planner_mode == "rule_based"
    assert any("LLM planning failed" in i.message for i in result.issues)


def test_invalid_llm_plan_falls_back():
    bad = llm_plan(steps=[{**llm_plan()["steps"][0], "depends_on": ["S9"]}])
    result = make_planner(FakeLLM(bad)).create_plan("AAPL", "price performance")
    assert result.plan.planner_mode == "rule_based"


def test_llm_disallowed_extra_tool_is_dropped():
    steps = [{**llm_plan()["steps"][0], "extra_tools": ["get_sec_filings"]}]
    result = make_planner(FakeLLM(llm_plan(steps=steps))).create_plan("AAPL", "price")
    assert "get_sec_filings" not in result.plan.steps[0].tools


def test_input_validation():
    planner = make_planner()
    with pytest.raises(ValueError):
        planner.create_plan("not a symbol!", "q")
    with pytest.raises(ValueError):
        planner.create_plan("AAPL", "   ")
    with pytest.raises(ValueError):
        planner.create_plan("AAPL", "x" * 5000)


def test_validate_plan_detects_problems():
    agents = default_subagent_registry()
    steps = [
        ResearchStep("S1", "a", SUBAGENT_MARKET, depends_on=["S2"], tools=["get_price_history"], completion_criteria="c"),
        ResearchStep("S2", "b", SUBAGENT_NEWS, depends_on=["S1"], tools=["get_price_history"], completion_criteria=""),
        ResearchStep("S2", "c", "ghost_agent", completion_criteria="c"),
    ]
    problems = validate_plan(ResearchPlan("AAPL", "q", "i", steps), agents, max_steps=6)
    text = " ".join(problems)
    assert "duplicate" in text and "ghost_agent" in text and "not allowed" in text and "missing completion" in text

    cyc = [
        ResearchStep("S1", "a", SUBAGENT_MARKET, depends_on=["S2"], completion_criteria="c"),
        ResearchStep("S2", "b", SUBAGENT_NEWS, depends_on=["S1"], completion_criteria="c"),
    ]
    assert "dependency cycle detected" in validate_plan(ResearchPlan("AAPL", "q", "i", cyc), agents, 6)


def test_lesson_hint_changes_plan():
    memories = [
        {
            "memory_id": "M1",
            "lesson": "Price questions need news context",
            "score": 0.9,
            "hint": {"require_subagents": [SUBAGENT_NEWS], "extra_criteria": "state data limitations"},
        }
    ]
    plain = make_planner().create_plan("AAPL", "price performance").plan
    adapted = make_planner().create_plan("AAPL", "price performance", memories)
    assert SUBAGENT_NEWS not in plain.selected_subagents
    assert SUBAGENT_NEWS in adapted.plan.selected_subagents
    assert "M1" in adapted.plan.lessons_applied
    assert all("state data limitations" in s.completion_criteria for s in adapted.plan.steps)
    # The previously-skipped decision for news must be replaced by a selected one.
    news = [r for r in adapted.routing if r.subagent == SUBAGENT_NEWS]
    assert len(news) == 1 and news[0].selected


def test_lessons_are_sent_to_llm_as_guidance():
    llm = FakeLLM(llm_plan())
    make_planner(llm).create_plan("AAPL", "price", [{"memory_id": "M7", "lesson": "dedupe news"}])
    assert "M7" in llm.calls[0][0][1]["content"]


def test_plan_validation_error_exposes_problems():
    err = PlanValidationError(["a", "b"])
    assert err.problems == ["a", "b"]

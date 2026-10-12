from src.models import (
    SUBAGENT_EARNINGS,
    SUBAGENT_MARKET,
    SUBAGENT_NEWS,
    AgentState,
    ResearchPlan,
    ResearchStep,
    Status,
)
from src.orchestrator import Orchestrator, aggregate_results, topological_order
from src.planner import Planner
from src.stubs import make_stub_subagent, register_stub_subagents, register_stub_tools
from src.tool_registry import TOOL_NEWS, TOOL_PRICE_HISTORY, ToolRegistry, default_subagent_registry


def build(report_pipeline=None, memory_retriever=None):
    tools, agents = ToolRegistry(), default_subagent_registry()
    register_stub_tools(tools)
    register_stub_subagents(agents)
    planner = Planner(agents, tools)
    return Orchestrator(planner, agents, tools, memory_retriever, report_pipeline), agents, tools


def test_narrow_run_only_calls_selected_tools():
    orch, _, _ = build()
    state = orch.run("AAPL", "Analyze one-year price performance")
    called = {c["tool_name"] for c in state.tool_call_log}
    assert called == {"get_company_profile", "get_price_history", "get_benchmark_history"}
    assert state.market_result and state.news_result is None and state.earnings_result is None
    assert [e["status"] for e in state.execution_log] == [Status.SUCCESS]


def test_comprehensive_run_executes_all_and_logs():
    orch, _, _ = build()
    state = orch.run("AAPL", "Produce comprehensive investment research")
    assert state.market_result and state.earnings_result and state.news_result
    assert len(state.execution_log) == 3
    log = state.tool_call_log[0]
    assert {"tool_name", "parameters", "status", "duration_ms", "output_summary", "error", "retrieved_at"} <= set(log)
    assert "data" not in log
    assert all(r.execution_status in (Status.SUCCESS, Status.SKIPPED) for r in state.routing_log)


def test_partial_failure_does_not_stop_workflow():
    orch, agents, _ = build()

    def boom(state, tools, step):
        raise RuntimeError("news API down")

    agents.attach(SUBAGENT_NEWS, boom)
    state = orch.run("AAPL", "Produce comprehensive investment research")
    statuses = {e["subagent"]: e["status"] for e in state.execution_log}
    assert statuses[SUBAGENT_NEWS] == Status.FAILED
    assert statuses[SUBAGENT_MARKET] == Status.SUCCESS and statuses[SUBAGENT_EARNINGS] == Status.SUCCESS
    assert any(w.severity == "error" and w.source == SUBAGENT_NEWS for w in state.warnings)
    assert any("did not complete" in l["text"] for l in state.aggregated["limitations"])


def test_unregistered_implementation_is_skipped():
    tools, agents = ToolRegistry(), default_subagent_registry()
    register_stub_tools(tools)
    orch = Orchestrator(Planner(agents, tools), agents, tools)
    state = orch.run("AAPL", "recent news")
    assert state.execution_log[0]["status"] == Status.SKIPPED
    assert state.news_result is None


def test_invalid_subagent_output_is_failed_not_crash():
    orch, agents, _ = build()
    agents.attach(SUBAGENT_MARKET, lambda s, t, st: {"agent": SUBAGENT_MARKET})
    state = orch.run("AAPL", "price performance")
    assert state.execution_log[0]["status"] == Status.FAILED
    assert state.market_result is None


def test_tool_session_blocks_tools_not_in_plan():
    orch, agents, _ = build()

    def greedy(state, tools, step):
        tools.call(TOOL_NEWS, symbol=state.symbol)  # news tool was not planned for the market step
        return make_stub_subagent(SUBAGENT_MARKET)(state, tools, step)

    agents.attach(SUBAGENT_MARKET, greedy)
    state = orch.run("AAPL", "price performance")
    blocked = [c for c in state.tool_call_log if c["tool_name"] == TOOL_NEWS]
    assert blocked and blocked[0]["status"] == Status.FAILED and "not selected" in blocked[0]["error"]


def test_failing_tool_logged_and_warned():
    orch, _, tools = build()
    tools.register(TOOL_PRICE_HISTORY, lambda **kw: (_ for _ in ()).throw(TimeoutError("slow")))
    state = orch.run("AAPL", "price performance")
    failed = [c for c in state.tool_call_log if c["status"] == Status.FAILED]
    assert failed[0]["tool_name"] == TOOL_PRICE_HISTORY and "TimeoutError" in failed[0]["error"]
    assert state.market_result["status"] == Status.PARTIAL
    assert any("get_price_history failed" in w.message for w in state.warnings)


def test_blocking_step_skipped_when_dependency_failed():
    orch, agents, _ = build()
    agents.attach(SUBAGENT_EARNINGS, lambda s, t, st: (_ for _ in ()).throw(RuntimeError("x")))
    state = AgentState("AAPL", "q")
    state.plan = ResearchPlan(
        "AAPL", "q", "i",
        [
            ResearchStep("S1", "e", SUBAGENT_EARNINGS, tools=["get_financial_statements"], completion_criteria="c"),
            ResearchStep("S2", "m", SUBAGENT_MARKET, depends_on=["S1"], tools=["get_price_history"],
                         completion_criteria="c", blocking=True),
        ],
    )
    orch._execute_plan(state)
    assert [e["status"] for e in state.execution_log] == [Status.FAILED, Status.SKIPPED]


def test_topological_order_respects_dependencies():
    steps = [
        ResearchStep("S1", "a", SUBAGENT_MARKET, depends_on=["S2"], completion_criteria="c"),
        ResearchStep("S2", "b", SUBAGENT_EARNINGS, completion_criteria="c"),
        ResearchStep("S3", "c", SUBAGENT_NEWS, completion_criteria="c"),
    ]
    assert [s.step_id for s in topological_order(steps)] == ["S2", "S1", "S3"]


def test_aggregation_dedupes_and_tags_evidence():
    state = AgentState("AAPL", "q")
    base = {"status": "success", "summary": "s", "findings": ["f"], "risks": [], "limitations": ["l"]}
    state.market_result = {"agent": SUBAGENT_MARKET, **base, "evidence": [{"claim": "Up 10%", "source": "yfinance"}]}
    state.news_result = {
        "agent": SUBAGENT_NEWS, **base,
        "evidence": [{"claim": "up 10%", "source": "yfinance"}, "Launch announced", {"nope": 1}],
    }
    agg = aggregate_results(state)
    assert [e["evidence_id"] for e in agg["evidence"]] == ["EV-001", "EV-002"]
    assert agg["evidence"][0]["source_agent"] == SUBAGENT_MARKET
    assert any("malformed" in w.message for w in state.warnings)
    assert agg["findings"][0] == {"text": "f", "agent": SUBAGENT_MARKET}


def test_report_pipeline_receives_aggregated_state_and_failures_are_contained():
    seen = {}

    def pipeline(state):
        seen["agg"] = state.aggregated
        state.final_report = {"ok": True}
        return state

    orch, _, _ = build(report_pipeline=pipeline)
    state = orch.run("AAPL", "price performance")
    assert seen["agg"] is not None and state.final_report == {"ok": True}

    def bad(state):
        raise RuntimeError("report down")

    orch, _, _ = build(report_pipeline=bad)
    state = orch.run("AAPL", "price performance")
    assert state.aggregated is not None
    assert any("Report pipeline failed" in w.message for w in state.warnings)


def test_memory_retrieval_failure_is_non_fatal_and_memories_recorded():
    def broken(symbol, question):
        raise OSError("db locked")

    orch, _, _ = build(memory_retriever=broken)
    state = orch.run("AAPL", "price performance")
    assert state.market_result and any("Memory retrieval failed" in w.message for w in state.warnings)

    orch, _, _ = build(memory_retriever=lambda s, q: [{"memory_id": "M1", "lesson": "be careful"}])
    state = orch.run("AAPL", "price performance")
    assert state.memories_used[0]["memory_id"] == "M1"

"""End to end: orchestrator, report, evaluator-optimizer, reflection, memory, and a second run that learns."""

import json

import pytest

import src.memory as memory
from src.config import Settings
from src.models import SUBAGENT_NEWS
from src.report_generator import make_report_pipeline, run_report_pipeline
from src.tool_registry import TOOL_MEMORY_RETRIEVAL, TOOL_MEMORY_STORAGE
from tests.report_fakes import COMPREHENSIVE, OFFLINE, PRICE_ONLY, build_orchestrator


@pytest.fixture
def private_memory(tmp_path):
    return memory.use_database(tmp_path / "memory.db", reset=True)


def make_runtime(tmp_path, with_tools=True):
    settings = Settings(output_dir=tmp_path / "outputs")
    orchestrator, tools = build_orchestrator(memory_retriever=memory.retrieve_lessons)
    tools.register(TOOL_MEMORY_RETRIEVAL, memory.retrieve_lessons, "Retrieve lessons")
    tools.register(TOOL_MEMORY_STORAGE, memory.store_lesson, "Store a lesson")
    orchestrator.report_pipeline = make_report_pipeline(llm=OFFLINE, settings=settings, tools=tools if with_tools else None)
    return orchestrator, settings


def test_pipeline_fills_the_state_and_writes_outputs(tmp_path, private_memory):
    orchestrator, settings = make_runtime(tmp_path)
    state = orchestrator.run("MSFT", COMPREHENSIVE)

    assert state.draft_report and state.final_report and state.evaluation_history
    assert state.reflection["run_id"] == state.run_id
    assert private_memory.runs()[0]["run_id"] == state.run_id
    log = json.loads((settings.output_dir / "logs" / "latest_run.json").read_text())
    assert log["run_id"] == state.run_id and log["evaluations"] and "report" not in log["evaluations"][0]
    assert (settings.output_dir / "reports" / "final_report.md").read_text().startswith("# MSFT investment research report")


def test_second_run_retrieves_the_lesson_and_changes_its_plan(tmp_path, private_memory):
    orchestrator, _ = make_runtime(tmp_path)

    first = orchestrator.run("MSFT", PRICE_ONLY)
    assert first.selected_subagents == ["market_agent"]
    stored = [l for l in first.reflection["lessons"] if l["trigger"] == "price_drivers_missing"][0]
    assert stored["memory_id"] == "MEM-0001"

    second = orchestrator.run("MSFT", PRICE_ONLY)
    assert SUBAGENT_NEWS in second.selected_subagents
    assert "MEM-0001" in second.plan.lessons_applied
    assert any(r.subagent == SUBAGENT_NEWS and "Added by lesson MEM-0001" in r.reason for r in second.routing_log)
    assert second.news_result is not None
    assert private_memory.get_lesson("MEM-0001")["times_applied"] == 1

    # A different question on the same symbol is not rerouted by the price lesson.
    third = orchestrator.run("MSFT", "Analyze revenue and margin trends")
    assert third.selected_subagents == ["earnings_agent"]


def test_memory_writes_appear_in_the_tool_call_log(tmp_path, private_memory):
    orchestrator, _ = make_runtime(tmp_path)
    state = orchestrator.run("MSFT", PRICE_ONLY)

    storage_calls = [c for c in state.tool_call_log if c["tool_name"] == TOOL_MEMORY_STORAGE]
    assert storage_calls and all(c["caller"] == "reflection" and c["status"] == "success" for c in storage_calls)


def test_memory_never_enters_the_evidence(tmp_path, private_memory):
    orchestrator, _ = make_runtime(tmp_path)
    orchestrator.run("MSFT", PRICE_ONLY)
    second = orchestrator.run("MSFT", PRICE_ONLY)

    evidence_ids = {item["id"] for item in second.aggregated["evidence"]}
    assert not any(eid.startswith("MEM-") for eid in evidence_ids)
    assert not any(source["id"].startswith("MEM-") for source in second.final_report["sources"])
    assert second.memories_used and second.memories_used[0]["memory_id"] == "MEM-0001"


def test_pipeline_without_aggregation_warns_instead_of_failing(tmp_path):
    orchestrator, _ = build_orchestrator()
    state = orchestrator.plan("MSFT", PRICE_ONLY)
    run_report_pipeline(state, llm=OFFLINE, settings=Settings(output_dir=tmp_path), save_outputs=False)

    assert state.final_report is None
    assert any("did not run" in w.message for w in state.warnings)

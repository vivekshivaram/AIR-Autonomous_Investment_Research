import sqlite3

import pytest

import src.memory as memory
from src.memory import MemoryStore, question_topics


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "memory.db")


NEWS_HINT = {"require_subagents": ["news_agent"]}


def test_question_topics_reuse_the_planner_keywords():
    assert question_topics("Analyze one-year price performance") == ["price"]
    assert question_topics("Analyze revenue and margin trends") == ["financials"]
    assert "comprehensive" in question_topics("Produce comprehensive investment research")


def test_symbol_lessons_rank_above_general_ones(store):
    store.store_lesson("Quote figures exactly.", hint={"report_guidance": "Quote figures exactly."})
    store.store_lesson("Add news for MSFT price questions.", scope="symbol", symbol="msft", tags=["price"], hint=NEWS_HINT)
    store.store_lesson("Add news for AAPL price questions.", scope="symbol", symbol="AAPL", tags=["price"], hint=NEWS_HINT)

    lessons = store.retrieve_lessons("MSFT", "Analyze one-year price performance")

    assert [l["lesson"] for l in lessons] == ["Add news for MSFT price questions.", "Quote figures exactly."]
    assert lessons[0]["symbol"] == "MSFT" and lessons[0]["score"] > lessons[1]["score"]
    assert set(lessons[0]) >= {"memory_id", "lesson", "scope", "symbol", "score", "hint"}


def test_a_lesson_that_adds_a_specialist_needs_a_matching_topic(store):
    store.store_lesson("Add news for MSFT price questions.", scope="symbol", symbol="MSFT", tags=["price"], hint=NEWS_HINT)
    assert store.retrieve_lessons("MSFT", "Analyze revenue and margin trends") == []
    assert len(store.retrieve_lessons("MSFT", "Analyze one-year price performance")) == 1


def test_repeated_lessons_are_merged(store):
    first = store.store_lesson("Quote  figures exactly.", hint={"report_guidance": "a"})
    second = store.store_lesson("quote figures exactly.", hint={"report_guidance": "b"})

    assert first["memory_id"] == second["memory_id"] == "MEM-0001"
    assert second["times_seen"] == 2 and second["hint"] == {"report_guidance": "b"}
    assert len(store.lessons()) == 1


def test_lessons_persist_across_connections_and_store_instances(tmp_path):
    path = tmp_path / "memory.db"
    MemoryStore(path).store_lesson("Add news for MSFT price questions.", scope="symbol", symbol="MSFT", tags=["price"], hint=NEWS_HINT)

    reopened = MemoryStore(path)
    assert reopened.retrieve_lessons("MSFT", "Analyze one-year price performance")[0]["memory_id"] == "MEM-0001"
    with sqlite3.connect(path) as raw:
        assert raw.execute("SELECT times_retrieved FROM lessons").fetchone()[0] == 1


def test_run_records_mark_which_lessons_were_used(store):
    lesson = store.store_lesson("Add news.", scope="symbol", symbol="MSFT", tags=["price"], hint=NEWS_HINT)
    store.record_run(run_id="r1", symbol="msft", question="q", lessons_retrieved=[lesson["memory_id"]],
                     lessons_applied=[lesson["memory_id"]], initial_score=70.0, final_score=85.0, reflection={"lessons": []})

    run = store.runs()[0]
    assert run["symbol"] == "MSFT" and run["lessons_applied"] == ["MEM-0001"] and run["final_score"] == 85.0
    assert store.get_lesson("MEM-0001")["times_applied"] == 1
    assert [u["event"] for u in store.usage()] == ["retrieved", "applied"]


def test_per_symbol_memory_stays_compact(store):
    for i in range(memory.MAX_LESSONS_PER_SYMBOL + 3):
        store.store_lesson(f"Lesson {i}.", scope="symbol", symbol="MSFT")
    assert len(store.lessons(active_only=True)) == memory.MAX_LESSONS_PER_SYMBOL


def test_reset_and_module_level_database_switch(tmp_path):
    memory.use_database(tmp_path / "a.db", reset=True)
    memory.store_lesson("General lesson.")
    assert len(memory.get_store().lessons()) == 1

    memory.use_database(tmp_path / "a.db", reset=True)
    assert memory.get_store().lessons() == []
    assert memory.retrieve_lessons("MSFT", "Analyze one-year price performance") == []


def test_scope_falls_back_to_general_without_a_symbol(store):
    lesson = store.store_lesson("No symbol given.", scope="symbol")
    assert lesson["scope"] == "general" and lesson["symbol"] is None

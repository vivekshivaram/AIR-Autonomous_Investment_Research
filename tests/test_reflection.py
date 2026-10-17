from dataclasses import replace
from datetime import timedelta

from src.optimizer import optimize_report
from src.reflection import derive_lessons, reflect
from src.report_evidence import build_evidence_pack
from src.report_generator import generate_report
from src.report_models import ReportIssue
from tests.report_fakes import COMPREHENSIVE, PRICE_ONLY, build_orchestrator, failing


def run(question=COMPREHENSIVE, **kwargs):
    orchestrator, _ = build_orchestrator(**kwargs)
    state = orchestrator.run("MSFT", question)
    pack = build_evidence_pack(state)
    report, _ = generate_report(pack)
    final, history = optimize_report(report, pack)
    return state, pack, history, final


def by_trigger(lessons):
    return {lesson.trigger: lesson for lesson in lessons}


def test_price_only_run_learns_to_add_news_for_this_symbol():
    state, pack, history, final = run(PRICE_ONLY)
    lesson = by_trigger(reflect(state, pack, history, final).lessons)["price_drivers_missing"]

    assert lesson.scope == "symbol" and lesson.symbol == "MSFT" and lesson.tags == ["price"]
    assert lesson.hint == {"require_subagents": ["news_agent"]}


def test_comprehensive_run_does_not_learn_the_news_lesson():
    state, pack, history, final = run()
    assert "price_drivers_missing" not in by_trigger(reflect(state, pack, history, final).lessons)


def test_failed_specialist_becomes_a_general_lesson_without_rerouting():
    state, pack, history, final = run(news_fn=failing)
    reflection = reflect(state, pack, history, final)
    lesson = by_trigger(reflection.lessons)["specialist_failed"]

    assert lesson.scope == "general" and "require_subagents" not in lesson.hint
    assert lesson.hint["subagent"] == "news_agent"
    assert any("news_agent failed" in failure for failure in reflection.failures)


def test_first_draft_problems_become_writing_guidance():
    state, pack, history, _ = run()
    history[0].issues.append(ReportIssue("numeric_consistency", "market_performance", "high", "x", "y", code="unverified_number"))
    history[0].issues.append(ReportIssue("risk_coverage", "risks", "medium", "x", "y", code="missing_risk"))
    lessons = by_trigger(derive_lessons(state, pack, history))

    assert "report_guidance" in lessons["numeric_or_citation_issues"].hint
    assert "report_guidance" in lessons["coverage_omissions"].hint


def test_readability_problems_become_writing_guidance():
    state, pack, history, _ = run()
    history[0].issues.append(ReportIssue("clarity", "financial_analysis", "medium", "x", "y", code="raw_figure"))
    lesson = by_trigger(derive_lessons(state, pack, history))["readability_issues"]
    assert lesson.scope == "general" and "billions" in lesson.hint["report_guidance"]


def test_old_annual_statements_trigger_a_symbol_lesson():
    state, pack, history, _ = run()
    later = replace(pack, run_date=pack.run_date + timedelta(days=400))
    lesson = by_trigger(derive_lessons(state, later, history))["stale_statements"]

    assert lesson.scope == "symbol" and lesson.hint["subagent"] == "earnings_agent"


def test_reflection_records_steps_scores_and_skips():
    state, pack, history, final = run(PRICE_ONLY)
    reflection = reflect(state, pack, history, final)

    assert any(s.startswith("S1 market_agent finished") for s in reflection.successes)
    assert any(s.startswith("earnings_agent not selected") for s in reflection.skipped_steps)
    assert reflection.initial_score == history[0].overall_score
    assert reflection.final_score == max(h.overall_score for h in history)
    assert reflection.improvements and reflection.stop_reason

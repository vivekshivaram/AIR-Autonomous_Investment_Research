from dataclasses import replace
from datetime import timedelta

from src.config import Settings
from src.evaluator import CRITERIA, extract_numbers, evaluate_report
from src.report_evidence import build_evidence_pack, disclosure
from src.report_generator import generate_report
from src.report_models import DECISION_ACCEPT, DECISION_REFINE, ReportStatement
from tests.report_fakes import COMPREHENSIVE, PRICE_ONLY, FakeLLM, build_orchestrator


def draft(question=COMPREHENSIVE):
    orchestrator, _ = build_orchestrator()
    pack = build_evidence_pack(orchestrator.run("MSFT", question))
    report, _ = generate_report(pack)
    return report, pack


def codes(evaluation):
    return {issue.code for issue in evaluation.issues}


def test_weights_sum_to_one_hundred_and_overall_is_their_weighted_mean():
    report, pack = draft()
    evaluation = evaluate_report(report, pack)

    assert sum(weight for _, _, weight in CRITERIA) == 100
    expected = sum(c.weight * c.score for c in evaluation.criteria) / 100
    assert abs(evaluation.overall_score - expected) < 0.06


def test_complete_template_report_is_accepted():
    report, pack = draft()
    evaluation = evaluate_report(report, pack)

    assert evaluation.decision == DECISION_ACCEPT and evaluation.overall_score >= 80
    assert evaluation.criterion("evidence_grounding").score == 100
    assert evaluation.criterion("numeric_consistency").score == 100
    assert evaluation.judge_mode == "heuristic"


def test_uncited_fact_and_misquoted_number_are_flagged_on_their_statement():
    report, pack = draft()
    market = report.section("market_performance")
    market.statements = [
        ReportStatement("MSFT beat the market comfortably.", kind="fact"),
        ReportStatement("MSFT returned +9.9% over the year.", kind="fact", evidence_ids=["MKT-1"]),
    ]
    evaluation = evaluate_report(report, pack)

    by_code = {issue.code: issue for issue in evaluation.issues if issue.section == "market_performance"}
    assert by_code["uncited_fact"].statement_index == 0
    assert by_code["unverified_number"].statement_index == 1 and "+9.9%" in by_code["unverified_number"].message
    assert evaluation.criterion("evidence_grounding").score < 100
    assert evaluation.criterion("numeric_consistency").score < 100


def test_numbers_match_across_units_rounding_and_sign():
    report, pack = draft()
    report.section("financial_analysis").statements = [
        ReportStatement("Revenue was USD 331.8 billion, up 17.8%.", evidence_ids=["ERN-1", "ERN-2"]),
        ReportStatement("Net margin was 36.3% against 35.8%.", evidence_ids=["ERN-4"]),
    ]
    report.section("market_performance").statements = [
        ReportStatement("Excess return was -12.6 percentage points; beta was 1.05.", evidence_ids=["MKT-4", "MKT-6"]),
        ReportStatement("Excess return was +12.6 percentage points.", evidence_ids=["MKT-4"]),
    ]
    evaluation = evaluate_report(report, pack)

    unverified = [i for i in evaluation.issues if i.code == "unverified_number"]
    assert len(unverified) == 1 and unverified[0].statement_index == 1  # the sign is wrong


def test_extract_numbers_skips_dates_ids_windows_and_years():
    numbers = extract_numbers("In FY2025 the 50-day average rose 3.2% [MKT-5] on 2026-10-12; Gartner 2026; 10-K; R2")
    assert [n.raw for n in numbers] == ["3.2%"]


def test_missing_risk_catalyst_and_limitation_are_flagged_with_fixes():
    report, pack = draft()
    report.section("risks").statements = []
    report.section("catalysts").statements = [ReportStatement("Nothing to add.", kind="disclosure")]
    report.section("confidence_and_limitations").statements = []
    evaluation = evaluate_report(report, pack)

    assert {"missing_risk", "missing_catalyst", "missing_limitation", "empty_section"} <= codes(evaluation)
    assert evaluation.criterion("risk_coverage").score == 0
    risk_issue = next(i for i in evaluation.issues if i.code == "missing_risk")
    assert risk_issue.refs and risk_issue.refs[0] in risk_issue.corrective_action
    assert evaluation.decision == DECISION_REFINE


def test_price_analysis_without_news_must_disclose_missing_drivers():
    report, pack = draft(PRICE_ONLY)
    first = evaluate_report(report, pack)
    issue = next(i for i in first.issues if i.code == "price_drivers_missing")
    assert issue.severity == "medium" and issue.actionable

    report.section("market_performance").statements.append(
        disclosure("The drivers of the price move were not analysed, because news analysis was not part of this run.")
    )
    second = evaluate_report(report, pack)
    issue = next(i for i in second.issues if i.code == "price_drivers_missing")
    assert issue.severity == "low" and not issue.actionable
    assert second.criterion("completeness").details["drivers"] == 0.5


def test_news_linked_market_section_explains_drivers():
    report, pack = draft()
    assert "price_drivers_unlinked" in codes(evaluate_report(report, pack))
    report.section("market_performance").statements.append(
        ReportStatement("The custom chip launch may have supported sentiment.", kind="interpretation", evidence_ids=["NEWS-1"])
    )
    evaluation = evaluate_report(report, pack)
    assert "price_drivers_unlinked" not in codes(evaluation)
    assert evaluation.criterion("completeness").details["drivers"] == 1.0


def test_stale_evidence_must_state_its_date():
    report, pack = draft()
    stale_pack = replace(pack, run_date=pack.run_date + timedelta(days=60))
    evaluation = evaluate_report(report, stale_pack)

    stale = {i.refs[0] for i in evaluation.issues if i.code == "stale_evidence"}
    assert "MKT-1" not in stale  # the market findings already state their dates
    assert evaluation.criterion("source_recency").score < evaluate_report(report, pack).criterion("source_recency").score


def test_memory_ids_are_not_evidence():
    report, pack = draft()
    report.section("executive_summary").statements.append(
        ReportStatement("Earlier runs showed the stock is volatile.", evidence_ids=["MEM-0001"])
    )
    evaluation = evaluate_report(report, pack)

    assert "invalid_citation" in codes(evaluation)
    assert evaluation.criterion("evidence_grounding").score < 100


def test_advice_wording_is_flagged():
    report, pack = draft()
    report.section("executive_summary").statements.append(
        ReportStatement("Investors should buy the stock before the next launch.", kind="interpretation")
    )
    assert "overclaim" in codes(evaluate_report(report, pack))


def test_floor_blocks_acceptance_even_with_a_zero_threshold():
    report, pack = draft()
    for section in report.sections:
        for st in section.statements:
            st.evidence_ids = []
    evaluation = evaluate_report(report, pack, settings=Settings(quality_threshold=0.0))

    assert evaluation.criterion("evidence_grounding").score < 50
    assert evaluation.decision == DECISION_REFINE


def test_clarity_uses_the_llm_judge_and_falls_back_to_heuristics():
    report, pack = draft()
    judged = evaluate_report(
        report, pack,
        FakeLLM({"clarity_review": {"score": 3, "summary": "Wordy.", "problems": [{"section": "risks", "problem": "Jargon.", "fix": "Use plain words."}]}}),
    )
    assert judged.judge_mode == "llm" and judged.criterion("clarity").score == 50  # min(LLM 50, rules 100)
    assert any(i.code == "clarity_llm" and i.section == "risks" for i in judged.issues)

    fallback = evaluate_report(report, pack, FakeLLM({"clarity_review": RuntimeError("timeout")}))
    assert fallback.judge_mode == "heuristic" and "llm_error" in fallback.criterion("clarity").details


def test_a_cited_headline_that_mentions_a_price_target_is_not_advice():
    report, pack = draft()
    pack.evidence["NEWS-1"]["claim"] = "UBS resets Microsoft stock price target for investors"
    report.section("news_and_events").statements = [
        ReportStatement("UBS resets Microsoft stock price target for investors.", evidence_ids=["NEWS-1"]),
    ]
    assert "overclaim" not in codes(evaluate_report(report, pack))
    report.section("executive_summary").statements.append(
        ReportStatement("Our price target implies upside.", kind="interpretation")
    )
    assert "overclaim" in codes(evaluate_report(report, pack))


def test_spelled_out_dates_are_not_checked_as_figures():
    numbers = extract_numbers("From October 17, 2025 to 16 October 2026 (since Oct 2025) MSFT returned 3.9%.")
    assert [n.raw for n in numbers] == ["3.9%"]


def test_ratios_and_gaps_derived_from_cited_figures_are_accepted():
    report, pack = draft()
    report.section("market_performance").statements = [
        ReportStatement("Volatility was 32.8%, roughly 2.2 times SPY's 15.2%.", evidence_ids=["MKT-2", "MKT-5"]),
        ReportStatement("Volatility was 32.8%, roughly 3.1 times SPY's 15.2%.", evidence_ids=["MKT-2", "MKT-5"]),
    ]
    unverified = [i for i in evaluate_report(report, pack).issues if i.code == "unverified_number"]
    assert len(unverified) == 1 and unverified[0].statement_index == 1 and '"3.1"' in unverified[0].message


def test_a_negated_guarantee_is_a_disclaimer_not_advice():
    report, pack = draft()
    report.section("confidence_and_limitations").statements.append(
        ReportStatement("Past performance does not guarantee future returns.", kind="disclosure")
    )
    assert "overclaim" not in codes(evaluate_report(report, pack))


def test_raw_figures_and_double_negatives_lower_clarity_and_block_acceptance():
    report, pack = draft()
    report.section("financial_analysis").statements = [
        ReportStatement("Revenue was USD 331800000000.0 in the fiscal year ended 2026-06-30.", evidence_ids=["ERN-1"]),
        ReportStatement("Net income was USD 120500000000.0.", evidence_ids=["ERN-3"]),
        ReportStatement("Net margin was 36.3% against 35.8%.", evidence_ids=["ERN-4"]),
    ]
    report.section("market_performance").statements.append(
        ReportStatement("MSFT underperformed SPY's 18.6% return by -12.6 percentage points.", evidence_ids=["MKT-3", "MKT-4"])
    )
    evaluation = evaluate_report(report, pack)

    assert {"raw_figure", "double_negative"} <= codes(evaluation)
    assert evaluation.criterion("numeric_consistency").score == 100  # the figures are right, only unreadable
    assert evaluation.criterion("clarity").score < 80


def test_clarity_takes_the_lower_of_the_llm_judgement_and_the_rules():
    report, pack = draft()
    report.section("financial_analysis").statements[0] = ReportStatement(
        "Revenue was USD 331800000000.0.", evidence_ids=["ERN-1"]
    )
    judged = evaluate_report(report, pack, FakeLLM({"clarity_review": {"score": 5, "summary": "Clear.", "problems": []}}))

    clarity = judged.criterion("clarity")
    assert clarity.details["llm_score_1_to_5"] == 5 and clarity.score == clarity.details["rule_score"] < 100
    assert any(i.code == "raw_figure" for i in judged.issues)


def test_a_criterion_below_sixty_forces_refinement():
    report, pack = draft()
    judged = evaluate_report(report, pack, FakeLLM({"clarity_review": {"score": 3, "summary": "Wordy.", "problems": []}}))
    assert judged.overall_score >= 80 and judged.criterion("clarity").score == 50
    assert judged.decision == DECISION_REFINE


def test_clarity_prompt_depends_on_content_not_on_the_generation_time():
    report, pack = draft()
    llm = FakeLLM({"clarity_review": {"score": 4, "summary": "Clear.", "problems": []}})
    evaluate_report(report, pack, llm)
    report.created_at = "2030-01-01T00:00:00+00:00"
    evaluate_report(report, pack, llm)

    first, second = [messages for name, messages in llm.calls if name == "clarity_review"]
    assert first == second and "2030-01-01" not in second[-1]["content"]

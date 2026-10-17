import src.optimizer as optimizer
from src.config import Settings
from src.evaluator import evaluate_report
from src.optimizer import optimize_report, refine_report
from src.report_evidence import build_evidence_pack
from src.report_generator import generate_report
from src.report_models import (
    DECISION_ACCEPT,
    DECISION_REFINE,
    STOP_ACCEPTED,
    STOP_LOW_GAIN,
    STOP_MAX_CYCLES,
    EvaluationResult,
    ReportIssue,
    ReportStatement,
)
from tests.report_fakes import COMPREHENSIVE, FakeLLM, build_orchestrator, statement


def draft():
    orchestrator, _ = build_orchestrator()
    pack = build_evidence_pack(orchestrator.run("MSFT", COMPREHENSIVE))
    report, _ = generate_report(pack)
    return report, pack


def weaken(report):
    """A realistic weak first draft: risks, catalysts and most limitations left out, one fact uncited."""
    report.section("risks").statements = []
    report.section("catalysts").statements = []
    report.section("confidence_and_limitations").statements = report.section("confidence_and_limitations").statements[:1]
    report.section("market_performance").statements.append(ReportStatement("The stock lagged the market.", kind="fact"))
    return report


def scripted(scores, weak=("risks",)):
    """Replace the evaluator with fixed scores so the loop's stop rules can be tested in isolation."""
    results = iter(scores)

    def fake_evaluate(report, pack, llm=None, settings=None, cycle=0):
        score = next(results)
        issues = [ReportIssue("risk_coverage", key, "medium", "missing", "add it", code="missing_risk") for key in weak]
        decision = DECISION_ACCEPT if score >= 80 else DECISION_REFINE
        return EvaluationResult(cycle, report.version, score, [], issues, decision, 80.0, 50.0)

    return fake_evaluate


def test_accepted_draft_is_not_refined():
    report, pack = draft()
    final, history = optimize_report(report, pack)

    assert len(history) == 1 and history[0].stop_reason == STOP_ACCEPTED
    assert final.version == 0


def test_rule_based_refinement_fixes_the_flagged_gaps_and_raises_the_score():
    report, pack = draft()
    final, history = optimize_report(weaken(report), pack)

    first, last = history[0], history[-1]
    assert first.decision == DECISION_REFINE and len(history) >= 2
    assert last.overall_score > first.overall_score
    assert final.version >= 1 and final.revisions
    assert all(r["method"] == "rules" for r in final.revisions)
    assert {"R1", "R2"} <= {rid for st in final.section("risks").statements for rid in st.risk_ids}
    assert not any(i.code == "uncited_fact" for i in last.issues)


def test_loop_stops_after_the_maximum_number_of_cycles(monkeypatch):
    monkeypatch.setattr(optimizer, "evaluate_report", scripted([40, 50, 60, 70]))
    report, pack = draft()
    final, history = optimize_report(report, pack, settings=Settings(max_refinement_cycles=2))

    assert [h.overall_score for h in history] == [40, 50, 60]
    assert history[-1].stop_reason == STOP_MAX_CYCLES and final.version == 2


def test_loop_stops_when_a_cycle_gains_too_little(monkeypatch):
    monkeypatch.setattr(optimizer, "evaluate_report", scripted([60, 61]))
    report, pack = draft()
    _, history = optimize_report(report, pack)

    assert len(history) == 2 and history[-1].stop_reason == STOP_LOW_GAIN


def test_a_worse_refinement_is_never_kept(monkeypatch):
    monkeypatch.setattr(optimizer, "evaluate_report", scripted([60, 45]))
    report, pack = draft()
    final, history = optimize_report(report, pack)

    assert history[-1].overall_score < history[0].overall_score
    assert final.version == 0


def test_llm_revision_cannot_drop_verified_statements():
    report, pack = draft()
    risks = report.section("risks")
    kept = risks.statements[0]
    risks.statements.append(ReportStatement("The stock is risk-free.", kind="interpretation"))
    evaluation = evaluate_report(report, pack)
    llm = FakeLLM({"section_revision": {"statements": [statement("Regulation may raise costs.", kind="interpretation", risk_ids=["R2"])]}})

    refined = refine_report(report, evaluation, pack, llm)

    texts = [st.text for st in refined.section("risks").statements]
    assert kept.text in texts and "The stock is risk-free." not in texts
    assert "Regulation may raise costs." in texts
    assert refined.revisions[0]["method"] == "llm"


def test_failed_llm_revision_falls_back_to_rules():
    report, pack = draft()
    weaken(report)
    evaluation = evaluate_report(report, pack)
    refined = refine_report(report, evaluation, pack, FakeLLM({"section_revision": RuntimeError("503")}))

    assert refined.revisions and all(r["method"].startswith("rules") for r in refined.revisions)
    assert refined.section("risks").statements


def test_rule_based_fix_replaces_unreadable_statements_with_the_findings():
    report, pack = draft()
    report.section("financial_analysis").statements[0] = ReportStatement(
        "Revenue was USD 331800000000.0, +17.8% year over year.", evidence_ids=["ERN-1", "ERN-2"]
    )
    evaluation = evaluate_report(report, pack)
    refined = refine_report(report, evaluation, pack)

    texts = [st.text for st in refined.section("financial_analysis").statements]
    assert not any("331800000000.0" in text for text in texts)
    assert any("USD 331.8 bn" in text for text in texts)


def test_a_rephrased_or_merged_locked_statement_is_not_put_back():
    report, pack = draft()
    risks = report.section("risks")
    risks.statements.append(ReportStatement("The stock is risk-free.", kind="interpretation"))
    evaluation = evaluate_report(report, pack)
    merged = statement("Price swings and regulation are the main risks.", kind="interpretation", evidence_ids=["NEWS-2"], risk_ids=["R1", "R2"])
    refined = refine_report(report, evaluation, pack, FakeLLM({"section_revision": {"statements": [merged]}}))

    texts = [st.text for st in refined.section("risks").statements]
    assert texts == ["Price swings and regulation are the main risks."]  # both originals' IDs survive in the merge

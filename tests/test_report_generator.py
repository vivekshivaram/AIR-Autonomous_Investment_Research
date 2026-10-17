from src.report_evidence import build_evidence_pack
from src.report_generator import generate_report
from src.report_models import FACT, SECTION_ORDER, SECTION_OUT_OF_SCOPE, SECTION_UNAVAILABLE, ResearchReport
from tests.report_fakes import COMPREHENSIVE, PRICE_ONLY, FakeLLM, build_orchestrator, failing, statement


def run(question=COMPREHENSIVE, **kwargs):
    orchestrator, _ = build_orchestrator(**kwargs)
    state = orchestrator.run("MSFT", question)
    return state, build_evidence_pack(state)


def test_template_report_has_every_section_and_only_grounded_facts():
    _, pack = run()
    report, notes = generate_report(pack, llm=None)

    assert [s.key for s in report.sections] == list(SECTION_ORDER)
    assert report.generated_by == "template" and notes == []
    facts = [st for _, _, st in report.statements() if st.kind == FACT]
    assert facts and all(st.evidence_ids and set(st.evidence_ids) <= set(pack.evidence) for st in facts)
    assert {s["id"] for s in report.sources} == report.cited_ids()
    assert report.confidence["level"] == "high" and report.confidence["reasons"]


def test_skipped_specialists_are_explained_not_silently_missing():
    _, pack = run(PRICE_ONLY)
    report, _ = generate_report(pack)

    financial = report.section("financial_analysis")
    assert financial.status == SECTION_OUT_OF_SCOPE and not financial.statements
    assert financial.note.startswith("Not covered: the earnings analysis was not part of this run.")
    assert "planner" not in financial.note and "specialist" not in financial.note  # written for the reader
    assert report.section("catalysts").status == SECTION_OUT_OF_SCOPE
    assert "not analysed in this run" in "; ".join(report.confidence["reasons"])


def test_failed_specialist_is_unavailable_and_its_gap_is_disclosed():
    _, pack = run(news_fn=failing)
    report, _ = generate_report(pack)

    assert report.section("news_and_events").status == SECTION_UNAVAILABLE
    gap = next(l for l in pack.limitations if "did not complete" in l["text"])
    assert any(gap["id"] in st.limitation_ids for st in report.section("conflicting_or_missing_evidence").statements)
    assert report.confidence["level"] == "low"


def test_llm_draft_keeps_valid_citations_and_drops_invented_ones():
    _, pack = run()
    draft = {key: [] for key in ("executive_summary", "market_performance", "financial_analysis", "news_and_events", "macro_context", "risks", "catalysts", "conflicting_or_missing_evidence", "confidence_and_limitations")}
    draft["executive_summary"] = [statement("MSFT returned +6.0% over the year.", evidence_ids=["MKT-1", "MEM-0001", "MKT-99"])]
    draft["risks"] = [statement("Volatility is high.", kind="interpretation", risk_ids=["R1", "R42"])]
    llm = FakeLLM({"research_report_draft": draft})

    report, notes = generate_report(pack, llm)

    summary = report.section("executive_summary").statements[0]
    assert summary.evidence_ids == ["MKT-1"]
    assert report.section("risks").statements[0].risk_ids == ["R1"]
    assert report.generated_by == "llm" and "Removed 3 citation" in notes[0]


def test_llm_failure_falls_back_to_the_template():
    _, pack = run()
    report, notes = generate_report(pack, FakeLLM({"research_report_draft": RuntimeError("quota exceeded")}))

    assert report.generated_by == "template"
    assert report.section("market_performance").statements
    assert "template used instead" in notes[0]


def test_news_event_tags_become_evidence_ids():
    _, pack = run()
    risk = next(r for r in pack.risks if r["agent"] == "news_agent")
    catalyst = pack.catalysts[0]

    assert risk["evidence_ids"] == ["NEWS-2"] and "[E2]" not in risk["text"]
    assert catalyst["evidence_ids"] == ["NEWS-1"]
    assert "[NEWS-1]" in pack.news_summary


def test_company_profile_facts_are_citable_evidence():
    state, _ = run()
    state.company_profile = {"symbol": "MSFT", "name": "Microsoft Corporation", "exchange": "NMS", "currency": "USD",
                             "sector": "Technology", "industry": "Software", "country": "United States",
                             "market_cap": 3.8e12, "source": "Yahoo Finance (yfinance)", "retrieved_at": "2026-10-17T00:00:00+00:00"}
    state.aggregated["company_profile"] = state.company_profile
    pack = build_evidence_pack(state)
    report, _ = generate_report(pack)

    overview = report.section("company_overview").statements
    assert [st.evidence_ids for st in overview] == [["PRO-1"], ["PRO-2"], ["PRO-3"], ["PRO-4"]]
    assert "listed on the Nasdaq Global Select Market" in overview[0].text  # exchange code NMS spelled out
    assert "USD 3,800.0 bn" in overview[-1].text


def test_markdown_labels_interpretations_and_lists_sources():
    _, pack = run()
    report, _ = generate_report(pack)
    markdown = report.to_markdown()

    assert "*Interpretation:*" in markdown and "*Note:*" in markdown
    assert "## Sources" in markdown and "| MKT-1 |" in markdown
    assert ResearchReport.from_dict(report.to_dict()).to_markdown() == markdown

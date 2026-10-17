import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import src.chart_notes as notes
import src.memory as memory
from src.config import Settings
from src.evaluation_viz import plot_agent_workflow, plot_criterion_scores, plot_issue_counts, plot_quality_across_runs, plot_score_by_cycle
from src.report_generator import make_report_pipeline
from src.run_outputs import save_figure
from tests.report_fakes import OFFLINE, PRICE_ONLY, build_orchestrator
from tests.test_optimizer import weaken


@pytest.fixture
def refined_state(tmp_path):
    """A run whose first draft was weakened, so the history has a real refinement cycle."""
    memory.use_database(tmp_path / "memory.db", reset=True)
    settings = Settings(output_dir=tmp_path / "outputs")
    orchestrator, _ = build_orchestrator(memory_retriever=memory.retrieve_lessons)
    pipeline = make_report_pipeline(llm=OFFLINE, settings=settings, save_outputs=False)

    def weakened_pipeline(state):
        import src.report_generator as generator

        original = generator.generate_report

        def weak_generate(pack, llm=None, settings=None):
            report, notes_ = original(pack, llm, settings)
            return weaken(report), notes_

        generator.generate_report = weak_generate
        try:
            return pipeline(state)
        finally:
            generator.generate_report = original

    orchestrator.report_pipeline = weakened_pipeline
    return orchestrator.run("MSFT", "Produce comprehensive investment research")


def labelled(fig):
    ax = fig.axes[0]
    return bool(ax.get_title(loc="left")) and bool(ax.get_xlabel() or ax.get_ylabel()) and ax.get_legend() is not None


def test_evaluation_charts_are_labelled(refined_state):
    history = refined_state.evaluation_history
    assert len(history) >= 2
    for fig in (
        plot_criterion_scores(history[0], history[-1]),
        plot_score_by_cycle(history),
        plot_issue_counts(history[0], history[-1]),
        plot_quality_across_runs([{"label": "Run 1", "initial": 70.0, "final": 90.0, "lessons_applied": []},
                                  {"label": "Run 2", "initial": 85.0, "final": 95.0, "lessons_applied": ["MEM-0001"]}]),
    ):
        assert labelled(fig)


def test_workflow_diagram_and_caption(refined_state):
    fig = plot_agent_workflow(refined_state)
    assert "Agent workflow for MSFT" in fig.axes[0].get_title(loc="left")
    assert "refinement cycle" in notes.workflow(refined_state)


def test_captions_quote_the_data(refined_state):
    history = refined_state.evaluation_history
    assert "First draft scored" not in notes.score_by_cycle(history)  # plain sentence case, not a label
    assert f"{history[0]['overall_score']:.1f}" in notes.score_by_cycle(history)
    assert "weakest criterion" in notes.criterion_scores(history[0], history[-1])

    dates = pd.bdate_range("2025-10-01", periods=60)
    perf = pd.DataFrame({"MSFT": np.linspace(100, 106, 60), "SPY": np.linspace(100, 118.6, 60)}, index=dates)
    assert "trailed the benchmark by 12.6" in notes.performance(perf)
    assert "times as wide as SPY" in notes.daily_returns(perf + np.sin(np.arange(60))[:, None])
    vol = pd.DataFrame({"MSFT": np.linspace(20, 30, 60), "SPY": np.linspace(10, 15, 60)}, index=dates)
    assert "peaked at 30.0%" in notes.volatility(vol)
    trends = pd.DataFrame({"revenue": [200.0, 245.0, 281.7, 331.8], "net_income": [70.0, 72.4, 88.1, 120.5]},
                          index=["FY2023", "FY2024", "FY2025", "FY2026"])
    assert "USD 331.8 bn in FY2026" in notes.financial_trends(trends, "USD")
    funnel = {"ingested": 125, "valid_and_recent": 60, "after_exact_dedup": 40, "relevant": 20, "after_near_dedup": 14,
              "selected_for_llm": 12, "material_events": 11}
    assert "125 headlines" in notes.news_funnel(funnel) and "11 material events" in notes.news_funnel(funnel)


def test_save_figure_writes_png(tmp_path, refined_state):
    path = save_figure(plot_score_by_cycle(refined_state.evaluation_history), "cycles", Settings(output_dir=tmp_path))
    assert path.exists() and path.stat().st_size > 1000


def test_routing_caption_counts_selected_specialists():
    orchestrator, _ = build_orchestrator()
    cases = {q: orchestrator.plan("MSFT", q).routing_log for q in (PRICE_ONLY, "Produce comprehensive investment research")}
    assert "between 1 and 3 specialists" in notes.routing(cases)

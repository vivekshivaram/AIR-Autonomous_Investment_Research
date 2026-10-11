import numpy as np
import pandas as pd
import pytest

from src.earnings_agent import cagr, cash_conversion, debt_to_equity, growth_rates, margins, run_earnings_agent
from src.models import SubagentInput
from tests.helpers import StubTools


def financials(**overrides):
    index = pd.DatetimeIndex(["2023-12-31", "2024-12-31", "2025-12-31"], name="period_end")
    frame = pd.DataFrame(
        {
            "revenue": [100e9, 120e9, 150e9],
            "gross_profit": [60e9, 72e9, 84e9],
            "operating_income": [30e9, 36e9, 42e9],
            "net_income": [20e9, 24e9, 27e9],
            "total_debt": [50e9, 50e9, 60e9],
            "stockholders_equity": [100e9, 110e9, 120e9],
            "cash": [30e9, 35e9, 40e9],
            "operating_cash_flow": [28e9, 30e9, 36e9],
            "free_cash_flow": [20e9, 21e9, 26e9],
        },
        index=index,
    )
    for column, values in overrides.items():
        frame[column] = values
    frame.attrs = {"source": "test", "retrieved_at": "2026-01-01T00:00:00+00:00", "missing_items": [c for c in frame if frame[c].isna().all()]}
    return frame


def test_growth_rates():
    growth = growth_rates(pd.Series([100.0, 120.0, 150.0]))
    assert np.isnan(growth.iloc[0])
    assert growth.iloc[1:].tolist() == pytest.approx([0.20, 0.25])


def test_growth_is_undefined_from_a_non_positive_base():
    assert np.isnan(growth_rates(pd.Series([-10.0, 20.0])).iloc[1])


def test_margins():
    ratio = margins(financials())
    assert ratio["gross_margin"].iloc[-1] == pytest.approx(0.56)
    assert ratio["net_margin"].iloc[-1] == pytest.approx(0.18)
    assert ratio["fcf_margin"].iloc[0] == pytest.approx(0.20)


def test_cagr():
    assert cagr(100, 121, 2) == pytest.approx(0.10)
    assert cagr(0, 121, 2) is None
    assert cagr(100, 121, 0) is None


def test_ratios_handle_missing_and_non_positive_denominators():
    assert debt_to_equity(60e9, 120e9) == pytest.approx(0.5)
    assert debt_to_equity(60e9, -5e9) is None
    assert cash_conversion(36e9, 27e9) == pytest.approx(36 / 27)
    assert cash_conversion(float("nan"), 27e9) is None


def test_agent_findings_are_evidenced_and_flag_margin_compression():
    tools = StubTools({"get_financial_statements": financials()})
    result = run_earnings_agent(SubagentInput(symbol="TEST", profile={"currency": "USD"}), tools)
    assert result.status == "success"
    evidence_ids = {item.id for item in result.evidence}
    assert result.findings and all(f.evidence_ids and set(f.evidence_ids) <= evidence_ids for f in result.findings)
    assert "+25.0% year over year" in result.findings[0].statement
    # Net margin fell from 20% to 18%, more than the one-point tolerance.
    assert any("Net margin compressed" in risk for risk in result.risks)
    # The optional SEC tool was unavailable: reported as a limitation, not a failure.
    assert any("SEC filing" in limit for limit in result.limitations)


def test_missing_line_items_give_partial_result_with_warning_and_limitation():
    tools = StubTools({"get_financial_statements": financials(gross_profit=np.nan)})
    result = run_earnings_agent(SubagentInput(symbol="TEST"), tools)
    assert result.status == "partial"
    assert any("gross_profit" in w for w in result.warnings)
    assert any("gross_profit" in limit for limit in result.limitations)
    assert not any("Gross margin" in item.claim for item in result.evidence)


def test_missing_statements_fail_without_invented_numbers():
    result = run_earnings_agent(SubagentInput(symbol="TEST"), StubTools())
    assert result.status == "failed"
    assert result.warnings and result.limitations and not result.evidence

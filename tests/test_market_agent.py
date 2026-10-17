import numpy as np
import pandas as pd
import pytest

from src.market_agent import (
    annualized_volatility, beta, daily_returns, max_drawdown, rolling_volatility, run_market_agent,
    total_return, trailing_returns,
)
from src.models import SubagentInput
from tests.helpers import StubTools, price_frame


def test_daily_and_total_return():
    close = pd.Series([100.0, 110.0, 99.0])
    assert daily_returns(close).tolist() == pytest.approx([0.10, -0.10])
    assert total_return(close) == pytest.approx(-0.01)


def test_annualized_volatility_matches_definition():
    returns = pd.Series([0.01, -0.02, 0.03, 0.00])
    assert annualized_volatility(returns) == pytest.approx(np.std(returns, ddof=1) * np.sqrt(252))


def test_rolling_volatility_window_and_constant_returns():
    returns = pd.Series([0.01] * 10)
    rolling = rolling_volatility(returns, window=5)
    assert len(rolling) == 6  # first full window ends at the fifth observation
    assert rolling.abs().max() == pytest.approx(0.0, abs=1e-12)


def test_max_drawdown_finds_peak_and_trough():
    close = pd.Series([100.0, 120.0, 90.0, 95.0, 130.0], index=pd.bdate_range("2026-01-01", periods=5))
    drawdown, peak, trough = max_drawdown(close)
    assert drawdown == pytest.approx(-0.25)
    assert (peak, trough) == (close.index[1], close.index[2])


def test_beta_of_scaled_series():
    benchmark = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
    assert beta(benchmark * 2, benchmark) == pytest.approx(2.0)
    assert beta(benchmark, benchmark) == pytest.approx(1.0)


def test_trailing_returns_skips_windows_longer_than_data():
    close = pd.Series(np.linspace(100, 130, 31))
    result = trailing_returns(close, {"1M": 21, "6M": 126})
    assert set(result) == {"1M"}
    assert result["1M"] == pytest.approx(130 / close.iloc[-22] - 1)


def test_agent_output_is_reproducible_and_every_finding_has_evidence():
    data = {
        "get_price_history": price_frame(np.linspace(100, 150, 260)),
        "get_benchmark_history": price_frame(np.linspace(100, 110, 260)),
    }
    first = run_market_agent(SubagentInput(symbol="TEST"), StubTools(data))
    second = run_market_agent(SubagentInput(symbol="TEST"), StubTools(data))
    assert first.status == "success"
    # The chart payloads are DataFrames, which do not support ==; compare them separately.
    assert first.model_dump(exclude={"visualization_data"}) == second.model_dump(exclude={"visualization_data"})
    for key, frame in first.visualization_data.items():
        pd.testing.assert_frame_equal(frame, second.visualization_data[key])
    evidence_ids = {item.id for item in first.evidence}
    assert first.findings and all(f.evidence_ids and set(f.evidence_ids) <= evidence_ids for f in first.findings)
    assert first.evidence[0].value == pytest.approx(50.0)  # 100 -> 150
    assert list(first.visualization_data["performance"].columns) == ["TEST", "SPY"]


def test_missing_benchmark_gives_partial_result_with_warning_and_limitation():
    tools = StubTools({"get_price_history": price_frame(np.linspace(100, 150, 260))}, failing={"get_benchmark_history"})
    result = run_market_agent(SubagentInput(symbol="TEST"), tools)
    assert result.status == "partial"
    assert any("Benchmark" in w for w in result.warnings)
    assert any("benchmark" in limit.lower() for limit in result.limitations)
    assert not any("SPY" in item.claim for item in result.evidence)  # nothing invented for the benchmark


def test_missing_prices_fail_without_invented_numbers():
    result = run_market_agent(SubagentInput(symbol="TEST"), StubTools())
    assert result.status == "failed"
    assert result.warnings and result.limitations
    assert not result.findings and not result.evidence

import numpy as np
import pandas as pd
import pytest
from tenacity import wait_none

from src.data_tools import DataToolError, _retry, normalize_prices, normalize_statements, normalize_symbol, short_company_name
from src.utils import clean_text, extract_json


def raw_prices(rows=40):
    index = pd.date_range("2026-01-01", periods=rows, freq="B", tz="America/New_York")
    close = np.linspace(100, 120, rows)
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1000, "Dividends": 0.0}, index=index)


def test_normalize_symbol_cleans_and_rejects_unsafe_input():
    assert normalize_symbol(" msft ") == "MSFT"
    assert normalize_symbol("brk-b") == "BRK-B"
    for bad in ("", "MSFT; DROP TABLE", "../etc/passwd", "A" * 20):
        with pytest.raises(DataToolError):
            normalize_symbol(bad)


@pytest.mark.parametrize("name, expected", [
    ("Microsoft Corporation", "Microsoft"), ("JPMorgan Chase & Co.", "JPMorgan Chase"), ("Tesla, Inc.", "Tesla"),
    ("Amazon.com, Inc.", "Amazon"), ("The Coca-Cola Company", "Coca-Cola"),
])
def test_short_company_name(name, expected):
    assert short_company_name(name) == expected


def test_normalize_prices_returns_clean_dated_frame_with_metadata():
    raw = raw_prices()
    raw.iloc[5, raw.columns.get_loc("Close")] = np.nan
    prices = normalize_prices(raw, "TEST", "1y")
    assert list(prices.columns) == ["open", "high", "low", "close", "volume"]
    assert prices.index.tz is None and prices.index.name == "date" and prices.index.is_monotonic_increasing
    assert len(prices) == 39 and prices["close"].notna().all()
    assert prices.attrs["rows_dropped"] == 1 and prices.attrs["source"] and prices.attrs["retrieved_at"]


def test_normalize_prices_rejects_empty_and_too_short_data():
    with pytest.raises(DataToolError):
        normalize_prices(pd.DataFrame(), "TEST", "1y")
    with pytest.raises(DataToolError):
        normalize_prices(raw_prices(rows=10), "TEST", "1y")


def test_normalize_statements_merges_sorts_and_reports_missing_items():
    periods = pd.to_datetime(["2025-12-31", "2024-12-31", "2023-12-31"])  # newest first, as providers return them
    raw = {
        "income_stmt": pd.DataFrame([[150.0, 120.0, np.nan], [27.0, 24.0, 20.0]], index=["Total Revenue", "Net Income"], columns=periods),
        "balance_sheet": pd.DataFrame([[60.0, 50.0, 50.0]], index=["Total Debt"], columns=periods),
        "cashflow": pd.DataFrame(),
    }
    fin = normalize_statements(raw, "TEST")
    assert list(fin.index.year) == [2024, 2025]  # oldest first; the period without revenue is dropped
    assert fin["revenue"].tolist() == [120.0, 150.0] and fin["total_debt"].tolist() == [50.0, 60.0]
    assert "free_cash_flow" in fin.attrs["missing_items"] and "revenue" not in fin.attrs["missing_items"]


def test_normalize_statements_requires_two_periods():
    raw = {"income_stmt": pd.DataFrame([[150.0]], index=["Total Revenue"], columns=pd.to_datetime(["2025-12-31"]))}
    with pytest.raises(DataToolError):
        normalize_statements(raw, "TEST")


def test_transient_errors_are_retried_but_validation_errors_are_not():
    attempts = []

    @_retry
    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("temporary")
        return "ok"

    flaky.retry.wait = wait_none()  # no pause between attempts in the test
    assert flaky() == "ok" and len(attempts) == 3

    calls = []

    @_retry
    def invalid():
        calls.append(1)
        raise DataToolError("unknown symbol")

    with pytest.raises(DataToolError):
        invalid()
    assert len(calls) == 1


def test_text_and_json_helpers():
    assert clean_text("  Acme &amp; Co <b>wins</b>\n deal ") == "Acme & Co wins deal"
    assert extract_json('Sure! {"a": 1, "b": [1, 2]} hope that helps') == {"a": 1, "b": [1, 2]}
    assert extract_json('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert extract_json("no json here") is None

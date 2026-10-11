import numpy as np
import pandas as pd
import pytest
from tenacity import wait_none

from src.data_tools import DataToolError, _retry, normalize_prices, normalize_statements, normalize_symbol, short_company_name
from src.tool_registry import ToolRegistry
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


def test_registry_logs_success_and_captures_failure():
    registry = ToolRegistry()
    registry.register("ok", lambda symbol: [1, 2, 3])
    registry.register("broken", lambda symbol: 1 / 0)
    good, bad = registry.call("ok", symbol="X"), registry.call("broken", symbol="X")
    assert good.ok and good.data == [1, 2, 3] and good.output_summary == "3 records"
    assert not bad.ok and "ZeroDivisionError" in bad.error and bad.data is None
    assert [entry.tool_name for entry in registry.log] == ["ok", "broken"]
    assert "data" not in registry.log_frame().columns  # payloads stay out of the log


def test_registry_rejects_unknown_tools():
    with pytest.raises(KeyError):
        ToolRegistry().call("delete_everything")


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


def test_local_chat_model_passes_tools_to_the_template_and_parses_tool_calls():
    import torch
    from langchain_core.messages import HumanMessage, ToolMessage

    from src.llm import LocalChatModel

    class Inputs(dict):
        def to(self, device):
            return self

    class Tokenizer:
        def apply_chat_template(self, messages, tools=None, **kwargs):
            self.messages, self.tools, self.kwargs = messages, tools, kwargs
            return Inputs(input_ids=torch.zeros((1, 3)))

        def decode(self, tokens, **kwargs):
            return self.reply

    class Model:
        device = "cpu"

        def generate(self, **kwargs):
            return torch.zeros((1, 5))

    tokenizer = Tokenizer()
    llm = LocalChatModel(model_id="fake", tokenizer=tokenizer, model=Model())
    tool = {"type": "function", "function": {"name": "market_agent", "description": "d", "parameters": {"type": "object", "properties": {}}}}

    tokenizer.reply = '<tool_call>\n{"name": "market_agent", "arguments": {"include_macro": true}}\n</tool_call>\n<tool_call>\n{"name": "news_agent", "arguments": {}}\n</tool_call>'
    first = llm.bind_tools([tool]).invoke([HumanMessage("question")])
    assert tokenizer.tools == [tool] and tokenizer.kwargs["enable_thinking"] is False
    assert first.content == "" and [(c["name"], c["args"], c["id"]) for c in first.tool_calls] == [
        ("market_agent", {"include_macro": True}, "call_0_0"), ("news_agent", {}, "call_0_1"),
    ]

    # The small model sometimes opens the next call without closing the previous one.
    tokenizer.reply = '<tool_call>\n{"name": "market_agent", "arguments": {"include_macro": false}}\n<tool_call>\n{"name": "news_agent", "arguments": {}}'
    unclosed = llm.invoke([HumanMessage("question")])
    assert [c["name"] for c in unclosed.tool_calls] == ["market_agent", "news_agent"] and unclosed.content == ""

    tokenizer.reply = "<think>\n\n</think>\n\nThe stock rose."
    answer = llm.invoke([HumanMessage("question"), first, ToolMessage("report", tool_call_id="call_0_0", name="market_agent")])
    assert answer.content == "The stock rose." and not answer.tool_calls
    assert [m["role"] for m in tokenizer.messages] == ["user", "assistant", "tool"]
    assert tokenizer.messages[1]["tool_calls"][0]["function"] == {"name": "market_agent", "arguments": '{"include_macro": true}'}
    assert tokenizer.messages[2] == {"role": "tool", "name": "market_agent", "tool_call_id": "call_0_0", "content": "report"}

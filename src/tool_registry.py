"""Registry of approved tools. Every call is timed, logged, and returned as a ToolCallResult."""

from __future__ import annotations

import time
from typing import Callable

import pandas as pd

from . import data_tools
from .models import ToolCallResult
from .utils import utc_now_iso


def summarize_output(data) -> str:
    """Short, log-friendly description of a tool's return value."""
    if isinstance(data, pd.DataFrame):
        span = ""
        if isinstance(data.index, pd.DatetimeIndex) and len(data):
            span = f", {data.index[0].date()} to {data.index[-1].date()}"
        return f"{len(data)} rows x {data.shape[1]} columns{span}"
    if isinstance(data, dict) and "articles" in data:
        return f"{len(data['articles'])} articles from feeds {data.get('feed_counts')}"
    if isinstance(data, dict):
        return "fields: " + ", ".join(list(data)[:8])
    if isinstance(data, (list, tuple)):
        return f"{len(data)} records"
    return type(data).__name__


def data_retrieved_at(data) -> str:
    """When the payload was fetched from its provider (earlier than now for a cached result)."""
    metadata = data if isinstance(data, dict) else getattr(data, "attrs", {})
    return metadata.get("retrieved_at") or utc_now_iso()


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Callable] = {}
        self.log: list[ToolCallResult] = []

    def register(self, name: str, fn: Callable) -> None:
        self._tools[name] = fn

    def call(self, name: str, **arguments) -> ToolCallResult:
        """Run an approved tool. Failures are captured in the result instead of raised,
        so one broken data source cannot terminate the whole research run."""
        if name not in self._tools:
            raise KeyError(f"Tool {name!r} is not registered. Approved tools: {sorted(self._tools)}")
        started = time.perf_counter()
        try:
            data = self._tools[name](**arguments)
            result = ToolCallResult(
                tool_name=name,
                arguments=arguments,
                status="success",
                duration_seconds=round(time.perf_counter() - started, 3),
                output_summary=summarize_output(data),
                retrieved_at=data_retrieved_at(data),
                data=data,
            )
        except Exception as exc:
            result = ToolCallResult(
                tool_name=name,
                arguments=arguments,
                status="error",
                duration_seconds=round(time.perf_counter() - started, 3),
                error=f"{type(exc).__name__}: {exc}",
            )
        self.log.append(result)
        return result

    def log_frame(self) -> pd.DataFrame:
        """The tool-call log as a DataFrame for display."""
        return pd.DataFrame([entry.model_dump() for entry in self.log])


def build_tool_registry() -> ToolRegistry:
    """Registry preloaded with the data tools."""
    registry = ToolRegistry()
    for tool in (
        data_tools.resolve_symbol, data_tools.get_company_profile, data_tools.get_price_history, data_tools.get_benchmark_history,
        data_tools.get_financial_statements, data_tools.get_company_news, data_tools.get_macro_series, data_tools.get_sec_filings,
    ):
        registry.register(tool.__name__, tool)
    return registry

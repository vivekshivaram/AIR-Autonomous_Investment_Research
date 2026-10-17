"""Charts built from the subagents' ``visualization_data``. Each function returns a Figure."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STOCK_COLOR, BENCHMARK_COLOR, NEUTRAL = "#1f5fa8", "#8a8a8a", "#c9c9c9"
IMPACT_COLORS = {"positive": "#2e8b57", "negative": "#c0392b", "neutral": "#8a8a8a", "unknown": "#c9c9c9"}


def _finish(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.figure.tight_layout()
    return ax.figure


def _line_colors(frame: pd.DataFrame) -> list[str]:
    return [STOCK_COLOR] + [BENCHMARK_COLOR] * (frame.shape[1] - 1)


def plot_performance(performance: pd.DataFrame):
    """Stock and benchmark, both rebased to 100 at the start of the window."""
    ax = performance.plot(figsize=(9, 4), color=_line_colors(performance), linewidth=1.8)
    ax.axhline(100, color=NEUTRAL, linewidth=0.8, linestyle="--")
    ax.legend(frameon=False)
    return _finish(ax, "Price performance, rebased to 100", "Date", "Index (start = 100)")


def plot_daily_return_distribution(performance: pd.DataFrame):
    """Histogram of daily returns for the stock and the benchmark, from the rebased performance series."""
    returns = performance.pct_change().dropna() * 100
    fig, ax = plt.subplots(figsize=(9, 4))
    low, high = returns.min().min(), returns.max().max()
    bins = np.linspace(low, high, 41)
    for column, color in zip(returns.columns, _line_colors(returns)):
        ax.hist(returns[column], bins=bins, alpha=0.55 if column != returns.columns[0] else 0.8, color=color,
                label=f"{column} (sd {returns[column].std():.2f}%)")
    ax.axvline(0, color=NEUTRAL, linewidth=0.8, linestyle="--")
    ax.legend(frameon=False)
    return _finish(ax, "Distribution of daily returns", "Daily return (%)", "Number of trading days")


def plot_rolling_volatility(volatility: pd.DataFrame, window: int = 21):
    """Rolling annualized volatility of the stock and the benchmark, in percent."""
    ax = volatility.plot(figsize=(9, 4), color=_line_colors(volatility), linewidth=1.8)
    ax.legend(frameon=False)
    return _finish(ax, f"Rolling {window}-day volatility, annualized", "Date", "Volatility (%)")


def plot_financial_trends(trends: pd.DataFrame, currency: str = ""):
    """Revenue, net income, and free cash flow per fiscal year."""
    labels = {"revenue": "Revenue", "net_income": "Net income", "free_cash_flow": "Free cash flow"}
    data = trends.dropna(axis=1, how="all").rename(columns=labels)
    ax = data.plot.bar(figsize=(9, 4), color=[STOCK_COLOR, "#5b9bd5", "#a9c8e8"][: data.shape[1]], width=0.8, rot=0)
    ax.legend(frameon=False)
    return _finish(ax, "Financial trends by fiscal year", "Fiscal year", f"{currency} billions".strip())


def plot_margins(margins: pd.DataFrame):
    """Gross, operating, net and free-cash-flow margins per fiscal year, in percent of revenue."""
    labels = {"gross_margin": "Gross", "operating_margin": "Operating", "net_margin": "Net", "fcf_margin": "Free cash flow"}
    data = margins.dropna(axis=1, how="all").rename(columns=labels)
    ax = data.plot(figsize=(9, 4), marker="o", linewidth=1.8)
    ax.legend(frameon=False, title="Margin")
    return _finish(ax, "Margins by fiscal year", "Fiscal year", "Share of revenue (%)")


def plot_news_funnel(funnel: dict[str, int]):
    """Article count remaining after each processing step of the news chain."""
    labels = [name.replace("_", " ").capitalize().replace(" llm", " LLM") for name in funnel]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(labels[::-1], list(funnel.values())[::-1], color=STOCK_COLOR)
    ax.bar_label(bars, padding=3)
    return _finish(ax, "News processing funnel", "Number of articles", "")


def plot_news_categories(category_counts: dict[str, int], impact_counts: dict[str, int]):
    """Event-type distribution (left) and impact distribution (right) of the classified articles."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [3, 2]})
    categories = pd.Series(category_counts).sort_values()
    left.barh(categories.index.str.replace("_", " "), categories.values, color=STOCK_COLOR)
    _finish(left, "Classified articles by event type", "Number of articles", "")
    impacts = pd.Series(impact_counts)
    right.bar(impacts.index, impacts.values, color=[IMPACT_COLORS.get(i, NEUTRAL) for i in impacts.index])
    return _finish(right, "By expected impact", "Impact", "Number of articles")

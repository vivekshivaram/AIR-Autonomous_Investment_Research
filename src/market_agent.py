"""Market Subagent: price performance, risk, and benchmark comparison.

All numbers come from deterministic Python functions (no LLM), so they are reproducible
and unit-tested. The LLM is reserved for language tasks: the news chain and the main agent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Settings, settings
from .evidence import EvidenceBuilder
from .macro_context import get_macro_context
from .models import Finding, SubagentInput
from .specialist_models import SpecialistResult as SubagentResult
from .tool_registry import ToolRegistry

AGENT_NAME = "market_agent"
DESCRIPTION = (
    "Share-price analysis: returns, volatility, drawdowns, trend, and comparison with a benchmark. "
    "Can add macroeconomic context such as interest rates."
)
TRADING_DAYS = 252
TRAILING_WINDOWS = {"1M": 21, "3M": 63, "6M": 126}  # trading days


# ------------------------------------------------------------- calculations
def daily_returns(close: pd.Series) -> pd.Series:
    """Simple day-over-day returns."""
    return close.pct_change().dropna()


def total_return(close: pd.Series) -> float:
    """Return from the first to the last observation."""
    return float(close.iloc[-1] / close.iloc[0] - 1)


def annualized_volatility(returns: pd.Series) -> float:
    """Sample standard deviation of daily returns, scaled to a year."""
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def rolling_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    """Annualized volatility over a rolling window of trading days."""
    return (returns.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)).dropna()


def max_drawdown(close: pd.Series) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    """Largest peak-to-trough decline, with the peak and trough dates."""
    drawdown = close / close.cummax() - 1
    trough = drawdown.idxmin()
    peak = close.loc[:trough].idxmax()
    return float(drawdown.min()), peak, trough


def beta(stock_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Sensitivity of the stock's daily returns to the benchmark's (covariance / variance)."""
    aligned = pd.concat([stock_returns, benchmark_returns], axis=1, join="inner").dropna()
    covariance = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1], ddof=1)
    return float(covariance[0, 1] / covariance[1, 1])


def trailing_returns(close: pd.Series, windows: dict[str, int] = TRAILING_WINDOWS) -> dict[str, float]:
    """Return over the most recent N trading days, for each window that fits in the data."""
    return {
        label: float(close.iloc[-1] / close.iloc[-1 - days] - 1)
        for label, days in windows.items()
        if len(close) > days
    }


# ------------------------------------------------------------------- agent
def run_market_agent(task: SubagentInput, tools: ToolRegistry, run_settings: Settings = settings) -> SubagentResult:
    """Analyze the stock's price history against the benchmark and return the standard result."""
    symbol, benchmark = task.symbol, run_settings.benchmark
    result = SubagentResult(agent=AGENT_NAME, status="success")

    price_call = tools.call("get_price_history", symbol=symbol, period=run_settings.price_period)
    if not price_call.ok:
        return SubagentResult(
            agent=AGENT_NAME, status="failed", warnings=[f"Price history unavailable for {symbol}: {price_call.error}"],
            limitations=["No market analysis was produced because price data could not be retrieved."],
        )
    prices = price_call.data
    close = prices["close"]

    # Losing the benchmark downgrades the result to partial; it never produces an invented number.
    bench_call = tools.call("get_benchmark_history", benchmark=benchmark, period=run_settings.price_period)
    bench_close = None
    if bench_call.ok:
        # Keep only the dates both series share, so every comparison covers the same window.
        both = pd.concat({symbol: close, benchmark: bench_call.data["close"]}, axis=1, join="inner")
        close, bench_close = both[symbol], both[benchmark]
    else:
        result.status = "partial"
        result.warnings.append(f"Benchmark {benchmark} unavailable: {bench_call.error}")
        result.limitations.append(f"No benchmark comparison: {benchmark} data could not be retrieved.")

    returns = daily_returns(close)
    start, end = close.index[0].date(), close.index[-1].date()
    evidence = EvidenceBuilder("MKT", prices.attrs["source"], as_of=str(end), retrieved_at=prices.attrs["retrieved_at"])
    stock_return = total_return(close)
    stock_vol = annualized_volatility(returns)
    drawdown, peak, trough = max_drawdown(close)
    trailing = trailing_returns(close)
    e_return = evidence.add(f"{symbol} total return from {start} to {end}", round(stock_return * 100, 2), "%")
    e_vol = evidence.add(f"{symbol} annualized volatility of daily returns", round(stock_vol * 100, 2), "%")
    performance = pd.DataFrame({symbol: close / close.iloc[0] * 100})
    volatility = pd.DataFrame({symbol: rolling_volatility(returns, run_settings.volatility_window) * 100})

    if bench_close is not None:
        bench_returns = daily_returns(bench_close)
        bench_return = total_return(bench_close)
        bench_vol = annualized_volatility(bench_returns)
        stock_beta = beta(returns, bench_returns)
        excess = stock_return - bench_return
        e_bench = evidence.add(f"{benchmark} total return from {start} to {end}", round(bench_return * 100, 2), "%")
        e_excess = evidence.add(f"{symbol} excess return versus {benchmark}", round(excess * 100, 2), "percentage points")
        e_bench_vol = evidence.add(f"{benchmark} annualized volatility of daily returns", round(bench_vol * 100, 2), "%")
        e_beta = evidence.add(f"{symbol} beta versus {benchmark} (daily returns)", round(stock_beta, 2))
        result.findings += [
            Finding(
                statement=f"{symbol} returned {stock_return:+.1%} from {start} to {end} versus {bench_return:+.1%} "
                f"for {benchmark}, {'outperforming' if excess >= 0 else 'underperforming'} by "
                f"{abs(excess) * 100:.1f} percentage points.",
                evidence_ids=[e_return, e_bench, e_excess],
            ),
            Finding(
                statement=f"Annualized volatility was {stock_vol:.1%} versus {bench_vol:.1%} for {benchmark}; "
                f"beta was {stock_beta:.2f}.",
                evidence_ids=[e_vol, e_bench_vol, e_beta],
            ),
        ]
        if stock_vol > 1.5 * bench_vol:
            result.risks.append(f"Volatility is {stock_vol / bench_vol:.1f}x the benchmark's, so price swings are larger.")
        if stock_beta > 1.2:
            result.risks.append(f"Beta of {stock_beta:.2f} means above-market sensitivity to broad sell-offs.")
        performance[benchmark] = bench_close / bench_close.iloc[0] * 100
        volatility[benchmark] = rolling_volatility(bench_returns, run_settings.volatility_window) * 100
        comparison = f" against {bench_return:+.1%} for {benchmark}"
    else:
        result.findings += [
            Finding(statement=f"{symbol} returned {stock_return:+.1%} from {start} to {end}.", evidence_ids=[e_return]),
            Finding(statement=f"Annualized volatility was {stock_vol:.1%}.", evidence_ids=[e_vol]),
        ]
        comparison = " (benchmark comparison unavailable)"
    result.summary = (
        f"{symbol} returned {stock_return:+.1%} from {start} to {end}{comparison}, "
        f"with {stock_vol:.1%} annualized volatility and a {drawdown:.1%} maximum drawdown."
    )

    e_drawdown = evidence.add(
        f"{symbol} maximum drawdown (peak {peak.date()}, trough {trough.date()})", round(drawdown * 100, 2), "%"
    )
    result.findings.append(
        Finding(
            statement=f"Maximum drawdown was {drawdown:.1%}, from the peak on {peak.date()} to the trough on {trough.date()}.",
            evidence_ids=[e_drawdown],
        )
    )
    if drawdown < -0.20:
        result.risks.append(f"A {drawdown:.1%} drawdown within the period shows material downside risk.")
    if trailing:
        e_trailing = evidence.add(f"{symbol} trailing returns", {k: round(v * 100, 2) for k, v in trailing.items()}, "%")
        trail_text = ", ".join(f"{label} {value:+.1%}" for label, value in trailing.items())
        result.findings.append(Finding(statement=f"Trailing returns: {trail_text}.", evidence_ids=[e_trailing]))

    # Trend: where the last close sits relative to its moving averages.
    for window in (50, 200):
        if len(close) >= window:
            average = float(close.rolling(window).mean().iloc[-1])
            gap = float(close.iloc[-1] / average - 1)
            e_sma = evidence.add(f"{symbol} last close relative to its {window}-day moving average", round(gap * 100, 2), "%")
            result.findings.append(
                Finding(
                    statement=f"The last close ({close.iloc[-1]:,.2f}) is {abs(gap):.1%} "
                    f"{'above' if gap >= 0 else 'below'} the {window}-day moving average ({average:,.2f}).",
                    evidence_ids=[e_sma],
                )
            )
            if window == 200 and gap < 0:
                result.risks.append("Price is below its 200-day moving average, a sign of a weak longer-term trend.")

    result.visualization_data = {
        "performance": performance,  # rebased to 100 at the start of the window
        "rolling_volatility": volatility,  # annualized, in percent
    }

    if task.include_macro:  # only when the main agent asked for it
        macro_findings, macro_limitations = get_macro_context(task.profile, tools, evidence)
        result.findings += macro_findings
        result.limitations += macro_limitations
        if macro_limitations:
            result.status = "partial"

    result.evidence = evidence.items
    result.limitations += [
        f"Based on adjusted daily closes from {start} to {end}; past performance does not predict future returns.",
        f"{benchmark} is a broad-market benchmark and may not reflect the company's sector.",
    ]
    if prices.attrs.get("rows_dropped"):
        result.limitations.append(f"{prices.attrs['rows_dropped']} price rows with missing closes were dropped.")
    return result

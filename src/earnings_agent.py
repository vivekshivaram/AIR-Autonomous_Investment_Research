"""Earnings Subagent: revenue, profit, margin, cash-flow, and balance-sheet trends.

Calculations are deterministic Python (no LLM) so every figure is reproducible.
"""

from __future__ import annotations

import pandas as pd
from .evidence import EvidenceBuilder
from .models import Finding, SubagentInput
from .specialist_models import SpecialistResult as SubagentResult
from .tool_registry import ToolRegistry
from .utils import safe_float

AGENT_NAME = "earnings_agent"
DESCRIPTION = (
    "Financial-statement analysis: revenue, profit, margins, cash flow, debt, and the latest SEC filing."
)


# ------------------------------------------------------------- calculations
def growth_rates(series: pd.Series) -> pd.Series:
    """Period-over-period growth. Undefined (NaN) when the prior value is zero or negative."""
    previous = series.shift(1)
    return ((series - previous) / previous).where(previous > 0)


def margins(financials: pd.DataFrame) -> pd.DataFrame:
    """Gross, operating, net, and free-cash-flow margins as fractions of revenue."""
    revenue = financials["revenue"]
    return pd.DataFrame(
        {
            "gross_margin": financials["gross_profit"] / revenue,
            "operating_margin": financials["operating_income"] / revenue,
            "net_margin": financials["net_income"] / revenue,
            "fcf_margin": financials["free_cash_flow"] / revenue,
        }
    )


def cagr(first: float, last: float, years: float) -> float | None:
    """Compound annual growth rate; None when it is not defined."""
    if not years or first is None or last is None or first <= 0 or last <= 0:
        return None
    return (last / first) ** (1 / years) - 1


def debt_to_equity(total_debt, equity) -> float | None:
    debt, equity = safe_float(total_debt), safe_float(equity)
    return debt / equity if debt is not None and equity and equity > 0 else None


def cash_conversion(operating_cash_flow, net_income) -> float | None:
    """Operating cash flow per unit of net income; below 1 means earnings outrun cash."""
    cash, income = safe_float(operating_cash_flow), safe_float(net_income)
    return cash / income if cash is not None and income and income > 0 else None


def _billions(value) -> str:
    return f"{value / 1e9:,.1f} bn"


# ------------------------------------------------------------------- agent
def run_earnings_agent(task: SubagentInput, tools: ToolRegistry) -> SubagentResult:
    """Analyze annual financial statements and return the standard subagent result."""
    symbol = task.symbol
    result = SubagentResult(agent=AGENT_NAME, status="success")

    call = tools.call("get_financial_statements", symbol=symbol)
    if not call.ok:
        return SubagentResult(
            agent=AGENT_NAME, status="failed", warnings=[f"Financial statements unavailable for {symbol}: {call.error}"],
            limitations=["No financial analysis was produced because statement data could not be retrieved."],
        )

    fin = call.data
    currency = task.profile.get("financial_currency") or task.profile.get("currency") or ""
    latest, prior = fin.iloc[-1], fin.iloc[-2]
    fiscal_end = fin.index[-1].date()
    label = f"fiscal year ended {fiscal_end}"
    evidence = EvidenceBuilder("ERN", fin.attrs["source"], as_of=str(fiscal_end), retrieved_at=fin.attrs["retrieved_at"])
    ratio = margins(fin)
    revenue_growth = growth_rates(fin["revenue"])
    income_growth = growth_rates(fin["net_income"])

    # Revenue
    rev_yoy = safe_float(revenue_growth.iloc[-1])
    years = (fin.index[-1] - fin.index[0]).days / 365.25
    rev_cagr = cagr(fin["revenue"].iloc[0], latest["revenue"], years)
    ids = [evidence.add(f"Revenue, {label}", float(latest["revenue"]), currency, kind="observed")]
    text = f"Revenue for the {label} was {currency} {_billions(latest['revenue'])}"
    if rev_yoy is not None:
        ids.append(evidence.add("Revenue growth versus the prior fiscal year", round(rev_yoy * 100, 2), "%"))
        text += f", {rev_yoy:+.1%} year over year"
        if rev_yoy < 0:
            result.risks.append(f"Revenue declined {abs(rev_yoy):.1%} year over year.")
    if rev_cagr is not None:
        ids.append(evidence.add(f"Revenue CAGR since {fin.index[0].date()}", round(rev_cagr * 100, 2), "%"))
        text += f" ({rev_cagr:+.1%} CAGR since {fin.index[0].year})"
    result.findings.append(Finding(statement=text + ".", evidence_ids=ids))

    # Net income and net margin
    if pd.notna(latest["net_income"]):
        ids = [evidence.add(f"Net income, {label}", float(latest["net_income"]), currency, kind="observed")]
        text = f"Net income was {currency} {_billions(latest['net_income'])}"
        income_yoy = safe_float(income_growth.iloc[-1])
        if income_yoy is not None:
            ids.append(evidence.add("Net income growth versus the prior fiscal year", round(income_yoy * 100, 2), "%"))
            text += f" ({income_yoy:+.1%} year over year)"
        net_now, net_before = safe_float(ratio["net_margin"].iloc[-1]), safe_float(ratio["net_margin"].iloc[-2])
        if net_now is not None and net_before is not None:
            ids.append(evidence.add("Net margin, latest versus prior year", [round(net_now * 100, 2), round(net_before * 100, 2)], "%"))
            text += f"; net margin was {net_now:.1%} versus {net_before:.1%} a year earlier"
            if net_now < net_before - 0.01:
                result.risks.append(f"Net margin compressed by {(net_before - net_now) * 100:.1f} percentage points.")
        if latest["net_income"] < 0:
            result.risks.append("The company reported a net loss in the latest fiscal year.")
        result.findings.append(Finding(statement=text + ".", evidence_ids=ids))

    # Gross and operating margins
    parts, ids = [], []
    for column, name in (("gross_margin", "Gross margin"), ("operating_margin", "Operating margin")):
        now, before = safe_float(ratio[column].iloc[-1]), safe_float(ratio[column].iloc[-2])
        if now is not None and before is not None:
            ids.append(evidence.add(f"{name}, latest versus prior year", [round(now * 100, 2), round(before * 100, 2)], "%"))
            parts.append(f"{name.lower()} {now:.1%} (prior year {before:.1%})")
            if now < before - 0.01:
                result.risks.append(f"{name} fell by {(before - now) * 100:.1f} percentage points.")
    if parts:
        result.findings.append(Finding(statement=f"Profitability: {'; '.join(parts)}.", evidence_ids=ids))

    # Cash flow. For banks and insurers, operating cash flow mostly reflects balance-sheet
    # movements (deposits, loans), so free cash flow and leverage ratios would mislead.
    is_financial = task.profile.get("sector") == "Financial Services"
    if is_financial:
        result.limitations.append("Free-cash-flow and leverage metrics are skipped: they are not meaningful for financial companies.")
    fcf = None if is_financial else safe_float(latest["free_cash_flow"])
    if fcf is not None:
        ids = [evidence.add(f"Free cash flow, {label}", fcf, currency, kind="observed")]
        text = f"Free cash flow was {currency} {_billions(fcf)}"
        fcf_margin = safe_float(ratio["fcf_margin"].iloc[-1])
        if fcf_margin is not None:
            text += f" ({fcf_margin:.1%} of revenue)"
        conversion = cash_conversion(latest["operating_cash_flow"], latest["net_income"])
        if conversion is not None:
            ids.append(evidence.add("Operating cash flow divided by net income", round(conversion, 2), "x"))
            text += f"; operating cash flow was {conversion:.2f}x net income"
            if conversion < 0.8:
                result.risks.append(f"Cash conversion of {conversion:.2f}x suggests earnings are not fully backed by cash.")
        if fcf < 0:
            result.risks.append("Free cash flow was negative in the latest fiscal year.")
        result.findings.append(Finding(statement=text + ".", evidence_ids=ids))

    # Balance sheet
    leverage = None if is_financial else debt_to_equity(latest["total_debt"], latest["stockholders_equity"])
    if leverage is not None:
        ids = [evidence.add(f"Total debt, {label}", float(latest["total_debt"]), currency, kind="observed")]
        text = f"Total debt was {currency} {_billions(latest['total_debt'])}"
        if pd.notna(latest["cash"]):
            ids.append(evidence.add(f"Cash and equivalents, {label}", float(latest["cash"]), currency, kind="observed"))
            text += f" against {currency} {_billions(latest['cash'])} of cash"
        ids.append(evidence.add("Debt-to-equity ratio", round(leverage, 2), "x"))
        result.findings.append(Finding(statement=f"{text}; debt-to-equity was {leverage:.2f}x.", evidence_ids=ids))
        if leverage > 2:
            result.risks.append(f"Debt-to-equity of {leverage:.2f}x indicates high financial leverage.")

    # Optional: point to the primary source documents on SEC EDGAR.
    filings_call = tools.call("get_sec_filings", symbol=symbol) if "get_sec_filings" in tools.allowed else None
    if filings_call and filings_call.ok:
        filing = filings_call.data.iloc[0]
        filing_id = evidence.add(
            f"Latest periodic SEC filing: form {filing['form']} for the period ended {filing['report_date']}",
            kind="observed", source=filings_call.data.attrs["source"], source_url=filing["url"],
            as_of=filing["filing_date"], retrieved_at=filings_call.data.attrs["retrieved_at"],
        )
        result.findings.append(
            Finding(
                statement=f"The most recent periodic filing is a {filing['form']} filed on {filing['filing_date']} "
                f"(period ended {filing['report_date']}).",
                evidence_ids=[filing_id],
            )
        )
    elif filings_call:
        result.status = "partial"
        result.limitations.append(f"SEC filing references unavailable ({filings_call.error}).")

    growth_text = f"{rev_yoy:+.1%}" if rev_yoy is not None else "n/a"
    result.summary = (
        f"For the {label}, {symbol} reported revenue of {currency} {_billions(latest['revenue'])} "
        f"({growth_text} year over year)"
        + (f" and net income of {currency} {_billions(latest['net_income'])}" if pd.notna(latest["net_income"]) else "")
        + (f", with free cash flow of {currency} {_billions(fcf)}" if fcf is not None else "")
        + "."
    )

    missing = fin.attrs.get("missing_items", [])
    if missing:
        result.status = "partial"
        result.warnings.append(f"Line items not reported by the data source for {symbol}: {', '.join(missing)}")
        result.limitations.append(f"Metrics that depend on {', '.join(missing)} could not be calculated.")
    result.limitations += [
        f"Based on {len(fin)} annual periods through {fiscal_end}; later quarterly results are not reflected.",
        "Line items are Yahoo Finance's standardized values and can differ from company-reported figures.",
    ]

    fiscal_years = [f"FY{d.year}" for d in fin.index]
    trends = (fin[["revenue", "net_income", "free_cash_flow"]] / 1e9).set_axis(fiscal_years)
    result.visualization_data = {
        "financial_trends": trends,  # billions of the reporting currency
        "margins": (ratio * 100).set_axis(fiscal_years),  # percent
        "currency": currency,
    }
    result.evidence = evidence.items
    return result

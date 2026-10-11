"""Optional macro context for the Market Subagent, chosen by the company's sector.

Observed macro facts (FRED data) are kept separate from the possible company
implication, which is an interpretation and is labelled as such.
"""

from __future__ import annotations

from .evidence import EvidenceBuilder
from .data_tools import SOURCE_FRED
from .models import Finding
from .tool_registry import ToolRegistry

RATES = [("DGS10", "10-year US Treasury yield", "%")]
CONSUMER = [("CPIAUCSL", "US consumer price index", "index"), ("UNRATE", "US unemployment rate", "%")]

# sector -> (FRED series to fetch, possible implication)
SECTOR_MACRO = {
    "Financial Services": (
        RATES + [("FEDFUNDS", "effective federal funds rate", "%")],
        "Interest-rate levels influence net interest margins, loan demand, and credit quality.",
    ),
    "Energy": (
        [("DCOILWTICO", "WTI crude oil spot price", "USD per barrel")],
        "Oil prices drive realized selling prices and upstream cash flow.",
    ),
    "Consumer Cyclical": (CONSUMER, "Inflation and employment shape discretionary consumer spending."),
    "Consumer Defensive": (CONSUMER, "Inflation affects input costs and pricing power; employment supports volumes."),
    "Technology": (RATES, "Higher long-term yields raise discount rates applied to long-duration growth earnings."),
    "Real Estate": (RATES, "Higher yields raise financing costs and pressure property valuations."),
    "Utilities": (RATES, "Higher yields raise financing costs and make dividend yields relatively less attractive."),
}


def get_macro_context(profile: dict, tools: ToolRegistry, evidence: EvidenceBuilder) -> tuple[list[Finding], list[str]]:
    """Fetch the macro series relevant to the company's sector.

    Returns the findings (one per observed series, then the possible implication, labelled as an
    interpretation) and the limitations for anything that could not be retrieved.
    """
    sector = profile.get("sector")
    if sector not in SECTOR_MACRO:
        return [], [f"No macro series is mapped to sector {sector!r}; macro context skipped."]

    findings, limitations = [], []
    series_list, implication = SECTOR_MACRO[sector]
    for series_id, label, unit in series_list:
        call = tools.call("get_macro_series", series_id=series_id)
        if not call.ok:
            limitations.append(f"Macro series {series_id} unavailable ({call.error}).")
            continue
        values = call.data["value"]
        first, last = values.iloc[0], values.iloc[-1]
        # Rates move in percentage points; price and index levels move in percent.
        change = f"{last - first:+.2f} percentage points" if unit == "%" else f"{last / first - 1:+.1%}"
        claim = (
            f"{label} was {last:,.2f}{'%' if unit == '%' else ' ' + unit} on {values.index[-1].date()}, "
            f"a change of {change} since {values.index[0].date()}"
        )
        evidence_id = evidence.add(
            claim, value=round(float(last), 4), unit=unit, kind="observed", source=f"{SOURCE_FRED} ({series_id})",
            as_of=str(values.index[-1].date()), retrieved_at=call.data.attrs["retrieved_at"],
        )
        findings.append(Finding(statement=f"Macro (observed): {claim}.", evidence_ids=[evidence_id]))
    if findings:
        observed = [evidence_id for finding in findings for evidence_id in finding.evidence_ids]
        findings.append(Finding(statement=f"Macro (interpretation, not an observed fact): {implication}", evidence_ids=observed))
    return findings, limitations

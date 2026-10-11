"""Small display helpers that keep the notebook cells short."""

from __future__ import annotations

import pandas as pd
from IPython.display import Markdown, display

from .models import SubagentResult


def md(text: str) -> None:
    display(Markdown(text))


def bullets(title: str, items: list[str]) -> str:
    body = "\n".join(f"- {item}" for item in items) if items else "- _none_"
    return f"**{title}**\n{body}"


def show_result(result: SubagentResult) -> None:
    """Render a subagent's standard output: summary, findings with evidence ids, risks, limitations."""
    findings = [finding.cited("`") for finding in result.findings]
    sections = [
        f"### `{result.agent}` — status: **{result.status}**",
        result.summary or "_no summary_",
        bullets("Findings (with evidence ids)", findings),
        bullets("Risks", result.risks),
    ]
    if result.catalysts:
        sections.append(bullets("Catalysts", result.catalysts))
    sections += [bullets("Limitations", result.limitations), bullets("Warnings", result.warnings)]
    md("\n\n".join(sections))


def evidence_frame(result: SubagentResult) -> pd.DataFrame:
    """Evidence items as a table: what was observed or calculated, from which source, as of when."""
    columns = ["id", "claim", "value", "unit", "kind", "source", "as_of", "retrieved_at"]
    return pd.DataFrame([item.model_dump() for item in result.evidence], columns=columns)


def records_frame(records: list, columns: list[str]) -> pd.DataFrame:
    """Selected fields of a list of pydantic models (articles, events, subagent calls, steps)."""
    frame = pd.DataFrame([record.model_dump() for record in records], columns=columns)
    return frame.astype(object).where(frame.notna(), "")


def funnel_frame(funnel: dict[str, int]) -> pd.DataFrame:
    """Article count after each news-processing step, with the share of ingested articles remaining."""
    frame = pd.DataFrame({"articles": pd.Series(funnel)})
    frame["removed_in_step"] = (-frame["articles"].diff()).fillna(0).astype(int)
    frame["share_of_ingested"] = (frame["articles"] / frame["articles"].iloc[0]).map("{:.0%}".format)
    return frame.rename_axis("step")

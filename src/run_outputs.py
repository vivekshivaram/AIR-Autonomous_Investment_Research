"""Files every run leaves behind under ``outputs/`` (gitignored): reports, a JSON run log and charts.

    outputs/reports/initial_report.md, final_report.md        latest run
    outputs/reports/<SYMBOL>_<run_id>_{initial,final}.md      every run
    outputs/logs/latest_run.json, <SYMBOL>_<run_id>.json      plan, routing, tools, evaluations, reflection
    outputs/charts/<name>.png                                 charts saved by ``save_figure``

The run log leaves out DataFrames and raw tool payloads, and it never contains API keys: the LLM
client's configuration is not part of the agent state.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .config import Settings
from .models import AgentState
from .report_models import ResearchReport


def save_run_outputs(state: AgentState, settings: Settings | None = None) -> dict[str, str]:
    """Write the first draft, the final report and the run log. Returns the paths that were written."""
    settings = settings or Settings()
    root = Path(settings.output_dir)
    reports, logs = root / "reports", root / "logs"
    reports.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    stem = f"{state.symbol}_{state.run_id}"
    files: dict[str, str] = {}
    for label, payload in (("initial", state.draft_report), ("final", state.final_report)):
        if payload:
            markdown = ResearchReport.from_dict(payload).to_markdown()
            for path in (reports / f"{label}_report.md", reports / f"{stem}_{label}.md"):
                path.write_text(markdown, encoding="utf-8")
            files[f"{label}_report"] = str(reports / f"{stem}_{label}.md")
    log = json.dumps(run_log(state), indent=2, default=_json_default)
    for path in (logs / "latest_run.json", logs / f"{stem}.json"):
        path.write_text(log, encoding="utf-8")
    files["run_log"] = str(logs / f"{stem}.json")
    return files


def run_log(state: AgentState) -> dict[str, Any]:
    """Everything needed to inspect a run afterwards, without data frames or report snapshots per cycle."""
    aggregated = state.aggregated or {}
    return {
        "run_id": state.run_id,
        "symbol": state.symbol,
        "question": state.research_question,
        "started_at": state.started_at,
        "plan": state.plan.to_dict() if state.plan else None,
        "routing": [decision.to_dict() for decision in state.routing_log],
        "tool_calls": state.tool_call_log,
        "execution": state.execution_log,
        "coverage": aggregated.get("coverage"),
        "evidence": aggregated.get("evidence"),
        "memories_used": state.memories_used,
        "warnings": [warning.to_dict() for warning in state.warnings],
        "evaluations": [{k: v for k, v in entry.items() if k != "report"} for entry in state.evaluation_history],
        "draft_report": state.draft_report,
        "final_report": state.final_report,
        "reflection": getattr(state, "reflection", None),
    }


def save_figure(figure: Any, name: str, settings: Settings | None = None) -> Path:
    """Save a Matplotlib figure as outputs/charts/<name>.png and return the path."""
    settings = settings or Settings()
    folder = Path(settings.output_dir) / "charts"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    return path


def _json_default(value: Any) -> Any:
    """JSON fallback for the objects found in agent state."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if callable(getattr(value, "model_dump", None)):
        return value.model_dump(mode="json")
    if hasattr(value, "shape") and hasattr(value, "dtypes"):
        return {"type": type(value).__name__, "shape": list(value.shape)}
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    item = getattr(value, "item", None)
    if callable(item):  # NumPy scalars
        number = item()
        return None if isinstance(number, float) and not math.isfinite(number) else number
    return str(value)

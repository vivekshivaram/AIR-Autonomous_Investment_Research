"""Plan, routing and execution visualizations and tables for the notebook (Engineer 1 scope)."""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from .models import AgentState, ResearchPlan, RoutingDecision, Status

AGENT_COLORS = {"market_agent": "#4C78A8", "earnings_agent": "#F58518", "news_agent": "#54A24B"}
STATUS_COLORS = {
    Status.SUCCESS: "#54A24B",
    Status.PARTIAL: "#EECA3B",
    Status.FAILED: "#E45756",
    Status.SKIPPED: "#9D9D9D",
}


def routing_table(routing: Sequence[RoutingDecision]) -> pd.DataFrame:
    """Routing decisions as a table: selected or skipped, reason, confidence and execution status."""
    return pd.DataFrame(
        [
            {
                "decision": "SELECTED" if r.selected else "skipped",
                "subagent": r.subagent,
                "task": r.task,
                "reason": r.reason,
                "confidence": r.confidence,
                "status": r.execution_status,
            }
            for r in routing
        ]
    )


def plan_table(plan: ResearchPlan) -> pd.DataFrame:
    """Plan steps as a table: subagent, description, dependencies, tools, completion criterion and status."""
    return pd.DataFrame(
        [
            {
                "step": s.step_id,
                "subagent": s.subagent,
                "description": s.description,
                "depends_on": ", ".join(s.depends_on) or "-",
                "tools": ", ".join(s.tools),
                "completion_criteria": s.completion_criteria,
                "status": s.status,
            }
            for s in plan.steps
        ]
    )


def tool_log_table(state: AgentState) -> pd.DataFrame:
    """The tool-call log as a table; payloads are never logged."""
    cols = ["caller", "tool_name", "status", "duration_ms", "output_summary", "error", "retrieved_at"]
    return pd.DataFrame(state.tool_call_log, columns=cols)


def execution_table(state: AgentState) -> pd.DataFrame:
    """One row per executed step with status, duration and the number of tool calls."""
    return pd.DataFrame(state.execution_log)


def plot_plan(plan: ResearchPlan, ax: plt.Axes | None = None) -> plt.Axes:
    """Dependency graph: columns are dependency depth, colour is the assigned subagent."""
    ax = ax or plt.subplots(figsize=(8, 3.5))[1]
    depth: dict[str, int] = {}

    def depth_of(step_id: str) -> int:
        if step_id not in depth:
            deps = plan.get_step(step_id).depends_on
            depth[step_id] = 0 if not deps else 1 + max(depth_of(d) for d in deps)
        return depth[step_id]

    columns: dict[int, list[str]] = {}
    for step in plan.steps:
        columns.setdefault(depth_of(step.step_id), []).append(step.step_id)
    pos = {sid: (col, -row) for col, ids in columns.items() for row, sid in enumerate(ids)}

    for step in plan.steps:
        for dep in step.depends_on:
            ax.annotate("", xy=pos[step.step_id], xytext=pos[dep], arrowprops={"arrowstyle": "->", "lw": 1.5})
    for step in plan.steps:
        x, y = pos[step.step_id]
        ax.scatter([x], [y], s=2600, color=AGENT_COLORS.get(step.subagent, "#888"), zorder=3)
        ax.text(x, y, f"{step.step_id}\n{step.subagent.replace('_agent', '')}", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold", zorder=4)
    ax.set_xlim(-0.6, max(columns) + 0.6)
    ax.set_ylim(-max(len(v) for v in columns.values()) + 0.4, 0.6)
    ax.set_title(f"Research plan for {plan.symbol} ({plan.scope}, {plan.planner_mode} planner)")
    ax.axis("off")
    return ax


def plot_routing_matrix(cases: Mapping[str, Sequence[RoutingDecision]], agents: Sequence[str]) -> plt.Axes:
    """Heatmap of selected (1) vs skipped (0) subagents per question, proving routing is conditional."""
    data = pd.DataFrame(
        {label: {a: int(any(r.subagent == a and r.selected for r in routing)) for a in agents} for label, routing in cases.items()}
    ).T
    fig, ax = plt.subplots(figsize=(1.6 * len(agents) + 3, 0.55 * len(cases) + 1.5))
    ax.imshow(data.values, cmap="Blues", vmin=0, vmax=1.4, aspect="auto")
    ax.set_xticks(range(len(agents)), [a.replace("_agent", "") for a in agents])
    ax.set_yticks(range(len(data)), [label if len(label) <= 70 else label[:67] + "..." for label in data.index])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, "run" if data.iat[i, j] else "skip", ha="center", va="center",
                    color="white" if data.iat[i, j] else "#666")
    ax.set_title("Routing decisions by research question")
    ax.set_xlabel("Specialist subagent")
    ax.set_ylabel("Research question")
    run_color, skip_color = plt.cm.Blues(1 / 1.4), plt.cm.Blues(0.0)  # the same colours imshow uses above
    ax.legend(handles=[Patch(color=run_color, label="run (selected)"), Patch(facecolor=skip_color, edgecolor="#bbbbbb", label="skip (not selected)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    return ax


def plot_execution_timeline(state: AgentState) -> plt.Axes:
    """Per-step duration, coloured by final status."""
    log = state.execution_log
    fig, ax = plt.subplots(figsize=(8, 0.6 * max(len(log), 2) + 1.2))
    labels = [f"{e['step_id']} {e['subagent']}" for e in log]
    ax.barh(labels, [e["duration_ms"] for e in log], color=[STATUS_COLORS.get(e["status"], "#888") for e in log])
    ax.invert_yaxis()
    ax.set_xlabel("Duration (ms)")
    ax.set_ylabel("Plan step")
    ax.set_title("Subagent execution")
    ax.legend(handles=[Patch(color=c, label=s) for s, c in STATUS_COLORS.items()], loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8, frameon=False)
    fig.tight_layout()
    return ax


def plot_tool_calls(state: AgentState) -> plt.Axes:
    """Tool-call durations by caller; failed calls are red."""
    df = tool_log_table(state)
    fig, ax = plt.subplots(figsize=(8, 0.45 * max(len(df), 2) + 1.2))
    labels = [f"{r.caller}: {r.tool_name}" for r in df.itertuples()]
    ax.barh(labels, df["duration_ms"], color=[STATUS_COLORS[Status.SUCCESS] if s == Status.SUCCESS else STATUS_COLORS[Status.FAILED] for s in df["status"]])
    ax.invert_yaxis()
    ax.set_xlabel("Duration (ms)")
    ax.set_ylabel("Caller: tool")
    ax.set_title("Tool calls (conditional tool selection)")
    ax.legend(handles=[Patch(color=STATUS_COLORS[Status.SUCCESS], label="success"), Patch(color=STATUS_COLORS[Status.FAILED], label="failed")],
              loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    return ax

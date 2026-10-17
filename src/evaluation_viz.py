"""Charts for the Evaluator-Optimizer, the learning loop and the agent workflow (Engineer 3).

Every function returns a Matplotlib Figure with a title, axis labels with units and a legend, in the
same visual style as ``visualizations.py``. ``chart_notes`` writes the matching interpretations.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

from .models import SUBAGENT_EARNINGS, SUBAGENT_MARKET, SUBAGENT_NEWS, AgentState, Status

FIRST_COLOR, FINAL_COLOR, THRESHOLD_COLOR, FLOOR_COLOR = "#a9c8e8", "#1f5fa8", "#c0392b", "#8a8a8a"
STATUS_FILL = {
    Status.SUCCESS: "#54A24B",
    Status.PARTIAL: "#EECA3B",
    Status.FAILED: "#E45756",
    "not selected": "#d0d0d0",
}
NEUTRAL_FILL = "#4C78A8"


def _finish(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.figure.tight_layout()
    return ax.figure


def _legend_below(ax, ncol: int) -> None:
    """Legend under the plot, so it never covers bars that reach the right-hand edge."""
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=ncol, fontsize=9)


def _as_dict(evaluation: Any) -> dict[str, Any]:
    return evaluation if isinstance(evaluation, dict) else evaluation.to_dict()


def plot_criterion_scores(first: Any, final: Any):
    """Score per rubric criterion for the first draft and the final report, with the criterion floor."""
    first, final = _as_dict(first), _as_dict(final)
    labels = [c["label"] for c in first["criteria"]][::-1]
    before = [c["score"] for c in first["criteria"]][::-1]
    after = [c["score"] for c in final["criteria"]][::-1]
    positions = range(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh([p + 0.2 for p in positions], before, height=0.4, color=FIRST_COLOR, label=f"First draft (v{first['report_version']})")
    ax.barh([p - 0.2 for p in positions], after, height=0.4, color=FINAL_COLOR, label=f"Final report (v{final['report_version']})")
    ax.axvline(first["floor"], color=FLOOR_COLOR, linestyle=":", linewidth=1.2, label=f"Criterion floor ({first['floor']:.0f})")
    ax.set_yticks(list(positions), labels)
    ax.set_xlim(0, 105)
    _legend_below(ax, 3)
    return _finish(ax, "Evaluation by criterion, first draft against final report", "Score (0 to 100)", "")


def plot_score_by_cycle(history: Sequence[Any]):
    """Overall score after each refinement cycle, with the acceptance threshold and each decision."""
    history = [_as_dict(h) for h in history]
    cycles = [h["cycle"] for h in history]
    scores = [h["overall_score"] for h in history]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(cycles, scores, marker="o", color=FINAL_COLOR, linewidth=2, label="Overall score")
    ax.axhline(history[0]["threshold"], color=THRESHOLD_COLOR, linestyle="--", linewidth=1, label=f"Acceptance threshold ({history[0]['threshold']:.0f})")
    for h in history:
        ax.annotate(f"v{h['report_version']}: {h['overall_score']:.1f}\n{h['decision']}", (h["cycle"], h["overall_score"]),
                    textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    ax.set_xticks(cycles)
    ax.set_ylim(min(scores + [history[0]["threshold"]]) - 15, 108)
    ax.legend(frameon=False, loc="lower right")
    return _finish(ax, "Overall quality score by refinement cycle", "Refinement cycle (0 = first draft)", "Overall score (0 to 100)")


def plot_quality_across_runs(runs: Sequence[Mapping[str, Any]], threshold: float = 80.0):
    """First-draft and final score for each run; labels note how many lessons the run applied.

    ``runs`` holds dicts with ``label``, ``initial``, ``final`` and ``lessons_applied`` (a list of memory IDs).
    """
    positions = range(len(runs))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    first = ax.bar([p - 0.2 for p in positions], [r["initial"] for r in runs], width=0.4, color=FIRST_COLOR, label="First draft")
    final = ax.bar([p + 0.2 for p in positions], [r["final"] for r in runs], width=0.4, color=FINAL_COLOR, label="Final report")
    ax.bar_label(first, fmt="%.1f", padding=2, fontsize=9)
    ax.bar_label(final, fmt="%.1f", padding=2, fontsize=9)
    ax.axhline(threshold, color=THRESHOLD_COLOR, linestyle="--", linewidth=1, label=f"Acceptance threshold ({threshold:.0f})")
    ax.set_xticks(list(positions), [f"{r['label']}\n{len(r.get('lessons_applied') or [])} lesson(s) applied" for r in runs])
    ax.set_ylim(0, 112)
    _legend_below(ax, 3)
    return _finish(ax, "Report quality across repeated runs", "", "Overall score (0 to 100)")


def plot_issue_counts(first: Any, final: Any):
    """Number of evaluator issues per criterion, first draft against final report."""
    first, final = _as_dict(first), _as_dict(final)
    labels = [c["label"] for c in first["criteria"]][::-1]  # reversed so the rubric reads top to bottom
    names = [c["name"] for c in first["criteria"]][::-1]

    def counts(evaluation):
        return [sum(1 for i in evaluation["issues"] if i["criterion"] == name) for name in names]

    positions = range(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.barh([p + 0.2 for p in positions], counts(first), height=0.4, color=FIRST_COLOR, label="First draft")
    ax.barh([p - 0.2 for p in positions], counts(final), height=0.4, color=FINAL_COLOR, label="Final report")
    ax.set_yticks(list(positions), labels)
    _legend_below(ax, 2)
    return _finish(ax, "Evaluator issues by criterion", "Number of issues", "")


def plot_agent_workflow(state: AgentState | None = None):
    """The agent's workflow: for one run, which specialists ran, the refinement loop and the memory traffic.

    Without a state it draws the architecture itself, with the three workflow patterns labelled.
    """
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    architecture = state is None
    if architecture:
        statuses = {agent: "available" for agent in (SUBAGENT_MARKET, SUBAGENT_EARNINGS, SUBAGENT_NEWS)}
        history, stored, read = [], [], "lessons"
    else:
        statuses = {step.subagent: step.status for step in (state.plan.steps if state.plan else [])}
        history = state.evaluation_history or []
        reflection = getattr(state, "reflection", None) or {}
        stored = [l for l in reflection.get("lessons", []) if l.get("memory_id")]
        read = f"{len(state.memories_used)} lesson(s) read"

    def box(x, y, text, fill=NEUTRAL_FILL, width=1.5, height=0.85, color="white"):
        ax.add_patch(FancyBboxPatch((x - width / 2, y - height / 2), width, height, boxstyle="round,pad=0.04", facecolor=fill, edgecolor="none"))
        ax.text(x, y, text, ha="center", va="center", fontsize=8.5, color=color, fontweight="bold", wrap=True)

    def arrow(start, end, text="", rad=0.0, skipped=False):
        style = {"arrowstyle": "->", "lw": 1.0 if skipped else 1.3, "color": "#c0c0c0" if skipped else "#555555",
                 "linestyle": "--" if skipped else "-", "connectionstyle": f"arc3,rad={rad}"}
        ax.annotate("", xy=end, xytext=start, arrowprops=style)
        if text:
            ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.18, text, ha="center", fontsize=7.5, color="#333333")

    box(0.9, 3.1, f"SQLite memory\n{read}", fill="#7f7f7f")
    box(2.7, 3.1, "Planner and\nrouter")
    arrow((1.65, 3.1), (1.95, 3.1))
    lanes = [(SUBAGENT_MARKET, "Market", 4.9), (SUBAGENT_EARNINGS, "Earnings", 3.1), (SUBAGENT_NEWS, "News\n(5-stage chain)", 1.3)]
    for agent, label, y in lanes:
        status = statuses.get(agent, "not selected")
        fill = NEUTRAL_FILL if architecture else STATUS_FILL.get(status, STATUS_FILL["not selected"])
        text = label if architecture else f"{label}\n{status}"
        box(4.7, y, text, fill=fill, color="black" if status in (Status.PARTIAL, "not selected") else "white")
        arrow((3.45, 3.1), (3.95, y), skipped=agent not in statuses)
        if agent in statuses:
            arrow((5.45, y), (5.95, 3.1))
    box(6.7, 3.1, "Evidence\naggregator")
    box(8.4, 3.1, "Report\nwriter")
    arrow((7.45, 3.1), (7.65, 3.1))
    cycles = max(len(history) - 1, 0)
    box(10.2, 4.3, "Evaluator", fill="#F58518")
    box(10.2, 1.9, "Optimizer", fill="#F58518")
    arrow((9.15, 3.3), (9.45, 4.2))
    arrow((10.0, 3.85), (10.0, 2.35), "refine (max 2)" if architecture else f"refine x{cycles}", rad=0.0)
    arrow((10.4, 2.35), (10.4, 3.85))
    final_score = max((h["overall_score"] for h in history), default=None)
    box(11.4, 5.6, "Final report" + (f"\n{final_score:.1f}/100" if final_score is not None else ""), fill="#2e8b57", width=1.1, height=0.8)
    arrow((10.6, 4.65), (11.1, 5.25))
    box(6.7, 0.45, "Reflection: lessons stored" if architecture else f"Reflection: {len(stored)} lesson(s) stored",
        fill="#7f7f7f", width=3.0, height=0.6)
    arrow((10.2, 1.45), (8.25, 0.45))
    arrow((5.2, 0.45), (0.9, 2.65), rad=-0.15)
    if architecture:
        # Where each required workflow pattern lives in the design.
        for x, y, text in ((3.2, 5.7, "Routing"), (4.7, 1.93, "Prompt chaining"), (10.2, 5.05, "Evaluator-Optimizer")):
            ax.text(x, y, text, ha="center", fontsize=9, color="#c0392b", fontweight="bold")
        ax.set_title("Agent architecture: planning, routing, specialists, evaluator-optimizer and memory", loc="left", fontweight="bold")
        ax.legend(handles=[Patch(color=NEUTRAL_FILL, label="agent component"), Patch(color="#F58518", label="evaluator-optimizer loop"),
                           Patch(color="#7f7f7f", label="memory and reflection")],
                  loc="upper left", fontsize=8, frameon=False)
    else:
        ax.legend(
            handles=[Patch(color=c, label=s) for s, c in STATUS_FILL.items()],
            loc="upper left", fontsize=8, frameon=False, title="Specialist status", title_fontsize=8,
        )
        ax.set_title(f'Agent workflow for {state.symbol}: "{state.research_question}"', loc="left", fontweight="bold")
    fig.tight_layout()
    return fig

"""Short interpretations written from the data behind each chart.

Market data, news and LLM wording change from run to run, so a fixed paragraph under a chart goes stale.
Each function here reads the same data the chart plots and returns one to three plain sentences, which
the notebook shows directly below the chart.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from .models import AgentState, ResearchPlan, RoutingDecision, Status


def _pct(value: float) -> str:
    return f"{value:+.1f}%"


def performance(perf: pd.DataFrame) -> str:
    """Rebased price chart: where the stock and the benchmark ended, and the stock's low point."""
    stock = perf.columns[0]
    start, end = perf.index[0].date(), perf.index[-1].date()
    last = perf.iloc[-1]
    text = f"{stock} ended the window at {last[stock]:.1f}, a {_pct(last[stock] - 100)} change since {start}"
    if perf.shape[1] > 1:
        bench = perf.columns[1]
        gap = last[stock] - last[bench]
        text += (
            f", while {bench} ended at {last[bench]:.1f} ({_pct(last[bench] - 100)}). "
            f"The stock {'outperformed' if gap >= 0 else 'trailed'} the benchmark by {abs(gap):.1f} index points"
        )
    low_date = perf[stock].idxmin().date()
    return text + f". Its lowest level was {perf[stock].min():.1f} on {low_date}, and the data ends on {end}."


def daily_returns(perf: pd.DataFrame) -> str:
    """Return histogram: typical daily move, extremes and how often the stock rose."""
    returns = perf.pct_change().dropna() * 100
    stock = returns.columns[0]
    series = returns[stock]
    text = (
        f"On a typical day {stock} moved about {series.std():.2f}% (standard deviation of daily returns), "
        f"and it closed higher on {100 * (series > 0).mean():.0f}% of {len(series)} trading days. "
        f"The worst day was {series.min():+.2f}% on {series.idxmin().date()} and the best {series.max():+.2f}% on {series.idxmax().date()}."
    )
    if returns.shape[1] > 1:
        bench = returns.columns[1]
        ratio = series.std() / returns[bench].std()
        text += f" Its daily moves were {ratio:.1f} times as wide as {bench}'s, so its histogram is the flatter of the two."
    return text


def volatility(vol: pd.DataFrame, window: int = 21) -> str:
    """Rolling volatility chart: average, peak and latest level, against the benchmark."""
    stock = vol.columns[0]
    series = vol[stock]
    text = (
        f"{stock}'s {window}-day annualised volatility averaged {series.mean():.1f}% and peaked at {series.max():.1f}% "
        f"on {series.idxmax().date()}; the latest reading is {series.iloc[-1]:.1f}%"
    )
    if vol.shape[1] > 1:
        bench = vol.columns[1]
        text += f", against an average of {vol[bench].mean():.1f}% for {bench}"
    return text + "."


def financial_trends(trends: pd.DataFrame, currency: str = "") -> str:
    """Revenue, net income and free cash flow by fiscal year."""
    first, last = trends.index[0], trends.index[-1]
    parts = []
    labels = {"revenue": "Revenue", "net_income": "Net income", "free_cash_flow": "Free cash flow"}
    for column, label in labels.items():
        if column not in trends or trends[column].isna().all():
            continue
        series = trends[column].dropna()
        change = (series.iloc[-1] / series.iloc[0] - 1) * 100 if series.iloc[0] > 0 else None
        trend = f" ({change:+.0f}%)" if change is not None else ""
        parts.append(f"{label} went from {currency} {series.iloc[0]:,.1f} bn in {series.index[0]} to {currency} {series.iloc[-1]:,.1f} bn in {series.index[-1]}{trend}")
    return "; ".join(parts) + f". The chart covers {len(trends)} fiscal years, {first} to {last}." if parts else "No financial trend data was available."


def margins(margin_table: pd.DataFrame) -> str:
    """Margins by fiscal year: first and latest value of each margin."""
    labels = {"gross_margin": "gross", "operating_margin": "operating", "net_margin": "net", "fcf_margin": "free-cash-flow"}
    parts = []
    for column, label in labels.items():
        if column in margin_table and margin_table[column].notna().sum() >= 2:
            series = margin_table[column].dropna()
            parts.append(f"{label} margin {series.iloc[0]:.1f}% to {series.iloc[-1]:.1f}%")
    if not parts:
        return "Margins could not be calculated from the reported line items."
    return f"From {margin_table.index[0]} to {margin_table.index[-1]}: " + "; ".join(parts) + "."


def news_funnel(funnel: Mapping[str, int]) -> str:
    """News funnel: how many headlines survived each preprocessing step."""
    ingested = funnel.get("ingested", 0)
    selected = funnel.get("selected_for_llm", 0)
    text = (
        f"{ingested} headlines came in from the feeds; {funnel.get('valid_and_recent', 0)} were recent and usable, "
        f"{funnel.get('after_exact_dedup', 0)} remained after exact duplicates were merged, "
        f"{funnel.get('relevant', 0)} were about the company and {funnel.get('after_near_dedup', 0)} distinct stories were left "
        f"after near duplicates were clustered."
    )
    if ingested:
        text += f" The {selected} most widely covered and most recent stories ({100 * selected / ingested:.0f}% of the intake) went to the language model"
        if "material_events" in funnel:
            text += f", which extracted {funnel['material_events']} material events"
        text += "."
    return text


def news_categories(categories: Mapping[str, int], impacts: Mapping[str, int]) -> str:
    """Event-type and impact distribution of the classified headlines."""
    if not categories:
        return "No headlines were classified."
    top, count = max(categories.items(), key=lambda item: item[1])
    total = sum(categories.values())
    impact_text = ", ".join(f"{n} {label}" for label, n in sorted(impacts.items(), key=lambda item: -item[1]))
    return (
        f"The most common event type was {top.replace('_', ' ')} ({count} of {total} classified headlines). "
        f"By expected impact: {impact_text}. These labels are language-model judgements, not observed facts."
    )


def routing(cases: Mapping[str, Sequence[RoutingDecision]]) -> str:
    """Routing matrix: how the number of selected specialists varied with the question."""
    counts = {question: sum(1 for r in routing if r.selected) for question, routing in cases.items()}
    full = [q for q, n in counts.items() if n == max(counts.values())]
    single = [q for q, n in counts.items() if n == 1]
    return (
        f"Across {len(cases)} questions the planner selected between {min(counts.values())} and {max(counts.values())} specialists. "
        f"{len(single)} question(s) needed a single specialist, and only {', '.join(repr(q) for q in full)} used the most. "
        "Because the selection changes with the question, routing is conditional rather than a fixed pipeline."
    )


def plan(research_plan: ResearchPlan) -> str:
    """Plan graph: number of steps, dependencies and planner mode."""
    deps = [f"{s.step_id} waits for {', '.join(s.depends_on)}" for s in research_plan.steps if s.depends_on]
    dependency_text = "; ".join(deps) if deps else "the steps are independent and run in plan order"
    return (
        f"The {research_plan.planner_mode.replace('_', '-')} planner produced {len(research_plan.steps)} step(s) for a "
        f"{research_plan.scope} question; {dependency_text}."
    )


def execution(state: AgentState) -> str:
    """Execution timeline: step outcomes and the slowest step."""
    log = state.execution_log
    if not log:
        return "No step was executed."
    ok = sum(1 for e in log if e["status"] in (Status.SUCCESS, Status.PARTIAL))
    slowest = max(log, key=lambda e: e["duration_ms"])
    return (
        f"{ok} of {len(log)} step(s) completed. The slowest was {slowest['step_id']} {slowest['subagent']} "
        f"at {slowest['duration_ms'] / 1000:.1f} s; cached data makes repeat runs of the data tools nearly instant."
    )


def tool_calls(state: AgentState) -> str:
    """Tool-call chart: how many calls, how many failed and which tools were never called."""
    calls = state.tool_call_log
    if not calls:
        return "No tool was called."
    failed = [c for c in calls if c["status"] != Status.SUCCESS]
    used = sorted({c["tool_name"] for c in calls})
    return (
        f"{len(calls)} tool call(s) across {len(used)} tool(s) ({', '.join(used)}); {len(failed)} failed. "
        "Each specialist could call only the tools its plan step listed, so tools the plan did not select never ran."
    )


def criterion_scores(first: Mapping[str, Any], final: Mapping[str, Any]) -> str:
    """Before-and-after criterion chart: biggest gains and the weakest remaining criterion."""
    before = {c["name"]: c for c in first["criteria"]}
    gains = sorted(
        ((c["score"] - before[c["name"]]["score"], c["label"], before[c["name"]]["score"], c["score"]) for c in final["criteria"]),
        reverse=True,
    )
    weakest = min(final["criteria"], key=lambda c: c["score"])
    improved = [f"{label} {old:.0f} to {new:.0f}" for gain, label, old, new in gains if gain > 0.05][:3]
    if improved:
        lead = "Refinement raised " + ", ".join(improved) + "."
    else:
        lead = "No criterion changed between the first draft and the final report."
    return f"{lead} The weakest criterion in the final report is {weakest['label'].lower()} at {weakest['score']:.0f}/100."


def score_by_cycle(history: Sequence[Mapping[str, Any]]) -> str:
    """Score-by-cycle chart: start, end and why the loop stopped."""
    first, last = history[0], history[-1]
    best = max(history, key=lambda h: h["overall_score"])
    cycles = len(history) - 1
    return (
        f"The first draft scored {first['overall_score']:.1f} against a threshold of {first['threshold']:.0f}. "
        f"After {cycles} refinement cycle(s) the best version (v{best['report_version']}) scored {best['overall_score']:.1f}; "
        f"the loop {last['stop_reason'] or 'ended'}."
    )


def quality_across_runs(runs: Sequence[Mapping[str, Any]]) -> str:
    """Repeated-runs chart: first-draft and final scores per run and the lessons applied."""
    parts = [
        f"{r['label']} went from {r['initial']:.1f} to {r['final']:.1f} with {len(r.get('lessons_applied') or [])} lesson(s) applied"
        for r in runs
    ]
    change = runs[-1]["initial"] - runs[0]["initial"]
    return "; ".join(parts) + f". The first draft of the last run scored {change:+.1f} points against the first run's."


def workflow(state: AgentState) -> str:
    """Workflow diagram: what ran, how many refinement cycles and the memory traffic."""
    ran = [s.subagent.replace("_agent", "") for s in (state.plan.steps if state.plan else [])]
    skipped = [r.subagent.replace("_agent", "") for r in state.routing_log if not r.selected]
    reflection = getattr(state, "reflection", None) or {}
    stored = [l for l in reflection.get("lessons", []) if l.get("memory_id")]
    cycles = max(len(state.evaluation_history) - 1, 0)
    return (
        f"This run read {len(state.memories_used)} lesson(s) from memory, ran {', '.join(ran) or 'no specialist'}"
        + (f" and skipped {', '.join(skipped)}" if skipped else "")
        + f", went through {cycles} refinement cycle(s), and stored {len(stored)} lesson(s) for later runs."
    )

"""Self-reflection: a structured look back at one run, and the lessons worth keeping for later runs.

The reflection is built in code from what the run recorded: the execution log, the routing log, the
tool-call log, the evidence pack and the evaluation history. Lessons come from explicit rules, so the
same run always yields the same lessons, and each rule says which observation triggered it.

| Trigger                                               | Scope   | Hint                                    |
|-------------------------------------------------------|---------|-----------------------------------------|
| Price analysis ran without news; drivers unexplained   | symbol  | require_subagents: news_agent           |
| A selected specialist failed                           | general | extra_criteria for that specialist      |
| Numeric or citation issues in the first draft          | general | report_guidance                         |
| Raw figures or double negatives in the first draft     | general | report_guidance                         |
| Risks or catalysts missing from the first draft        | general | report_guidance                         |
| Latest annual statements older than nine months        | symbol  | extra_criteria for the earnings agent   |

General lessons never add specialists, so a lesson learned on one symbol cannot reroute another.
"""

from __future__ import annotations

from datetime import date

from .memory import TOPIC_BY_AGENT
from .models import SUBAGENT_EARNINGS, SUBAGENT_MARKET, SUBAGENT_NEWS, AgentState, Status
from .report_evidence import AGENT_LABEL, EvidencePack
from .report_models import EvaluationResult, Lesson, ResearchReport, RunReflection

STALE_STATEMENTS_DAYS = 274  # about nine months after the fiscal year ended


def reflect(state: AgentState, pack: EvidencePack, history: list[EvaluationResult], final: ResearchReport) -> RunReflection:
    """Summarise what worked, what failed, what is still missing, and what to do differently next time."""
    reflection = RunReflection(run_id=state.run_id, symbol=state.symbol, question=state.research_question)

    for entry in state.execution_log:
        label = f"{entry['step_id']} {entry['subagent']}"
        if entry["status"] in (Status.SUCCESS, Status.PARTIAL):
            reflection.successes.append(
                f"{label} finished ({entry['status']}) with {entry['tool_calls']} tool call(s) in {entry['duration_ms'] / 1000:.1f} s."
            )
        else:
            reflection.failures.append(f"{label} {entry['status']}: {entry['message']}")
    for decision in state.routing_log:
        if not decision.selected:
            reflection.skipped_steps.append(f"{decision.subagent} not selected: {decision.reason}")
    for call in state.tool_call_log:
        if call.get("status") != Status.SUCCESS:
            reflection.tool_problems.append(f"{call.get('caller')} called {call.get('tool_name')}: {call.get('error')}")

    reflection.evidence_gaps = [l["text"] for l in pack.limitations if "did not complete" in l["text"]]
    if history:
        first = history[0]
        best = max(history, key=lambda evaluation: evaluation.overall_score)
        reflection.initial_score = first.overall_score
        reflection.final_score = best.overall_score
        reflection.refinement_cycles = len(history) - 1
        reflection.stop_reason = history[-1].stop_reason
        open_keys = {issue.key() for issue in best.issues}
        reflection.issues_fixed = [f"{i.criterion}: {i.message}" for i in first.issues if i.key() not in open_keys]
        reflection.issues_open = [f"{i.criterion}: {i.message}" for i in best.issues]
        reflection.evidence_gaps += [i.message for i in best.issues if i.criterion in ("completeness", "source_recency")]
        strong = [f"{c.label.lower()} {c.score:.0f}" for c in best.criteria if c.score >= 80]
        if strong:
            reflection.successes.append("Criteria at 80 or above in the final report: " + ", ".join(strong) + ".")
        improved = [
            f"{c.label} {first.criterion(c.name).score:.0f} to {c.score:.0f}"
            for c in best.criteria
            if c.score > first.criterion(c.name).score + 0.05
        ]
        reflection.improvements.append(
            f"Overall score went from {first.overall_score:.1f} to {best.overall_score:.1f} "
            f"over {reflection.refinement_cycles} refinement cycle(s); {history[-1].stop_reason}."
        )
        if improved:
            reflection.improvements.append("Criteria that improved: " + "; ".join(improved) + ".")
        refined = sorted({r["section"] for r in final.revisions})
        if refined:
            reflection.improvements.append("Sections refined: " + ", ".join(refined) + ".")
    reflection.lessons = derive_lessons(state, pack, history)
    return reflection


def derive_lessons(state: AgentState, pack: EvidencePack, history: list[EvaluationResult]) -> list[Lesson]:
    """Turn the run's observations into compact lessons with machine-readable hints."""
    lessons: list[Lesson] = []
    symbol = state.symbol
    selected = set(state.selected_subagents)
    all_codes = {issue.code for evaluation in history for issue in evaluation.issues}
    first_codes = {issue.code for issue in history[0].issues} if history else set()

    if "price_drivers_missing" in all_codes and SUBAGENT_NEWS not in selected and pack.completed(SUBAGENT_MARKET):
        lessons.append(
            Lesson(
                lesson=f"Price analysis of {symbol} without recent news left the share-price move unexplained; "
                f"include the news specialist for price questions on {symbol}.",
                scope="symbol",
                symbol=symbol,
                tags=["price"],
                hint={"require_subagents": [SUBAGENT_NEWS]},
                trigger="price_drivers_missing",
            )
        )

    for step in state.plan.steps if state.plan else []:
        if step.status == Status.FAILED:
            label = AGENT_LABEL.get(step.subagent, step.subagent)
            lessons.append(
                Lesson(
                    lesson=f"The {label} specialist failed in an earlier run; if its data source fails again, "
                    "record the coverage gap explicitly.",
                    tags=[TOPIC_BY_AGENT.get(step.subagent, label)],
                    hint={"subagent": step.subagent, "extra_criteria": "If the data source fails, record the coverage gap explicitly"},
                    trigger="specialist_failed",
                )
            )

    if first_codes & {"uncited_fact", "unverified_number", "invalid_citation"}:
        lessons.append(
            Lesson(
                lesson="First drafts misquoted figures or left them uncited; quote every figure exactly as in "
                "the evidence and cite its ID beside it.",
                hint={"report_guidance": "Quote every figure exactly as it appears in the evidence and put its evidence ID right after it."},
                trigger="numeric_or_citation_issues",
            )
        )

    if first_codes & {"raw_figure", "double_negative"}:
        lessons.append(
            Lesson(
                lesson="First drafts printed raw figures or double negatives; write amounts in billions with one decimal "
                "and state gaps as positive numbers after the word that gives their direction.",
                hint={"report_guidance": "Write amounts in billions with one decimal (for example USD 331.8 bn) and percentages "
                      "with one decimal, as the specialist findings do; state a gap as a positive number after the word that "
                      "gives its direction (trailed by 12.3 percentage points)."},
                trigger="readability_issues",
            )
        )

    if first_codes & {"missing_risk", "missing_catalyst"}:
        lessons.append(
            Lesson(
                lesson="First drafts left out specialist risks or catalysts; cover every numbered risk and catalyst with its ID.",
                hint={"report_guidance": "Cover every numbered risk (R...) and catalyst (C...) in its section and tag each statement with its ID."},
                trigger="coverage_omissions",
            )
        )

    fiscal_end = _latest_fiscal_year_end(pack)
    if fiscal_end is not None and (pack.run_date - fiscal_end).days > STALE_STATEMENTS_DAYS:
        lessons.append(
            Lesson(
                lesson=f"{symbol}'s latest annual statements end {fiscal_end.isoformat()}, more than nine months "
                "before the run; state the fiscal-year end and that later quarters are not covered.",
                scope="symbol",
                symbol=symbol,
                tags=["financials"],
                hint={"subagent": SUBAGENT_EARNINGS, "extra_criteria": "State the fiscal-year end date and that later quarters are not covered"},
                trigger="stale_statements",
            )
        )
    return lessons


def _latest_fiscal_year_end(pack: EvidencePack) -> date | None:
    """End date of the latest annual statements the earnings specialist used (SEC filing dates excluded)."""
    dates = []
    for item in pack.evidence.values():
        if item.get("source_agent") == SUBAGENT_EARNINGS and item.get("as_of") and "SEC" not in str(item.get("source", "")):
            try:
                dates.append(date.fromisoformat(str(item["as_of"])[:10]))
            except ValueError:
                continue
    return max(dates) if dates else None


"""Report Generator and the Engineer 3 pipeline entry point.

``run_report_pipeline(state)`` is the hand-off Engineer 1's orchestrator calls after aggregation;
``bootstrap.build_runtime`` finds it by name. It runs these stages in order:

1. Evidence pack (``report_evidence``): numbered risks, catalysts and limitations, citable profile facts.
2. First draft: the LLM writes each section's statements under a JSON schema. Without an LLM, a
   template turns the specialists' findings into statements word for word.
3. Evaluator-Optimizer loop (``optimizer.optimize_report``).
4. Reflection and lessons (``reflection.reflect``), stored in SQLite memory (``memory``).
5. Report and run-log files under ``outputs/`` (``run_outputs``).

Each stage is contained: a failure becomes a warning on the state, and the aggregated evidence stays intact.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from . import memory
from .config import PROJECT_ROOT, Settings
from .llm_client import create_llm_client
from .models import AgentState
from .optimizer import optimize_report
from .reflection import reflect
from .report_evidence import (
    PROFILE_AGENT,
    STATEMENT_SCHEMA,
    WRITABLE_SECTIONS,
    EvidencePack,
    build_evidence_pack,
    clean_statements,
    finalize_report,
    template_statements,
)
from .report_models import (
    FACT,
    SECTION_COMPLETE,
    SECTION_ORDER,
    SECTION_TITLES,
    ReportSection,
    ReportStatement,
    ResearchReport,
)
from .run_outputs import save_run_outputs
from .tool_registry import TOOL_MEMORY_STORAGE

SOURCE = "report_generator"
PROMPT_PATH = PROJECT_ROOT / "prompts" / "report_generator.txt"

DRAFT_STATEMENT_CAP = 8  # hard cap per section; the prompt asks for at most four


# ---- report construction --------------------------------------------------------------------


def build_skeleton(pack: EvidencePack) -> ResearchReport:
    """All eleven sections with their status and note; the company overview is filled from profile evidence."""
    sections = []
    for key in SECTION_ORDER:
        status, note = pack.section_status(key)
        sections.append(ReportSection(key=key, title=SECTION_TITLES[key], status=status, note=note))
    report = ResearchReport(symbol=pack.symbol, question=pack.question, sections=sections)
    report.section("company_overview").statements = company_overview(pack)
    return report


def company_overview(pack: EvidencePack) -> list[ReportStatement]:
    """Profile facts, each citing its PRO evidence item, so the overview is grounded like the rest."""
    statements = []
    for evidence_id, item in pack.evidence.items():
        if item.get("source_agent") != PROFILE_AGENT:
            continue
        text = item["claim"]
        if item.get("value") is not None:  # market capitalisation
            text = f"Market capitalisation was {item.get('unit') or ''} {item['value'] / 1e9:,.1f} bn when the profile was retrieved"
        statements.append(ReportStatement(text=" ".join(text.split()) + ".", kind=FACT, evidence_ids=[evidence_id]))
    return statements


def draft_schema() -> dict[str, Any]:
    """JSON schema of the writer's reply: one list of statements per writable section."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(WRITABLE_SECTIONS),
        "properties": {key: {"type": "array", "items": STATEMENT_SCHEMA} for key in WRITABLE_SECTIONS},
    }


def draft_payload(pack: EvidencePack, report: ResearchReport) -> dict[str, Any]:
    """Compact JSON input for the writer: only what it may cite, plus guidance learned in earlier runs."""
    return {
        "symbol": pack.symbol,
        "company": {k: pack.company.get(k) for k in ("name", "sector", "industry")},
        "question": pack.question,
        "sections_to_write": [k for k in WRITABLE_SECTIONS if report.section(k).status == SECTION_COMPLETE],
        "sections_not_covered": {k: report.section(k).note for k in WRITABLE_SECTIONS if report.section(k).status != SECTION_COMPLETE},
        "evidence": [
            {k: item.get(k) for k in ("id", "claim", "value", "unit", "as_of")}
            for item in pack.evidence.values()
            if item.get("source_agent") != PROFILE_AGENT
        ],
        "findings": [{k: f[k] for k in ("agent", "statement", "evidence_ids")} for f in pack.findings],
        "risks": pack.risks,
        "catalysts": pack.catalysts,
        "limitations": [{"id": l["id"], "text": l["text"]} for l in pack.limitations],
        "data_periods": pack.data_periods,
        "news_summary": pack.news_summary,
        "writing_guidance": [text for _, text in pack.guidance],
    }


def generate_report(pack: EvidencePack, llm: Any = None, settings: Settings | None = None) -> tuple[ResearchReport, list[str]]:
    """Write the first draft (version 0). Returns the report and any warnings about how it was written."""
    report = build_skeleton(pack)
    notes: list[str] = []
    writable = [k for k in WRITABLE_SECTIONS if report.section(k).status == SECTION_COMPLETE]
    drafted = False
    if llm is not None and getattr(llm, "available", False):
        try:
            messages = [
                {"role": "system", "content": PROMPT_PATH.read_text(encoding="utf-8")},
                {"role": "user", "content": json.dumps(draft_payload(pack, report), default=str)},
            ]
            raw = llm.chat_json(messages, draft_schema(), schema_name="research_report_draft")
            removed = 0
            for key in writable:
                statements, dropped = clean_statements(raw.get(key), pack, DRAFT_STATEMENT_CAP)
                report.section(key).statements = statements
                removed += dropped
            if removed:
                notes.append(f"Removed {removed} citation(s) to IDs that are not in the evidence pack from the first draft.")
            report.generated_by = "llm"
            report.guidance_used = [memory_id for memory_id, _ in pack.guidance]
            drafted = True
        except Exception as exc:  # any LLM failure falls back to the template, never to an empty report
            notes.append(f"LLM draft failed, template used instead: {type(exc).__name__}: {exc}")
    if not drafted:
        for key in writable:
            report.section(key).statements = template_statements(pack, key)
        report.generated_by = "template"
    return finalize_report(report, pack), notes


# ---- pipeline ---------------------------------------------------------------------------------


def run_report_pipeline(
    state: AgentState,
    *,
    llm: Any = None,
    settings: Settings | None = None,
    tools: Any = None,
    save_outputs: bool = True,
) -> AgentState:
    """Report, evaluator-optimizer, reflection, memory and output files for one orchestrated run.

    ``llm``, ``settings`` and ``tools`` default to a fresh client, default settings and direct memory
    calls, so the orchestrator can call this with the state alone. When ``tools`` (the tool registry) is
    given, lessons are stored through the ``memory_storage`` tool and appear in the tool-call log.
    """
    settings = settings or Settings()
    if state.aggregated is None:
        state.warn(SOURCE, "No aggregated evidence was handed over; the report pipeline did not run.")
        return state
    llm = llm if llm is not None else create_llm_client()

    pack = build_evidence_pack(state, settings)
    draft, notes = generate_report(pack, llm, settings)
    for note in notes:
        state.warn(SOURCE, note)
    state.draft_report = draft.to_dict()

    try:
        final, history = optimize_report(draft, pack, llm, settings)
    except Exception as exc:  # keep the draft rather than lose the run
        state.warn(SOURCE, f"Evaluator-optimizer failed, first draft kept: {type(exc).__name__}: {exc}", severity="error")
        final, history = draft, []
    state.final_report = final.to_dict()
    state.evaluation_history = [evaluation.to_dict() for evaluation in history]

    try:
        reflection = reflect(state, pack, history, final)
        _remember(state, reflection, history, final, llm, tools)
        state.reflection = reflection.to_dict()  # shown in the notebook and written to the run log
    except Exception as exc:  # reflection and memory are learning aids; the report stands without them
        state.warn(SOURCE, f"Reflection or memory update failed: {type(exc).__name__}: {exc}", severity="error")

    if save_outputs:
        try:
            state.output_files = save_run_outputs(state, settings)
        except Exception as exc:
            state.warn(SOURCE, f"Could not write output files: {type(exc).__name__}: {exc}")
    return state


def make_report_pipeline(
    llm: Any = None, settings: Settings | None = None, tools: Any = None, save_outputs: bool = True
) -> Callable[[AgentState], AgentState]:
    """Bind the runtime's LLM, settings and tool registry, giving the ``(state) -> state`` hook the orchestrator expects."""

    def pipeline(state: AgentState) -> AgentState:
        return run_report_pipeline(state, llm=llm, settings=settings, tools=tools, save_outputs=save_outputs)

    return pipeline


def _remember(state: AgentState, reflection: Any, history: list[Any], final: ResearchReport, llm: Any, tools: Any) -> None:
    """Store the reflection's lessons and a run record. Memory guides later plans; it is never evidence."""
    for lesson in reflection.lessons:
        params = {
            "lesson": lesson.lesson,
            "scope": lesson.scope,
            "symbol": lesson.symbol,
            "tags": lesson.tags,
            "hint": lesson.hint,
            "trigger": lesson.trigger,
            "source_run_id": state.run_id,
        }
        if tools is not None and tools.has(TOOL_MEMORY_STORAGE):
            result = tools.session("reflection", [TOOL_MEMORY_STORAGE], state).call(TOOL_MEMORY_STORAGE, **params)
            lesson.memory_id = result.data.get("memory_id", "") if result.ok and isinstance(result.data, dict) else ""
        else:
            lesson.memory_id = memory.store_lesson(**params)["memory_id"]

    applied = list(dict.fromkeys((state.plan.lessons_applied if state.plan else []) + final.guidance_used))
    memory.record_run(
        run_id=state.run_id,
        symbol=state.symbol,
        question=state.research_question,
        scope=state.plan.scope if state.plan else "",
        selected_subagents=list(state.selected_subagents),
        initial_score=history[0].overall_score if history else None,
        final_score=max((h.overall_score for h in history), default=None),
        refinement_cycles=max(len(history) - 1, 0),
        stop_reason=history[-1].stop_reason if history else "",
        llm=f"{getattr(llm, 'provider', 'none')}:{getattr(llm, 'model', '')}" if llm is not None else "none",
        lessons_retrieved=[str(m.get("memory_id")) for m in state.memories_used if m.get("memory_id")],
        lessons_applied=applied,
        reflection=reflection.to_dict(),
        started_at=state.started_at,
    )

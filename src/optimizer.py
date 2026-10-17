"""Optimizer: rewrites only the weak sections of a report, then the evaluator scores it again.

The loop (``optimize_report``) is the Evaluator-Optimizer workflow:

1. Evaluate the first draft.
2. Stop if it is accepted. Otherwise refine every section that has an actionable issue, worst first.
3. Evaluate the new version. Stop when it is accepted, when ``Settings.max_refinement_cycles`` is
   reached, or when a cycle gains less than ``MIN_GAIN`` points; otherwise repeat.

The best-scoring version becomes the final report, so a refinement that makes things worse is never
kept, and the stop reason records why the loop ended.

Each weak section is rewritten by the LLM, which receives that section's issues and fixes and the IDs
it may cite. Statements that passed every check are locked: the LLM is asked to keep them, and a locked
statement is put back if the rewrite drops what it cites.
Without an LLM, a rule-based fallback applies the corrective actions directly.
"""

from __future__ import annotations

import json
from typing import Any

from .config import PROJECT_ROOT, Settings
from .evaluator import evaluate_report
from .models import SUBAGENT_MARKET, SUBAGENT_NEWS
from .report_evidence import (
    PROFILE_AGENT,
    SECTION_AGENT,
    STATEMENT_SCHEMA,
    WRITABLE_SECTIONS,
    EvidencePack,
    clean_statements,
    disclosure,
    finalize_report,
    finding_statement,
    template_statements,
)
from .report_models import (
    DECISION_ACCEPT,
    INTERPRETATION,
    SECTION_COMPLETE,
    STOP_ACCEPTED,
    STOP_LOW_GAIN,
    STOP_MAX_CYCLES,
    STOP_NOTHING_TO_REFINE,
    EvaluationResult,
    ReportIssue,
    ReportSection,
    ReportStatement,
    ResearchReport,
)

PROMPT_PATH = PROJECT_ROOT / "prompts" / "optimizer.txt"
MIN_GAIN = 2.0  # a refinement cycle must add at least this many points to justify another one
DEFAULT_CAP = 10
SECTION_CAPS = {"confidence_and_limitations": 14, "risks": 12, "catalysts": 12}
REVISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["statements"],
    "properties": {"statements": {"type": "array", "items": STATEMENT_SCHEMA}},
}


def optimize_report(
    draft: ResearchReport,
    pack: EvidencePack,
    llm: Any = None,
    settings: Settings | None = None,
) -> tuple[ResearchReport, list[EvaluationResult]]:
    """Run the evaluate-refine loop. Returns the best version and the evaluation of every version."""
    settings = settings or Settings()
    current = draft
    evaluation = evaluate_report(current, pack, llm, settings, cycle=0)
    history = [evaluation]
    best, best_score = current, evaluation.overall_score
    while True:
        if evaluation.decision == DECISION_ACCEPT:
            evaluation.stop_reason = STOP_ACCEPTED
            break
        if evaluation.cycle >= settings.max_refinement_cycles:
            evaluation.stop_reason = STOP_MAX_CYCLES
            break
        if not [key for key in evaluation.weak_sections if key in WRITABLE_SECTIONS]:
            evaluation.stop_reason = STOP_NOTHING_TO_REFINE
            break
        refined = refine_report(current, evaluation, pack, llm)
        previous = evaluation
        evaluation = evaluate_report(refined, pack, llm, settings, cycle=previous.cycle + 1)
        history.append(evaluation)
        if evaluation.overall_score > best_score:
            best, best_score = refined, evaluation.overall_score
        current = refined
        if evaluation.decision != DECISION_ACCEPT and evaluation.overall_score - previous.overall_score < MIN_GAIN:
            evaluation.stop_reason = STOP_LOW_GAIN
            break
    return best, history


def refine_report(report: ResearchReport, evaluation: EvaluationResult, pack: EvidencePack, llm: Any = None) -> ResearchReport:
    """A new version in which every weak section has been revised against its issues."""
    refined = report.copy()
    refined.version = report.version + 1
    for key in evaluation.weak_sections:
        if key not in WRITABLE_SECTIONS:
            continue
        section = refined.section(key)
        issues = [issue for issue in evaluation.issues if issue.section == key and issue.actionable]
        if section.status != SECTION_COMPLETE or not issues:
            continue
        method, statements = "rules", None
        if llm is not None and getattr(llm, "available", False):
            try:
                statements, method = llm_revision(section, issues, pack, llm), "llm"
            except Exception as exc:  # the rule-based fallback keeps the loop going
                method = f"rules (LLM revision failed: {type(exc).__name__})"
        if statements is None:
            statements = rule_based_revision(section, issues, pack)
        section.statements = statements
        refined.revisions.append({"version": refined.version, "section": key, "method": method, "issues": len(issues)})
    return finalize_report(refined, pack)


# ---- LLM revision ---------------------------------------------------------------------------------


def llm_revision(section: ReportSection, issues: list[ReportIssue], pack: EvidencePack, llm: Any) -> list[ReportStatement]:
    """Ask the LLM to rewrite one section; locked statements survive even if the reply drops them."""
    cap = SECTION_CAPS.get(section.key, DEFAULT_CAP)
    flagged = {issue.statement_index for issue in issues if issue.statement_index is not None}
    locked = [index for index in range(len(section.statements)) if index not in flagged]
    payload = {
        "section": {"key": section.key, "title": section.title},
        "max_statements": cap,
        "statements": [
            {**statement.to_dict(), "index": index, "locked": index in locked}
            for index, statement in enumerate(section.statements)
        ],
        "issues": [{"problem": issue.message, "fix": issue.corrective_action} for issue in issues],
        **allowed_material(section.key, pack),
        "writing_guidance": [text for _, text in pack.guidance],
    }
    messages = [
        {"role": "system", "content": PROMPT_PATH.read_text(encoding="utf-8")},
        {"role": "user", "content": json.dumps(payload, default=str)},
    ]
    raw = llm.chat_json(messages, REVISION_SCHEMA, schema_name="section_revision")
    statements, _ = clean_statements(raw.get("statements"), pack, cap)
    # Locked statements may be rephrased or merged, but what they cite must survive. A statement comes
    # back word for word only when its evidence, risk, catalyst or limitation IDs vanished from the section;
    # putting back every reworded original would duplicate it.
    cited = set().union(*(_refs(statement) for statement in statements)) if statements else set()
    present = {statement.text.lower() for statement in statements}
    for index in locked:
        statement = section.statements[index]
        if statement.text.lower() not in present and not _refs(statement) <= cited:
            statements.append(statement)
    return statements


def _refs(statement: ReportStatement) -> set[str]:
    """Every ID a statement cites."""
    return set(statement.evidence_ids + statement.risk_ids + statement.catalyst_ids + statement.limitation_ids)


def allowed_material(key: str, pack: EvidencePack) -> dict[str, Any]:
    """What the optimizer may cite when it rewrites a given section (kept small to stay on topic)."""

    def evidence(ids: list[str]) -> list[dict[str, Any]]:
        return [{k: pack.evidence[i].get(k) for k in ("id", "claim", "value", "unit", "as_of")} for i in ids]

    specialist_ids = [eid for eid, item in pack.evidence.items() if item.get("source_agent") != PROFILE_AGENT]
    material: dict[str, Any] = {"evidence": [], "findings": [], "risks": [], "catalysts": [], "limitations": [], "data_period": ""}
    if key in SECTION_AGENT:
        agent = SECTION_AGENT[key]
        ids = pack.evidence_for(agent, macro=False)
        if key == "market_performance":  # news evidence lets the rewrite explain the price move
            ids += pack.evidence_for(SUBAGENT_NEWS)
            material["catalysts"] = pack.catalysts
        material.update(evidence=evidence(ids), findings=pack.findings_for(agent), data_period=pack.data_periods.get(agent, ""))
    elif key == "macro_context":
        material.update(
            evidence=evidence(pack.evidence_for(SUBAGENT_MARKET, macro=True)),
            findings=pack.findings_for(SUBAGENT_MARKET, macro=True),
            data_period=pack.data_periods.get("macro", ""),
        )
    elif key == "executive_summary":
        material.update(evidence=evidence(specialist_ids), findings=pack.findings, risks=pack.risks, catalysts=pack.catalysts)
    elif key == "risks":
        material.update(evidence=evidence(specialist_ids), risks=pack.risks)
    elif key == "catalysts":
        material.update(evidence=evidence(pack.evidence_for(SUBAGENT_NEWS)), catalysts=pack.catalysts)
    else:  # conflicting or missing evidence, confidence and limitations: disclosures need no evidence
        material.update(limitations=pack.limitations, data_period=" ".join(pack.data_periods.values()))
        material["warnings"] = pack.warnings
    return material


# ---- rule-based fallback ----------------------------------------------------------------------------


def rule_based_revision(section: ReportSection, issues: list[ReportIssue], pack: EvidencePack) -> list[ReportStatement]:
    """Apply each issue's corrective action directly, without an LLM."""
    statements = list(section.statements)
    drop: set[int] = set()
    additions: list[ReportStatement] = []
    items = {item["id"]: item for item in pack.risks + pack.catalysts + pack.limitations}
    agent = SECTION_AGENT.get(section.key)

    for issue in issues:
        code, index = issue.code, issue.statement_index
        if code in ("uncited_fact", "unverified_number", "overclaim", "duplicate_statement", "raw_figure", "double_negative") and index is not None:
            drop.add(index)
            if code in ("unverified_number", "raw_figure", "double_negative"):
                # Replace the statement with the specialist's own finding, which states the figures correctly and readably.
                additions += [finding_statement(f) for f in pack.findings if set(f["evidence_ids"]) & set(issue.refs)][:1]
        elif code == "invalid_citation" and index is not None:
            statement = statements[index]
            statement.evidence_ids = [eid for eid in statement.evidence_ids if eid in pack.evidence]
        elif code in ("missing_findings", "no_facts"):
            source = [f for f in pack.findings if set(f["evidence_ids"]) & set(issue.refs)] if issue.refs else []
            if not source and agent:
                source = pack.findings_for(agent)[:4]
            additions += [finding_statement(f) for f in source]
        elif code in ("missing_risk", "missing_catalyst"):
            for ref in issue.refs:
                item = items.get(ref)
                if item:
                    id_field = "risk_ids" if code == "missing_risk" else "catalyst_ids"
                    additions.append(ReportStatement(text=item["text"], kind=INTERPRETATION, evidence_ids=item["evidence_ids"], **{id_field: [ref]}))
        elif code in ("missing_limitation", "gap_unexplained"):
            additions += [disclosure(items[ref]["text"], [ref]) for ref in issue.refs if ref in items]
        elif code == "price_drivers_missing":
            additions.append(disclosure("The drivers of the price move were not analysed, because news analysis was not part of this run."))
        elif code == "price_drivers_unlinked":
            news_ids = pack.evidence_for(SUBAGENT_NEWS)[:2]
            if news_ids:
                events = "; ".join(pack.evidence[eid]["claim"].rstrip(".") for eid in news_ids)
                additions.append(
                    ReportStatement(text=f"Recent company events that may bear on the share price include: {events}.", kind=INTERPRETATION, evidence_ids=news_ids)
                )
        elif code in ("data_date_missing", "stale_evidence"):
            period = pack.data_periods.get("macro" if section.key == "macro_context" else agent or "", "")
            if not period and issue.refs and issue.refs[0] in pack.evidence:
                period = f"{issue.refs[0]} is dated {pack.evidence[issue.refs[0]].get('as_of')}."
            if period:
                additions.append(disclosure(period))
        elif code == "empty_section":
            additions += template_statements(pack, section.key)
        elif code == "no_confidence":
            additions.append(disclosure("Confidence is stated with its reasons at the top of this section."))

    kept = [statement for index, statement in enumerate(statements) if index not in drop]
    present = {statement.text.lower() for statement in kept}
    for statement in additions:
        if statement.text and statement.text.lower() not in present:
            kept.append(statement)
            present.add(statement.text.lower())
    return kept[: SECTION_CAPS.get(section.key, DEFAULT_CAP)]


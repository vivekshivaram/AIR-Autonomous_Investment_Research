"""Orchestrator: plans, executes selected subagents in dependency order, and aggregates evidence."""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable

from .models import (
    ALL_SUBAGENTS,
    AgentState,
    EvidenceItem,
    MemoryItem,
    ResearchStep,
    SchemaError,
    Status,
    SubagentResult,
    utc_now_iso,
)
from .planner import Planner
from .tool_registry import TOOL_COMPANY_PROFILE, SubagentRegistry, ToolRegistry

MemoryRetriever = Callable[[str, str], Iterable[Any]]
ReportPipeline = Callable[[AgentState], "AgentState | None"]


def topological_order(steps: list[ResearchStep]) -> list[ResearchStep]:
    """Dependency order that preserves plan order among independent steps."""
    index = {s.step_id: i for i, s in enumerate(steps)}
    remaining = {s.step_id: set(s.depends_on) for s in steps}
    ordered: list[ResearchStep] = []
    while remaining:
        ready = sorted((sid for sid, deps in remaining.items() if not deps), key=index.__getitem__)
        if not ready:
            raise ValueError("dependency cycle in plan")
        sid = ready[0]
        ordered.append(steps[index[sid]])
        del remaining[sid]
        for deps in remaining.values():
            deps.discard(sid)
    return ordered


class Orchestrator:
    """Coordinates the Planner, subagents and tools around one shared AgentState."""

    def __init__(
        self,
        planner: Planner,
        subagents: SubagentRegistry,
        tools: ToolRegistry,
        memory_retriever: MemoryRetriever | None = None,
        report_pipeline: ReportPipeline | None = None,
    ):
        self.planner = planner
        self.subagents = subagents
        self.tools = tools
        self.memory_retriever = memory_retriever
        self.report_pipeline = report_pipeline

    # ---- public API -------------------------------------------------------------

    def plan(self, symbol: str, question: str) -> AgentState:
        """Retrieve lessons, build the plan and routing log. No subagent or data tool runs."""
        state = AgentState(symbol=symbol.strip().upper(), research_question=question)
        memories = self._retrieve_memories(state)
        planning = self.planner.create_plan(symbol, question, memories)
        state.symbol = planning.plan.symbol
        state.research_question = planning.plan.question
        state.plan = planning.plan
        state.routing_log = planning.routing
        state.selected_subagents = planning.plan.selected_subagents
        state.warnings.extend(planning.issues)
        return state

    def run(self, symbol: str, question: str) -> AgentState:
        """Full workflow: plan, execute, aggregate, then hand off to the report pipeline."""
        state = self.plan(symbol, question)
        self._fetch_profile(state)
        self._execute_plan(state)
        state.aggregated = aggregate_results(state)
        state.evidence = state.aggregated["evidence"]
        return self._hand_off(state)

    # ---- steps ------------------------------------------------------------------

    def _retrieve_memories(self, state: AgentState) -> list[MemoryItem]:
        if self.memory_retriever is None:
            return []
        try:
            items = [MemoryItem.from_raw(m) for m in self.memory_retriever(state.symbol, state.research_question)]
        except Exception as exc:  # memory is optional guidance; planning proceeds without it
            state.warn("orchestrator", f"Memory retrieval failed: {exc}")
            return []
        state.memories_used = [m.to_dict() for m in items]
        return items

    def _fetch_profile(self, state: AgentState) -> None:
        """Company profile is shared context (e.g. company name for news) and non-critical."""
        if not self.tools.has(TOOL_COMPANY_PROFILE):
            return
        session = self.tools.session("orchestrator", [TOOL_COMPANY_PROFILE], state)
        result = session.call(TOOL_COMPANY_PROFILE, symbol=state.symbol)
        if result.ok and isinstance(result.data, dict):
            state.company_profile = result.data

    def _execute_plan(self, state: AgentState) -> None:
        assert state.plan is not None
        plan = state.plan
        decisions = {r.step_id: r for r in state.routing_log if r.step_id}
        for step in topological_order(plan.steps):
            self._execute_step(state, step)
            if step.step_id in decisions:
                decisions[step.step_id].execution_status = step.status
        for decision in state.routing_log:
            if not decision.selected:
                decision.execution_status = Status.SKIPPED

    def _execute_step(self, state: AgentState, step: ResearchStep) -> None:
        assert state.plan is not None
        spec = self.subagents.get(step.subagent)
        started_at = utc_now_iso()
        t0 = time.perf_counter()
        calls_before = len(state.tool_call_log)
        message = ""

        failed_deps = [d for d in step.depends_on if state.plan.get_step(d).status in Status.UNSATISFIED]
        if failed_deps:
            state.warn(step.subagent, f"Step {step.step_id} dependencies unsatisfied: {failed_deps}")

        if failed_deps and step.blocking:
            step.status, message = Status.SKIPPED, f"blocked by {failed_deps}"
        elif spec.fn is None:
            step.status, message = Status.SKIPPED, "subagent implementation not registered"
            state.warn(step.subagent, f"No implementation registered for {step.subagent}; step skipped")
        else:
            step.status = Status.RUNNING
            session = self.tools.session(step.subagent, step.tools, state)
            try:
                result = SubagentResult.from_raw(spec.fn(state, session, step), step.subagent)
            except Exception as exc:  # one failing subagent must not stop the others
                step.status = Status.FAILED
                message = f"{type(exc).__name__}: {exc}"
                state.warn(step.subagent, f"Subagent failed: {message}", severity="error", step_id=step.step_id)
            else:
                result.duration_ms = round((time.perf_counter() - t0) * 1000, 2)
                state.set_result(result)
                step.status = result.status
                step.result_summary = result.summary
                message = result.summary
                if result.status == Status.FAILED:
                    state.warn(step.subagent, f"Subagent reported failure: {result.summary}", severity="error")

        state.execution_log.append(
            {
                "step_id": step.step_id,
                "subagent": step.subagent,
                "status": step.status,
                "started_at": started_at,
                "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
                "tools_allowed": list(step.tools),
                "tool_calls": len(state.tool_call_log) - calls_before,
                "message": message,
            }
        )

    def _hand_off(self, state: AgentState) -> AgentState:
        if self.report_pipeline is None:
            return state
        try:
            returned = self.report_pipeline(state)
        except Exception as exc:  # the aggregated evidence stays available even if reporting fails
            state.warn("orchestrator", f"Report pipeline failed: {type(exc).__name__}: {exc}", severity="error")
            return state
        return returned if isinstance(returned, AgentState) else state


def _tagged(items: list[Any], agent: str) -> list[dict[str, Any]]:
    tagged = []
    for item in items:
        entry = dict(item) if isinstance(item, dict) else {"text": str(item)}
        entry.setdefault("agent", agent)
        tagged.append(entry)
    return tagged


def aggregate_results(state: AgentState) -> dict[str, Any]:
    """Merge specialist outputs into the single structure handed to the Report Generator."""
    agents: dict[str, Any] = {}
    findings: list[dict[str, Any]] = []
    risks: list[dict[str, Any]] = []
    limitations: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for name in ALL_SUBAGENTS:
        result = state.get_result(name)
        if result is None:
            continue
        agents[name] = {"status": result["status"], "summary": result["summary"]}
        findings += _tagged(result["findings"], name)
        risks += _tagged(result["risks"], name)
        limitations += _tagged(result["limitations"], name)
        for raw in result["evidence"]:
            try:
                item = EvidenceItem.from_raw(raw, name)
            except SchemaError as exc:
                state.warn(name, f"Dropped malformed evidence item: {exc}")
                continue
            key = (item.claim.strip().lower(), item.source)
            if key in seen:
                continue
            seen.add(key)
            item.evidence_id = f"EV-{len(evidence) + 1:03d}"
            evidence.append(item.to_dict())

    # Selected-but-unfinished work is surfaced as a limitation so reports never imply coverage.
    coverage: dict[str, str] = {}
    if state.plan:
        for step in state.plan.steps:
            coverage[step.subagent] = step.status
            if step.status in (Status.FAILED, Status.SKIPPED):
                limitations.append(
                    {"agent": step.subagent, "text": f"{step.subagent} did not complete ({step.status}); no analysis available."}
                )

    return {
        "symbol": state.symbol,
        "question": state.research_question,
        "company_profile": state.company_profile,
        "agents": agents,
        "coverage": coverage,
        "findings": findings,
        "risks": risks,
        "limitations": limitations,
        "evidence": evidence,
        "warnings": [w.to_dict() for w in state.warnings],
        "memories_used": state.memories_used,
        "aggregated_at": utc_now_iso(),
    }

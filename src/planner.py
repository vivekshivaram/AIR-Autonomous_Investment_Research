"""Planner and Router.

Routing is part of planning (no separate Router agent). The Planner asks the configured LLM for a
structured plan, validates it, and falls back to a deterministic keyword router when the
LLM is unavailable or returns an invalid plan, so a run never fails on planning alone.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

from .config import PROJECT_ROOT, Settings
from .models import (
    SUBAGENT_EARNINGS,
    SUBAGENT_MARKET,
    SUBAGENT_NEWS,
    Issue,
    MemoryItem,
    ResearchPlan,
    ResearchStep,
    RoutingDecision,
    validate_symbol,
)
from .tool_registry import TOOL_MACRO, TOOL_SEC_FILINGS, SubagentRegistry, ToolRegistry

PROMPT_PATH = PROJECT_ROOT / "prompts" / "planner.txt"
MAX_LESSONS = 5

_COMPREHENSIVE_RE = re.compile(
    r"\b(comprehensive|full (analysis|research|report)|investment research|complete (analysis|research)|deep dive|"
    r"investment thesis|everything|end-to-end|holistic|due diligence|should i (invest|buy))",
    re.I,
)
_MACRO_RE = re.compile(
    r"\b(macro\w*|interest rates?|inflation|oil|natural gas|treasury|yields?|fed|federal reserve|"
    r"unemployment|econom\w+|gdp)\b",
    re.I,
)
_FILING_RE = re.compile(r"\b(filings?|10-k|10-q|8-k|sec|edgar)\b", re.I)
_SEQUENTIAL_RE = re.compile(
    r"\b(resulting|after (the )?earnings|following (the )?earnings|reaction|in response|post-earnings|after the report)",
    re.I,
)

# Templates keep rule-based plans concrete and checkable.
_STEP_TEMPLATES: dict[str, tuple[str, str]] = {
    SUBAGENT_MARKET: (
        "Analyze {symbol} price performance against the benchmark: returns, rolling volatility and major trends",
        "Returns, benchmark comparison and rolling volatility computed with chart-ready data and data limitations noted",
    ),
    SUBAGENT_EARNINGS: (
        "Analyze {symbol} revenue, profit, margin and cash-flow trends from financial statements",
        "Revenue, profit, margin and cash-flow trends reported with strengths, risks and statement coverage",
    ),
    SUBAGENT_NEWS: (
        "Summarize recent {symbol} news through the five-stage prompt chain: events, catalysts and risks",
        "All five chain stages completed with source-backed events, catalysts and risks",
    ),
}

_SKIP_REASONS = {
    SUBAGENT_MARKET: "The question does not ask about price performance, returns, volatility or trends.",
    SUBAGENT_EARNINGS: "The question does not ask about revenue, profit, margins, cash flow or financial statements.",
    SUBAGENT_NEWS: "The question does not ask about recent news, events or announcements.",
}


class PlanValidationError(ValueError):
    """The plan violates structural rules; carries every detected problem."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


class SupportsJSON(Protocol):
    """What the planner needs from an LLM client: an ``available`` flag and schema-constrained ``chat_json``."""

    available: bool

    def chat_json(self, messages: list[dict[str, str]], json_schema: dict[str, Any], **kwargs: Any) -> dict[str, Any]: ...


@dataclass
class PlanningResult:
    """The plan, a routing decision for every subagent (selected or skipped) and any issues raised while planning."""

    plan: ResearchPlan
    routing: list[RoutingDecision]
    issues: list[Issue] = field(default_factory=list)


@dataclass
class QuestionProfile:
    """Heuristic interpretation of the research question."""

    scope: str
    matches: dict[str, list[str]]
    macro: bool
    filings: bool
    sequential: bool
    broad_fallback: bool


def normalize_question(question: str, max_chars: int) -> str:
    """Collapse whitespace and enforce the length limit; an empty or overlong question raises ValueError."""
    cleaned = " ".join((question or "").split())
    if not cleaned:
        raise ValueError("Research question must not be empty")
    if len(cleaned) > max_chars:
        raise ValueError(f"Research question exceeds {max_chars} characters")
    return cleaned


def interpret_question(question: str, subagents: SubagentRegistry) -> QuestionProfile:
    """Find which specialist domains the question touches."""
    q = question.lower()
    matches: dict[str, list[str]] = {}
    for spec in subagents.specs():
        hits = [kw for kw in spec.keywords if re.search(r"\b" + re.escape(kw), q)]
        if hits:
            matches[spec.name] = hits

    macro = bool(_MACRO_RE.search(q))
    filings = bool(_FILING_RE.search(q))
    # A macro or filing request implies the specialist that owns the optional tool.
    if macro and subagents.has(SUBAGENT_MARKET):
        matches.setdefault(SUBAGENT_MARKET, []).append("macro context")
    if filings and subagents.has(SUBAGENT_EARNINGS):
        matches.setdefault(SUBAGENT_EARNINGS, []).append("filings")

    comprehensive = bool(_COMPREHENSIVE_RE.search(q))
    broad_fallback = not comprehensive and not matches
    scope = "comprehensive" if (comprehensive or broad_fallback) else "narrow"
    sequential = (
        bool(_SEQUENTIAL_RE.search(q)) and SUBAGENT_EARNINGS in matches and SUBAGENT_MARKET in matches
    )
    return QuestionProfile(scope, matches, macro, filings, sequential, broad_fallback)


def validate_plan(plan: ResearchPlan, subagents: SubagentRegistry, max_steps: int) -> list[str]:
    """Return every structural problem found (empty list means valid)."""
    problems: list[str] = []
    if not plan.steps:
        problems.append("plan has no steps")
    if len(plan.steps) > max_steps:
        problems.append(f"plan has {len(plan.steps)} steps, maximum is {max_steps}")

    ids = [s.step_id for s in plan.steps]
    if len(set(ids)) != len(ids):
        problems.append("duplicate step_id values")

    for step in plan.steps:
        if not subagents.has(step.subagent):
            problems.append(f"{step.step_id}: unknown subagent {step.subagent!r}")
            continue
        allowed = set(subagents.get(step.subagent).allowed_tools())
        bad_tools = [t for t in step.tools if t not in allowed]
        if bad_tools:
            problems.append(f"{step.step_id}: tools {bad_tools} not allowed for {step.subagent}")
        if not step.completion_criteria.strip():
            problems.append(f"{step.step_id}: missing completion criteria")
        if not step.description.strip():
            problems.append(f"{step.step_id}: missing description")
        for dep in step.depends_on:
            if dep not in ids:
                problems.append(f"{step.step_id}: depends on unknown step {dep!r}")
            elif dep == step.step_id:
                problems.append(f"{step.step_id}: depends on itself")

    if not problems and _has_cycle(plan.steps):
        problems.append("dependency cycle detected")
    return problems


def _has_cycle(steps: list[ResearchStep]) -> bool:
    deps = {s.step_id: list(s.depends_on) for s in steps}
    state: dict[str, int] = {}

    def visit(node: str) -> bool:
        if state.get(node) == 1:
            return True
        if state.get(node) == 2:
            return False
        state[node] = 1
        if any(visit(d) for d in deps.get(node, [])):
            return True
        state[node] = 2
        return False

    return any(visit(n) for n in deps)


class Planner:
    """Builds a validated ResearchPlan plus routing decisions for one request."""

    def __init__(
        self,
        subagents: SubagentRegistry,
        tools: ToolRegistry,
        llm: SupportsJSON | None = None,
        settings: Settings | None = None,
        prompt_path: Path = PROMPT_PATH,
    ):
        self.subagents = subagents
        self.tools = tools
        self.llm = llm
        self.settings = settings or Settings()
        self.prompt_path = prompt_path

    # ---- public API -------------------------------------------------------------

    def create_plan(self, symbol: str, question: str, memories: Iterable[Any] = ()) -> PlanningResult:
        """Plan one request and record a routing decision for every subagent.

        The LLM plans first and keyword routing takes over on any failure; lesson hints are then applied
        and the result is validated. Raises PlanValidationError when the final plan breaks a structural rule.
        """
        symbol = validate_symbol(symbol)
        question = normalize_question(question, self.settings.max_question_chars)
        lessons = self._select_lessons(memories)
        issues: list[Issue] = []

        result: PlanningResult | None = None
        if self.llm is not None and getattr(self.llm, "available", False):
            try:
                result = self._llm_plan(symbol, question, lessons)
            except Exception as exc:  # any LLM-path failure must degrade to rule-based, never abort the run
                issues.append(Issue("planner", f"LLM planning failed, using rule-based planner: {exc}"))
        else:
            issues.append(Issue("planner", "LLM unavailable (provider credentials not configured); using rule-based planner", "warning"))

        if result is None:
            result = self._rule_based_plan(symbol, question)

        self._apply_lessons(result, lessons)
        self._complete_routing(result)

        problems = validate_plan(result.plan, self.subagents, self.settings.max_plan_steps)
        if problems:
            raise PlanValidationError(problems)
        result.issues = issues + result.issues
        return result

    # ---- LLM path ---------------------------------------------------------------

    def _plan_schema(self) -> dict[str, Any]:
        agent_names = self.subagents.names()
        optional = sorted({t for s in self.subagents.specs() for t in s.optional_tools})
        extra_item: dict[str, Any] = {"type": "string"}
        if optional:
            extra_item["enum"] = optional
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["intent", "scope", "macro_context_relevant", "rationale", "steps", "skipped", "lessons_applied"],
            "properties": {
                "intent": {"type": "string"},
                "scope": {"type": "string", "enum": ["narrow", "comprehensive"]},
                "macro_context_relevant": {"type": "boolean"},
                "rationale": {"type": "string"},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "step_id", "description", "subagent", "depends_on", "extra_tools",
                            "completion_criteria", "reason", "confidence",
                        ],
                        "properties": {
                            "step_id": {"type": "string"},
                            "description": {"type": "string"},
                            "subagent": {"type": "string", "enum": agent_names},
                            "depends_on": {"type": "array", "items": {"type": "string"}},
                            "extra_tools": {"type": "array", "items": extra_item},
                            "completion_criteria": {"type": "string"},
                            "reason": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                    },
                },
                "skipped": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["subagent", "reason"],
                        "properties": {
                            "subagent": {"type": "string", "enum": agent_names},
                            "reason": {"type": "string"},
                        },
                    },
                },
                "lessons_applied": {"type": "array", "items": {"type": "string"}},
            },
        }

    def _llm_plan(self, symbol: str, question: str, lessons: list[tuple[str, MemoryItem]]) -> PlanningResult:
        payload = {
            "symbol": symbol,
            "question": question,
            "max_steps": self.settings.max_plan_steps,
            "subagents": [
                {
                    "name": s.name,
                    "description": s.description,
                    "required_tools": list(s.required_tools),
                    "optional_tools": list(s.optional_tools),
                }
                for s in self.subagents.specs()
            ],
            "lessons": [{"id": lid, "lesson": m.lesson} for lid, m in lessons],
        }
        messages = [
            {"role": "system", "content": self.prompt_path.read_text(encoding="utf-8")},
            {"role": "user", "content": json.dumps(payload)},
        ]
        raw = self.llm.chat_json(messages, self._plan_schema(), schema_name="research_plan")  # type: ignore[union-attr]

        steps: list[ResearchStep] = []
        routing: list[RoutingDecision] = []
        for item in raw["steps"]:
            spec = self.subagents.get(item["subagent"])
            extras = [t for t in item["extra_tools"] if t in spec.optional_tools and self.tools.has(t)]
            step = ResearchStep(
                step_id=item["step_id"],
                description=item["description"],
                subagent=item["subagent"],
                depends_on=list(item["depends_on"]),
                tools=list(spec.required_tools) + extras,
                completion_criteria=item["completion_criteria"],
            )
            steps.append(step)
            routing.append(
                RoutingDecision(
                    task=step.description,
                    subagent=step.subagent,
                    selected=True,
                    reason=item["reason"],
                    confidence=_clamp(item["confidence"]),
                    step_id=step.step_id,
                )
            )

        selected = {s.subagent for s in steps}
        for item in raw["skipped"]:
            if item["subagent"] not in selected:
                routing.append(
                    RoutingDecision("none", item["subagent"], False, item["reason"], 0.8)
                )

        known_ids = {lid for lid, _ in lessons}
        plan = ResearchPlan(
            symbol=symbol,
            question=question,
            intent=raw["intent"],
            steps=steps,
            scope=raw["scope"],
            macro_context_requested=bool(raw["macro_context_relevant"]),
            lessons_applied=[l for l in raw["lessons_applied"] if l in known_ids],
            planner_mode="llm",
            rationale=raw["rationale"],
        )
        problems = validate_plan(plan, self.subagents, self.settings.max_plan_steps)
        if problems:
            raise PlanValidationError(problems)
        return PlanningResult(plan, routing)

    # ---- rule-based path --------------------------------------------------------

    def _rule_based_plan(self, symbol: str, question: str) -> PlanningResult:
        profile = interpret_question(question, self.subagents)
        names = self.subagents.names()
        if profile.scope == "comprehensive":
            selected = names
        else:
            selected = [n for n in names if n in profile.matches]
        if profile.sequential:
            # Earnings must come first so the market step can examine the reaction.
            selected = sorted(selected, key=lambda n: n != SUBAGENT_EARNINGS)

        steps: list[ResearchStep] = []
        routing: list[RoutingDecision] = []
        for idx, name in enumerate(selected, start=1):
            step = self._template_step(f"S{idx}", name, symbol, profile)
            if profile.sequential and name == SUBAGENT_MARKET:
                step.depends_on = [s.step_id for s in steps if s.subagent == SUBAGENT_EARNINGS]
            steps.append(step)

            if profile.broad_fallback:
                reason, conf = "No specific focus recognised; defaulting to broad coverage.", 0.4
            elif profile.scope == "comprehensive":
                reason, conf = "Comprehensive research requested; every specialist is required.", 0.9
            else:
                hits = ", ".join(sorted(set(profile.matches[name]))[:4])
                reason, conf = f"Question mentions: {hits}.", min(0.95, 0.6 + 0.1 * len(profile.matches[name]))
            routing.append(RoutingDecision(step.description, name, True, reason, conf, step.step_id))

        intent = (
            "Comprehensive investment research"
            if profile.scope == "comprehensive"
            else "Focused research: " + ", ".join(n.replace("_agent", "") for n in selected)
        )
        plan = ResearchPlan(
            symbol=symbol,
            question=question,
            intent=intent,
            steps=steps,
            scope=profile.scope,
            macro_context_requested=profile.macro and SUBAGENT_MARKET in selected,
            planner_mode="rule_based",
            rationale="Deterministic keyword routing over the research question.",
        )
        return PlanningResult(plan, routing)

    def _template_step(self, step_id: str, name: str, symbol: str, profile: QuestionProfile | None) -> ResearchStep:
        spec = self.subagents.get(name)
        description, criteria = _STEP_TEMPLATES.get(
            name, (f"Run {name} for {{symbol}}", f"{name} returns a valid structured result")
        )
        tools = list(spec.required_tools)
        if profile is not None:
            # Optional tools are added only when the question calls for them.
            if profile.macro and TOOL_MACRO in spec.optional_tools and self.tools.has(TOOL_MACRO):
                tools.append(TOOL_MACRO)
                description += "; include macroeconomic context"
            if profile.filings and TOOL_SEC_FILINGS in spec.optional_tools and self.tools.has(TOOL_SEC_FILINGS):
                tools.append(TOOL_SEC_FILINGS)
                description += "; include SEC filing context"
        return ResearchStep(step_id, description.format(symbol=symbol), name, tools=tools, completion_criteria=criteria)

    # ---- memory and routing completion ------------------------------------------

    def _select_lessons(self, memories: Iterable[Any]) -> list[tuple[str, MemoryItem]]:
        items = [MemoryItem.from_raw(m) for m in memories]
        items.sort(key=lambda m: m.score, reverse=True)
        return [(m.memory_id or f"L{i}", m) for i, m in enumerate(items[:MAX_LESSONS], start=1)]

    def _apply_lessons(self, result: PlanningResult, lessons: list[tuple[str, MemoryItem]]) -> None:
        """Apply machine-readable lesson hints; lessons only shape the plan, never the evidence."""
        plan = result.plan
        for lid, memory in lessons:
            hint = memory.hint or {}
            applied = False

            for name in hint.get("require_subagents", []):
                if self.subagents.has(name) and name not in plan.selected_subagents:
                    step = self._template_step(f"S{len(plan.steps) + 1}", name, plan.symbol, None)
                    plan.steps.append(step)
                    result.routing = [r for r in result.routing if not (r.subagent == name and not r.selected)]
                    result.routing.append(
                        RoutingDecision(step.description, name, True, f"Added by lesson {lid}: {memory.lesson}", 0.7, step.step_id)
                    )
                    applied = True

            extra = str(hint.get("extra_criteria") or "").strip()
            if extra:
                target = hint.get("subagent")
                for step in plan.steps:
                    if target in (None, step.subagent) and extra not in step.completion_criteria:
                        step.completion_criteria = f"{step.completion_criteria}; {extra}"
                        applied = True

            if applied and lid not in plan.lessons_applied:
                plan.lessons_applied.append(lid)

    def _complete_routing(self, result: PlanningResult) -> None:
        """Guarantee that every registered subagent has a selected or skipped decision with a reason."""
        selected = set(result.plan.selected_subagents)
        by_agent = {r.subagent: r for r in result.routing if not r.selected}
        final = [r for r in result.routing if r.selected]
        for name in self.subagents.names():
            if name in selected:
                continue
            existing = by_agent.get(name)
            reason = (existing.reason if existing and existing.reason.strip() else _SKIP_REASONS.get(name, "Not required for this question."))
            final.append(RoutingDecision("none", name, False, reason, existing.confidence if existing else 0.8))
        result.routing = final


def _clamp(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0

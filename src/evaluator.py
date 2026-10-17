"""Evaluator: scores a report against an eight-criterion rubric and says what to fix.

Seven criteria are checked in code against the evidence pack, so the scores are reproducible and every
issue points at a section, often at one statement, with a concrete corrective action. Clarity is the
one subjective criterion: the LLM judges it, and simple text heuristics stand in without an LLM.

The overall score is the weighted mean of the criterion scores (0 to 100). A report is accepted when
the overall score reaches ``Settings.quality_threshold`` and no criterion falls below ``CRITERION_FLOOR``.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable

from .config import PROJECT_ROOT, Settings
from .models import SUBAGENT_EARNINGS, SUBAGENT_MARKET, SUBAGENT_NEWS
from .report_evidence import AGENT_LABEL, AGENT_SECTION, WRITABLE_SECTIONS, EvidencePack
from .report_models import (
    DECISION_ACCEPT,
    DECISION_REFINE,
    FACT,
    SECTION_COMPLETE,
    SECTION_ORDER,
    CriterionScore,
    EvaluationResult,
    ReportIssue,
    ResearchReport,
)

PROMPT_PATH = PROJECT_ROOT / "prompts" / "evaluator.txt"

# (name, label, weight). Weights sum to 100, so the overall score reads as a percentage.
CRITERIA: tuple[tuple[str, str, float], ...] = (
    ("evidence_grounding", "Evidence grounding", 20.0),
    ("completeness", "Completeness", 15.0),
    ("numeric_consistency", "Numeric consistency", 15.0),
    ("source_recency", "Source recency", 10.0),
    ("risk_coverage", "Risk coverage", 10.0),
    ("catalyst_coverage", "Catalyst coverage", 10.0),
    ("clarity", "Clarity", 10.0),
    ("uncertainty_disclosure", "Uncertainty disclosure", 10.0),
)
CRITERION_FLOOR = 60.0  # every criterion must at least pass; a clarity rating of 3 of 5 (50) does not
FINDING_COVERAGE_TARGET = 0.75  # share of a specialist's findings the report should reflect
DATA_SECTIONS = ("market_performance", "financial_analysis", "news_and_events", "macro_context")

_DRIVER_WORDS = re.compile(r"\b(driver|drove|explain|cause|reason|why)\w*", re.I)
_NEGATION = re.compile(r"\b(not|no|without|unexplained)\b", re.I)
_DATE_MENTION = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|\bfiscal\b|\bFY\s?\d{2,4}\b|\b(19|20)\d{2}\b|"
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b",
    re.I,
)
_OVERCLAIM = re.compile(
    r"\b(guarantee[sd]?|certain to|will (?:definitely|certainly|surely)|risk[- ]free|strong (?:buy|sell)|"
    r"(?:you|investors) should (?:buy|sell|hold)|must (?:buy|sell)|sure to|cannot lose|price target)\b",
    re.I,
)
_NEGATED = re.compile(r"\b(not|no|never|without|cannot|nor)\b[^.;]{0,20}$", re.I)
_RAW_FIGURE = re.compile(r"(?<![\w.,])\d{7,}(?:\.\d+)?(?![\d,])")  # 331839000000.0: no separators, no scale word
_DOUBLE_NEGATIVE = re.compile(
    r"\b(underperform\w*|trail\w*|lag\w*|fell|declin\w*|dropp\w*|decreas\w*|lost)\b(?:[^.;]|\.(?=\d)){0,60}?\bby\s+[-\u2212]\d",
    re.I,
)
_STOPWORDS = frozenset(
    "the and for with that this from was were are has have had its their into over than also more most less "
    "which while would could about after before since been being per year years".split()
)

CLARITY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["score", "summary", "problems"],
    "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
        "summary": {"type": "string"},
        "problems": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["section", "problem", "fix"],
                "properties": {
                    "section": {"type": "string", "enum": list(SECTION_ORDER)},
                    "problem": {"type": "string"},
                    "fix": {"type": "string"},
                },
            },
        },
    },
}


@dataclass
class Check:
    """Result of one criterion check."""

    score: float
    issues: list[ReportIssue] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


# ---- entry point ------------------------------------------------------------------------------


def evaluate_report(
    report: ResearchReport,
    pack: EvidencePack,
    llm: Any = None,
    settings: Settings | None = None,
    cycle: int = 0,
) -> EvaluationResult:
    """Score every criterion, collect issues with fixes, and decide whether to accept or refine."""
    settings = settings or Settings()
    checks = {
        "evidence_grounding": check_grounding(report, pack),
        "completeness": check_completeness(report, pack),
        "numeric_consistency": check_numbers(report, pack),
        "source_recency": check_recency(report, pack),
        "risk_coverage": check_coverage(report, pack, "risks"),
        "catalyst_coverage": check_coverage(report, pack, "catalysts"),
        "clarity": check_clarity(report, llm),
        "uncertainty_disclosure": check_uncertainty(report, pack),
    }
    criteria = [
        CriterionScore(name, label, weight, round(max(0.0, min(100.0, checks[name].score)), 1), checks[name].details)
        for name, label, weight in CRITERIA
    ]
    overall = round(sum(c.weight * c.score for c in criteria) / sum(c.weight for c in criteria), 1)
    issues = [issue for name, _, _ in CRITERIA for issue in checks[name].issues]
    for issue in issues:
        # Code-built sections (overview, sources) are not rewritten, so their issues are informational.
        if issue.section not in WRITABLE_SECTIONS:
            issue.actionable = False
    accept = overall >= settings.quality_threshold and min(c.score for c in criteria) >= CRITERION_FLOOR
    return EvaluationResult(
        cycle=cycle,
        report_version=report.version,
        overall_score=overall,
        criteria=criteria,
        issues=issues,
        decision=DECISION_ACCEPT if accept else DECISION_REFINE,
        threshold=settings.quality_threshold,
        floor=CRITERION_FLOOR,
        judge_mode=checks["clarity"].details.get("judge_mode", "heuristic"),
        report=report.to_dict(),
    )


# ---- evidence grounding -------------------------------------------------------------------------


def check_grounding(report: ResearchReport, pack: EvidencePack) -> Check:
    """Facts must cite evidence that exists; every completed data section must state at least one fact."""
    issues: list[ReportIssue] = []
    facts = cited = 0
    for section, index, statement in report.statements():
        unknown = [eid for eid in statement.evidence_ids if eid not in pack.evidence]
        if unknown:
            issues.append(
                ReportIssue(
                    "evidence_grounding", section.key, "high",
                    f"Cites {', '.join(unknown)}, which is not in the evidence pack: \"{_short(statement.text)}\"",
                    "Cite only evidence IDs from the pack; memory and lesson IDs are guidance, never evidence.",
                    code="invalid_citation", refs=unknown, statement_index=index,
                )
            )
        if statement.kind != FACT:
            continue
        facts += 1
        if any(eid in pack.evidence for eid in statement.evidence_ids):
            cited += 1
        else:
            issues.append(
                ReportIssue(
                    "evidence_grounding", section.key, "high",
                    f"Fact without a valid citation: \"{_short(statement.text)}\"",
                    "Cite the evidence ID that supports it, or relabel it as an interpretation if it is a judgement.",
                    code="uncited_fact", statement_index=index,
                )
            )
    data_sections = [s for s in report.sections if s.key in DATA_SECTIONS and s.status == SECTION_COMPLETE]
    without_facts = [s for s in data_sections if not any(st.kind == FACT for st in s.statements)]
    for section in without_facts:
        issues.append(
            ReportIssue(
                "evidence_grounding", section.key, "medium",
                f"The {section.title.lower()} section states no facts.",
                "Add the key figures from the specialist's findings as cited facts.",
                code="no_facts",
            )
        )
    cited_share = cited / facts if facts else 0.0
    sections_share = 1 - len(without_facts) / len(data_sections) if data_sections else 1.0
    return Check(
        100 * (0.85 * cited_share + 0.15 * sections_share),
        issues,
        {"facts": facts, "cited_facts": cited, "data_sections_without_facts": [s.key for s in without_facts]},
    )


# ---- numeric consistency ------------------------------------------------------------------------


@dataclass(frozen=True)
class Number:
    """A number read from text or evidence, normalised so that "331.8 bn" and 331,794,000,000 compare."""

    raw: str
    value: float  # signed, after the scale word is applied
    percent: bool  # percent and percentage points compare only with each other
    decimals: int  # precision written in the text; sets the rounding tolerance
    scale: float = 1.0
    signed: bool = False  # an explicit sign was written, or the value came from evidence
    after: str = ""  # the words right after the number, which say whether it is a ratio ("2.6 times")

    def tolerance(self) -> float:
        """Half a unit in the last written digit: "6.0%" covers 5.95 to 6.05."""
        return 0.5 * 10 ** (-self.decimals) * self.scale + 1e-9

    def close_to(self, other: "Number") -> bool:
        """Same kind and the same magnitude within rounding; the sign is judged by ``supported``."""
        return self.percent == other.percent and abs(abs(self.value) - abs(other.value)) <= max(self.tolerance(), other.tolerance())


def supported(number: Number, refs: list[Number]) -> bool:
    """True when some reference backs the number, directly or by simple arithmetic.

    An explicit sign must agree with signed references: "+12.6 percentage points" fails against an
    excess return of -12.58 even though the specialist's own sentence says "underperforming by 12.6
    percentage points" without a sign. A number worded as a ratio ("2.6 times", "1.9x") may be the
    quotient of two cited figures, and a gap in percentage points may be their difference.
    """
    close = [ref for ref in refs if number.close_to(ref)]
    if close:
        signed = [ref for ref in close if ref.signed and ref.value]
        if not (number.signed and number.value and signed):
            return True
        return any((ref.value > 0) == (number.value > 0) for ref in signed)
    return _derived(number, refs)


def _derived(number: Number, refs: list[Number]) -> bool:
    tolerance = number.tolerance()
    values = [ref.value for ref in refs if ref.value]
    is_ratio = not number.percent and (number.raw.lower().endswith("x") or number.after.lstrip().startswith(("times", "x ", "x,", "-fold", "fold")))
    if is_ratio:
        return any(abs(a / b - abs(number.value)) <= tolerance for a in values for b in values if a is not b and b)
    if number.percent and ("point" in number.raw.lower() or number.raw.lower().endswith("pp")):
        points = [ref.value for ref in refs if ref.percent]
        return any(abs(abs(a - b) - abs(number.value)) <= tolerance for a in points for b in points if a is not b)
    return False


_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
_STRIP = [
    re.compile(r"\[[^\]]*\]"),  # bracketed citations
    re.compile(r"\b(?:MKT|ERN|NEWS|PRO|EV|MEM)-\d+\b"),  # evidence and memory IDs
    re.compile(r"\b[RCLEFS]\d+\b"),  # risk, catalyst, limitation, event, finding and step IDs
    re.compile(r"\b\d{4}-\d{2}-\d{2}(?:T[\d:]+(?:[+-]\d{2}:\d{2})?)?\b"),  # ISO dates and timestamps
    re.compile(rf"\b{_MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b", re.I),  # October 17, 2025
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}\.?(?:,?\s+\d{{4}})?\b", re.I),  # 17 October 2025
    re.compile(rf"\b{_MONTH}\.?\s+\d{{4}}\b", re.I),  # October 2025
    re.compile(r"\b\d+(?:\.\d+)?[- ](?:day|week|month|year|quarter)s?\b", re.I),  # windows such as 50-day
    re.compile(r"\b\d+[MYQ]\b"),  # 1M, 3M, 6M
    re.compile(r"\b(?:FY|Q)\s?\d{1,4}\b", re.I),  # FY2025, Q3
    re.compile(r"\b\d+-[KQ]\b|\b8-K\b", re.I),  # 10-K, 10-Q, 8-K
    re.compile(r"\bS&P\s?500\b", re.I),
    re.compile(r"\b\d+(?:st|nd|rd|th)\b", re.I),  # ordinals
]
_NUMBER = re.compile(
    r"(?<![\w.])(?P<sign>[+\-−])?(?P<currency>[$€£])?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s?(?P<unit>%|percentage points?|per ?cent\b|pp\b|bn\b|billion\b|million\b|mn\b|trillion\b|tn\b|thousand\b|x\b))?",
    re.I,
)
_SCALES = {"bn": 1e9, "billion": 1e9, "million": 1e6, "mn": 1e6, "trillion": 1e12, "tn": 1e12, "thousand": 1e3}


def extract_numbers(text: str) -> list[Number]:
    """Numbers worth checking in a sentence: IDs, dates, windows, years and form names are skipped."""
    clean = str(text or "")
    for pattern in _STRIP:
        clean = pattern.sub(" ", clean)
    numbers = []
    for match in _NUMBER.finditer(clean):
        digits, unit, sign = match.group("num"), (match.group("unit") or "").lower(), match.group("sign")
        value = float(digits.replace(",", ""))
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        percent = unit.startswith(("%", "percent", "per cent", "pp"))
        scale = _SCALES.get(unit, 1.0)
        if not (percent or match.group("currency") or decimals or scale != 1.0) and 1900 <= value <= 2100:
            continue  # a bare year
        if sign in ("-", "−"):
            value = -value
        after = clean[match.end(): match.end() + 16].lower()
        numbers.append(Number(match.group(0).strip(), value * scale, percent, decimals, scale, bool(sign), after))
    return numbers


def reference_numbers(cited: Iterable[dict[str, Any]], findings: Iterable[dict[str, Any]]) -> list[Number]:
    """Every figure a statement may legitimately quote: cited evidence values and claims, and the
    specialists' own finding sentences for the same evidence (they hold computed figures such as prices)."""
    refs: list[Number] = []
    for item in cited:
        percent = str(item.get("unit") or "").strip().lower() in ("%", "percentage points", "percent")
        refs += [Number(str(v), float(v), percent, 6, 1.0, True) for v in _flatten(item.get("value"))]
        refs += extract_numbers(str(item.get("claim", "")))
    for finding in findings:
        refs += extract_numbers(finding["statement"])
    return refs


def check_numbers(report: ResearchReport, pack: EvidencePack) -> Check:
    """Every number in a cited fact must match a figure in its cited evidence (after unit normalisation)."""
    issues: list[ReportIssue] = []
    checked = matched = 0
    for section, index, statement in report.statements():
        cited = [pack.evidence[eid] for eid in statement.evidence_ids if eid in pack.evidence]
        if statement.kind != FACT or not cited:
            continue
        related = [f for f in pack.findings if set(f["evidence_ids"]) & set(statement.evidence_ids)]
        refs = reference_numbers(cited, related)
        for number in extract_numbers(statement.text):
            checked += 1
            if supported(number, refs):
                matched += 1
                continue
            issues.append(
                ReportIssue(
                    "numeric_consistency", section.key, "high",
                    f"\"{number.raw}\" matches none of the cited evidence ({', '.join(statement.evidence_ids)}): "
                    f"\"{_short(statement.text)}\"",
                    "Copy the figure exactly from the evidence, or cite the evidence item that contains it.",
                    code="unverified_number", refs=list(statement.evidence_ids), statement_index=index,
                )
            )
    return Check(100.0 * matched / checked if checked else 100.0, issues, {"numbers_checked": checked, "numbers_matched": matched})


# ---- completeness -------------------------------------------------------------------------------


def check_completeness(report: ResearchReport, pack: EvidencePack) -> Check:
    """Sections filled, findings reflected, coverage gaps explained, and price moves given their drivers."""
    issues: list[ReportIssue] = []
    cited = report.cited_ids()
    parts: dict[str, tuple[float, float]] = {}  # component -> (score 0..1, weight)

    writable = [s for s in report.sections if s.status == SECTION_COMPLETE and s.key in WRITABLE_SECTIONS]
    empty = [s for s in writable if not s.statements]
    for section in empty:
        issues.append(
            ReportIssue(
                "completeness", section.key, "high", f"The {section.title.lower()} section is empty.",
                "Write the section from the evidence pack, or add a disclosure if there is nothing to report.",
                code="empty_section",
            )
        )
    parts["sections"] = (1 - len(empty) / len(writable) if writable else 1.0, 0.2)

    ratios = []
    groups = [(agent, False, AGENT_SECTION[agent]) for agent in AGENT_SECTION] + [(SUBAGENT_MARKET, True, "macro_context")]
    for agent, macro, section_key in groups:
        if not pack.completed(agent) or report.section(section_key).status != SECTION_COMPLETE:
            continue
        findings = [f for f in pack.findings_for(agent, macro=macro) if f["evidence_ids"]]
        if not findings:
            continue
        covered = [f for f in findings if set(f["evidence_ids"]) & cited]
        ratios.append(len(covered) / len(findings))
        if len(covered) / len(findings) < FINDING_COVERAGE_TARGET:
            missing = [f for f in findings if f not in covered]
            label = "macro" if macro else AGENT_LABEL[agent]
            issues.append(
                ReportIssue(
                    "completeness", section_key, "medium",
                    f"Only {len(covered)} of {len(findings)} {label} findings are reflected in the report.",
                    "Add the missing findings as cited facts: "
                    + "; ".join(f"{_short(f['statement'], 90)} [{', '.join(f['evidence_ids'])}]" for f in missing[:4]),
                    code="missing_findings", refs=[eid for f in missing for eid in f["evidence_ids"]],
                )
            )
    parts["findings"] = (sum(ratios) / len(ratios) if ratios else 1.0, 0.45)

    gaps = [l for l in pack.limitations if "did not complete" in l["text"]]
    disclosed_in = report.section("conflicting_or_missing_evidence").statements + report.section("confidence_and_limitations").statements
    explained = 0
    for gap in gaps:
        if any(gap["id"] in st.limitation_ids or similar(gap["text"], st.text) for st in disclosed_in):
            explained += 1
            continue
        issues.append(
            ReportIssue(
                "completeness", "conflicting_or_missing_evidence", "high", f"Coverage gap not explained: {gap['text']}",
                f"State the gap in the conflicting or missing evidence section and tag it {gap['id']}.",
                code="gap_unexplained", refs=[gap["id"]],
            )
        )
    parts["gaps"] = (explained / len(gaps) if gaps else 1.0, 0.15)

    drivers = _check_drivers(report, pack)
    if drivers is not None:
        score, issue = drivers
        parts["drivers"] = (score, 0.2)
        if issue:
            issues.append(issue)

    total_weight = sum(w for _, w in parts.values())
    score = 100 * sum(s * w for s, w in parts.values()) / total_weight
    return Check(score, issues, {name: round(s, 2) for name, (s, _) in parts.items()})


def _check_drivers(report: ResearchReport, pack: EvidencePack) -> tuple[float, ReportIssue | None] | None:
    """A price analysis should say what drove the move, or admit that the drivers were not analysed."""
    if not pack.completed(SUBAGENT_MARKET) or report.section("market_performance").status != SECTION_COMPLETE:
        return None
    market, summary = report.section("market_performance"), report.section("executive_summary")
    # Linked means a market statement cites news or catalysts, or one summary statement cites both kinds
    # of evidence; a summary that merely lists a headline next to the return does not explain the move.
    in_market = any(st.catalyst_ids or any(e.startswith("NEWS-") for e in st.evidence_ids) for st in market.statements)
    in_summary = any(
        any(e.startswith("NEWS-") for e in st.evidence_ids) and any(e.startswith("MKT-") for e in st.evidence_ids)
        for st in summary.statements
    )
    if in_market or in_summary:
        return 1.0, None
    if pack.completed(SUBAGENT_NEWS):
        return 0.0, ReportIssue(
            "completeness", "market_performance", "medium",
            "The market section reports the price move without linking it to the news analysed in this run.",
            "Add an interpretation that connects the notable price move to relevant events, citing their NEWS IDs.",
            code="price_drivers_unlinked",
        )
    candidates = market.statements + report.section("conflicting_or_missing_evidence").statements
    disclosed = any(_DRIVER_WORDS.search(st.text) and _NEGATION.search(st.text) for st in candidates)
    if disclosed:
        return 0.5, ReportIssue(
            "completeness", "market_performance", "low",
            "The price move is reported without its drivers: news analysis was not part of this run (disclosed).",
            "Explaining the move needs the news specialist in the plan; the disclosure is the best this run can do.",
            code="price_drivers_missing", actionable=False,
        )
    return 0.0, ReportIssue(
        "completeness", "market_performance", "medium",
        "The price move is reported without its drivers, and the report does not say they were not analysed.",
        "State that the drivers of the price move were not analysed because news analysis was not part of this run.",
        code="price_drivers_missing",
    )


# ---- source recency -----------------------------------------------------------------------------


def freshness_limit_days(item: dict[str, Any], pack: EvidencePack) -> int:
    """How old a cited item may be before the report must flag its age."""
    source, agent = str(item.get("source", "")), item.get("source_agent")
    if "FRED" in source:
        return 120  # monthly series such as CPI are published with a lag
    if agent == SUBAGENT_MARKET:
        return 7
    if agent == SUBAGENT_NEWS:
        return pack.lookback_days + 3
    if agent == SUBAGENT_EARNINGS:
        return 457  # about 15 months: an annual report plus its filing delay
    return 3650


def check_recency(report: ResearchReport, pack: EvidencePack) -> Check:
    """Cited evidence should be fresh, and each data section should say which period it covers."""
    issues: list[ReportIssue] = []
    credits: list[float] = []
    flagged: set[tuple[str, str]] = set()
    stale: list[str] = []
    for section, _, statement in report.statements():
        for eid in statement.evidence_ids:
            item = pack.evidence.get(eid)
            as_of = _parse_date(item.get("as_of")) if item else None
            if as_of is None:
                continue
            age = (pack.run_date - as_of).days
            if age <= freshness_limit_days(item, pack):
                credits.append(1.0)
                continue
            stated = str(item["as_of"]) in section.text()
            credits.append(0.5 if stated else 0.0)
            stale.append(eid)
            if not stated and (eid, section.key) not in flagged:
                flagged.add((eid, section.key))
                issues.append(
                    ReportIssue(
                        "source_recency", section.key, "medium",
                        f"{eid} is dated {item['as_of']} ({age} days old), and the section does not say how old the data is.",
                        f"State the data date ({item['as_of']}) in the section.",
                        code="stale_evidence", refs=[eid],
                    )
                )
    dated = []
    for key in DATA_SECTIONS:
        section = report.section(key)
        if section.status != SECTION_COMPLETE:
            continue
        ok = bool(_DATE_MENTION.search(section.text()))
        dated.append(ok)
        if not ok:
            period = pack.data_periods.get("macro" if key == "macro_context" else _section_agent(key), "the period the data covers")
            issues.append(
                ReportIssue(
                    "source_recency", key, "low", f"The {section.title.lower()} section does not say which period its data covers.",
                    f"State the period, for example: {period}", code="data_date_missing",
                )
            )
    fresh = sum(credits) / len(credits) if credits else 1.0
    dated_share = sum(dated) / len(dated) if dated else 1.0
    return Check(
        100 * (0.7 * fresh + 0.3 * dated_share),
        issues,
        {"cited_items_dated": len(credits), "stale_items": sorted(set(stale)), "sections_stating_period": f"{sum(dated)}/{len(dated)}"},
    )


# ---- risk and catalyst coverage ------------------------------------------------------------------


def check_coverage(report: ResearchReport, pack: EvidencePack, kind: str) -> Check:
    """Share of the specialists' numbered risks (or catalysts) that the report covers."""
    name, id_field, singular = (
        ("risk_coverage", "risk_ids", "risk") if kind == "risks" else ("catalyst_coverage", "catalyst_ids", "catalyst")
    )
    section, items = report.section(kind), getattr(pack, kind)
    if section.status != SECTION_COMPLETE:
        return Check(100.0, [], {"items": 0, "covered": 0, "note": "not assessed in this run, and the section says so"})
    if not items:
        if section.statements:
            return Check(100.0, [], {"items": 0, "covered": 0})
        issue = ReportIssue(
            name, kind, "medium", f"No {kind} were flagged, and the section does not say so.",
            f"Add a disclosure that the specialists flagged no {kind}.", code="empty_section",
        )
        return Check(60.0, [issue], {"items": 0, "covered": 0})
    everywhere = [st for _, _, st in report.statements()]
    issues, covered = [], []
    for item in items:
        if any(item["id"] in getattr(st, id_field) for st in everywhere) or any(similar(item["text"], st.text) for st in section.statements):
            covered.append(item["id"])
            continue
        cite = f", citing {', '.join(item['evidence_ids'])}" if item["evidence_ids"] else ""
        issues.append(
            ReportIssue(
                name, kind, "medium", f"{item['id']} is not covered: \"{_short(item['text'])}\"",
                f"Add {item['id']} to the {kind} section with its ID{cite}.",
                code=f"missing_{singular}", refs=[item["id"]],
            )
        )
    return Check(100.0 * len(covered) / len(items), issues, {"items": len(items), "covered": len(covered)})


# ---- uncertainty disclosure ----------------------------------------------------------------------


def check_uncertainty(report: ResearchReport, pack: EvidencePack) -> Check:
    """Confidence with reasons, every limitation disclosed, and no certain or advisory wording."""
    issues: list[ReportIssue] = []
    confidence = report.confidence or {}
    confidence_ok = 1.0 if confidence.get("level") and confidence.get("reasons") else 0.0
    if not confidence_ok:
        issues.append(
            ReportIssue(
                "uncertainty_disclosure", "confidence_and_limitations", "high", "The report states no confidence level with reasons.",
                "State the confidence level and the reasons for it.", code="no_confidence",
            )
        )
    disclosed_in = report.section("confidence_and_limitations").statements + report.section("conflicting_or_missing_evidence").statements
    covered = 0
    for limitation in pack.limitations:
        if any(limitation["id"] in st.limitation_ids or similar(limitation["text"], st.text) for st in disclosed_in):
            covered += 1
            continue
        issues.append(
            ReportIssue(
                "uncertainty_disclosure", "confidence_and_limitations",
                "medium" if "did not complete" in limitation["text"] else "low",
                f"{limitation['id']} is not disclosed: \"{_short(limitation['text'])}\"",
                f"Add {limitation['id']} to the limitations with its ID.",
                code="missing_limitation", refs=[limitation["id"]],
            )
        )
    if pack.limitations:
        limitations_ok = covered / len(pack.limitations)
    else:
        limitations_ok = 1.0 if disclosed_in else 0.5
    statements = list(report.statements())
    overclaims = [(s, i, st) for s, i, st in statements if _advisory(st, pack)]
    for section, index, statement in overclaims:
        issues.append(
            ReportIssue(
                "uncertainty_disclosure", section.key, "high", f"Certain or advisory wording: \"{_short(statement.text)}\"",
                "Rephrase it as a hedged interpretation; the report gives no investment advice.",
                code="overclaim", statement_index=index,
            )
        )
    wording_ok = 1 - len(overclaims) / len(statements) if statements else 1.0
    return Check(
        100 * (0.2 * confidence_ok + 0.6 * limitations_ok + 0.2 * wording_ok),
        issues,
        {"limitations": len(pack.limitations), "disclosed": covered, "overclaims": len(overclaims)},
    )


def _advisory(statement: Any, pack: EvidencePack) -> bool:
    """Certain or advisory wording in the agent's own voice.

    A cited fact may quote a headline such as "UBS resets its price target"; that wording belongs to the
    source, so it only counts when the cited evidence does not contain it.
    """
    match = _OVERCLAIM.search(statement.text)
    if not match or _NEGATED.search(statement.text[: match.start()]):
        return False  # "past performance does not guarantee future returns" is a disclaimer, not a claim
    if statement.kind != FACT:
        return True
    quoted = " ".join(str(pack.evidence[eid].get("claim", "")) for eid in statement.evidence_ids if eid in pack.evidence)
    return match.group(0).lower() not in quoted.lower()


# ---- clarity ------------------------------------------------------------------------------------


def check_clarity(report: ResearchReport, llm: Any = None) -> Check:
    """LLM judgement of how clearly the report reads, with text heuristics as the fallback."""
    heuristic = clarity_heuristics(report)
    if llm is not None and getattr(llm, "available", False):
        try:
            # Content only: no timestamp (stable cache key) and no sources table (not prose).
            text = report.to_markdown(metadata=False).split("\n## Sources")[0]
            raw = llm.chat_json(
                [
                    {"role": "system", "content": PROMPT_PATH.read_text(encoding="utf-8")},
                    {"role": "user", "content": json.dumps({"report": text})},
                ],
                CLARITY_SCHEMA,
                schema_name="clarity_review",
            )
            score = max(1, min(5, int(raw["score"])))
            issues = [
                ReportIssue(
                    "clarity", p.get("section") if p.get("section") in SECTION_ORDER else "executive_summary", "low",
                    str(p.get("problem", "")), str(p.get("fix", "")), code="clarity_llm",
                )
                for p in (raw.get("problems") or [])[:5]
                if isinstance(p, dict) and p.get("problem")
            ]
            details = {"judge_mode": "llm", "llm_score_1_to_5": score, "summary": str(raw.get("summary", "")), "rule_score": round(heuristic.score, 1)}
            # The LLM judges the prose; the rules catch what it may overlook (raw figures, double negatives).
            return Check(min((score - 1) * 25.0, heuristic.score), issues + heuristic.issues, details)
        except Exception as exc:  # judge unavailable: fall back to heuristics and say why
            heuristic.details["llm_error"] = f"{type(exc).__name__}: {exc}"[:200]
    heuristic.details["judge_mode"] = "heuristic"
    return heuristic


def clarity_heuristics(report: ResearchReport) -> Check:
    """Penalise unformatted figures, double negatives, long sentences, repeated statements and crowded sections."""
    issues: list[ReportIssue] = []
    penalty = 0.0
    statements = list(report.statements())
    raw = [(s, i, st) for s, i, st in statements if _RAW_FIGURE.search(st.text)]
    penalty += min(40.0, 10.0 * len(raw))
    for section, index, statement in raw:
        figure = _RAW_FIGURE.search(statement.text).group(0)
        issues.append(
            ReportIssue(
                "clarity", section.key, "medium", f"Unformatted figure {figure} is hard to read: \"{_short(statement.text)}\"",
                "Write large amounts in billions with one decimal (for example USD 331.8 bn), as the specialist findings do.",
                code="raw_figure", refs=list(statement.evidence_ids), statement_index=index,
            )
        )
    doubled = [(s, i, st) for s, i, st in statements if _DOUBLE_NEGATIVE.search(st.text)]
    penalty += min(15.0, 5.0 * len(doubled))
    for section, index, statement in doubled:
        issues.append(
            ReportIssue(
                "clarity", section.key, "low", f"Double negative: \"{_short(statement.text)}\"",
                "State the gap as a positive number after the word that gives its direction, for example \"trailed SPY by 12.3 percentage points\".",
                code="double_negative", refs=list(statement.evidence_ids), statement_index=index,
            )
        )
    long = [(s, i, st) for s, i, st in statements if len(st.text.split()) > 45]
    penalty += min(30.0, 6.0 * len(long))
    for section, index, statement in long[:5]:
        issues.append(
            ReportIssue(
                "clarity", section.key, "low", f"Long sentence ({len(statement.text.split())} words): \"{_short(statement.text)}\"",
                "Split it into shorter sentences.", code="long_statement", statement_index=index,
            )
        )
    seen: dict[str, str] = {}
    duplicates = 0
    for section, index, statement in statements:
        if section.key == "executive_summary":
            continue  # a summary is expected to restate key findings; repetition elsewhere is penalised
        key = re.sub(r"\W+", " ", statement.text.lower()).strip()
        if key in seen:
            duplicates += 1
            issues.append(
                ReportIssue(
                    "clarity", section.key, "low", f"Repeats a statement from {seen[key]}: \"{_short(statement.text)}\"",
                    "Remove the repetition.", code="duplicate_statement", statement_index=index,
                )
            )
        else:
            seen[key] = section.key
    penalty += min(30.0, 10.0 * duplicates)
    for section in report.sections:
        if len(section.statements) > 12:
            penalty += 10.0
            issues.append(
                ReportIssue(
                    "clarity", section.key, "low", f"The section has {len(section.statements)} statements.",
                    "Merge related statements so the section stays readable.", code="crowded_section",
                )
            )
    lengths = [len(st.text.split()) for _, _, st in statements]
    average = sum(lengths) / len(lengths) if lengths else 0.0
    if average > 32:
        penalty += 10.0
    return Check(
        max(0.0, 100.0 - penalty),
        issues,
        {"average_words_per_statement": round(average, 1), "raw_figures": len(raw), "double_negatives": len(doubled),
         "long_statements": len(long), "duplicates": duplicates},
    )


# ---- helpers ------------------------------------------------------------------------------------


def similar(a: str, b: str, threshold: float = 0.6) -> bool:
    """Loose match: most of ``a``'s content words appear in ``b``. Used when an item's ID was not tagged."""
    words_a, words_b = _content_words(a), _content_words(b)
    if not words_a or not words_b:
        return False
    return len(words_a & words_b) / len(words_a) >= threshold


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z0-9\-]{2,}", str(text).lower()) if w not in _STOPWORDS}


def _flatten(value: Any) -> list[float]:
    if value is None or isinstance(value, (bool, str)):
        return []
    if isinstance(value, dict):
        return [x for v in value.values() for x in _flatten(v)]
    if isinstance(value, (list, tuple)):
        return [x for v in value for x in _flatten(v)]
    try:
        number = float(value)
    except (TypeError, ValueError):
        return []
    return [number] if math.isfinite(number) else []


def _parse_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _section_agent(key: str) -> str:
    return next((agent for agent, section in AGENT_SECTION.items() if section == key), "")


def _short(text: str, limit: int = 120) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


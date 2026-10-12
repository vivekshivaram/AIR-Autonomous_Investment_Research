"""The five stages of the news Prompt Chain: Ingest -> Preprocess -> Classify -> Extract -> Summarize.

Each stage is a function whose output is the next stage's input; `news_agent.py` wires them
into a LangGraph chain. Stages 1-2 are deterministic; stages 3-5 run LangChain chains on the
local model, and each model output is validated before the next stage may use it.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .config import PROJECT_ROOT, Settings, settings as default_settings
from .data_tools import DataToolError, company_terms
from .models import ExtractedNewsEvent, NewsArticle
from .tool_registry import ToolRegistry
from .utils import clean_text, utc_now

EVENT_TYPES = [
    "earnings", "guidance", "product", "regulatory_legal", "m_and_a", "partnership", "management",
    "analyst_rating", "capital_return", "market_commentary", "macro", "other",
]
RELEVANCE = ["high", "medium", "low"]
IMPACT = ["positive", "negative", "neutral"]
HORIZONS = ["short_term", "medium_term", "long_term"]
UNKNOWN = "unknown"  # label used when the model's answer cannot be validated
# Templated items that news feeds return alongside real articles: quote and option-chain pages,
# exchange filing notices, and auto-generated notes about a fund changing its position.
NOISE_PATTERN = re.compile(
    r"stock price, news, quote|latest stock news|quote and history|\(\w+\d{6}[CP]\d{8}\)"
    r"|^REG\s*-|holding\(s\) in company"
    r"|\b(shares|stock|position|holdings)\b.{0,30}\b(acquired|bought|purchased|sold|trimmed|boosted) by\b",
    re.IGNORECASE,
)


def _load_prompt(name: str) -> str:
    return (PROJECT_ROOT / "prompts" / f"{name}.txt").read_text(encoding="utf-8").strip()


def _json_answer(llm: Any, prompt: str, user: str, schema_name: str, schema: dict[str, Any]) -> dict[str, Any]:
    chat_json = getattr(llm, "chat_json", None)
    if callable(chat_json):
        try:
            return chat_json(
                [{"role": "system", "content": prompt}, {"role": "user", "content": user}],
                schema,
                schema_name=schema_name,
            )
        except Exception:
            return {}
    return {}


def _json_answers(
    llm: Any, prompt: str, users: list[str], schema_name: str, schema: dict[str, Any]
) -> list[dict[str, Any]]:
    return [_json_answer(llm, prompt, user, schema_name, schema) for user in users]


def _text_answer(llm: Any, prompt: str, user: str) -> str:
    chat = getattr(llm, "chat", None)
    if callable(chat):
        try:
            return str(chat([{"role": "system", "content": prompt}, {"role": "user", "content": user}])).strip()
        except Exception:
            return ""
    return ""


# ============================================================ Stage 1: Ingest
def ingest_news(
    symbol: str, company_name: str, tools: ToolRegistry, settings: Settings = default_settings
) -> tuple[list[NewsArticle], dict[str, str]]:
    """Retrieve recent headlines through the registered news tool.

    Returns the articles and the errors of any feed that failed (empty when all feeds worked).
    """
    call = tools.call(
        "get_company_news", symbol=symbol, company_name=company_name, lookback_days=settings.news_lookback_days
    )
    if not call.ok:
        raise DataToolError(call.error)
    articles = [
        NewsArticle(
            id=f"A{index:03d}",
            **{"retrieved_at": call.data.get("retrieved_at"), **{k: v for k, v in article.items() if v is not None}},
        )
        for index, article in enumerate(call.data["articles"], start=1)
    ]
    return articles, call.data.get("feed_errors", {})


# ======================================================== Stage 2: Preprocess
def _title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _strip_publisher(title: str, publisher: str) -> str:
    """Aggregator headlines end with ' - Publisher'; remove it so duplicates across feeds match."""
    if publisher and title.lower().endswith(publisher.lower()):
        return title[: -len(publisher)].rstrip(" -|")
    return title


def is_relevant(article: NewsArticle, symbol: str, company_name: str) -> bool:
    """Keep articles whose text names the company or its ticker.

    The feed's own ticker tags are not trusted: they are also attached to stories about other companies.
    """
    text = f"{article.title} {article.summary}"
    if re.search(rf"\b{re.escape(symbol)}\b", text):
        return True
    # Names match at the start of a word, so "Exxon" finds "ExxonMobil" but "Meta" does not find "Fermeta".
    return any(re.search(rf"\b{re.escape(term)}", text, re.IGNORECASE) for term in company_terms(symbol, company_name))


def _merge(kept: NewsArticle, others: list[NewsArticle]) -> NewsArticle:
    """Fold duplicates into the article that is kept: add up the coverage and collect the other publishers."""
    publishers = {p for a in (kept, *others) for p in (a.publisher, *a.also_reported_by)} - {kept.publisher, ""}
    return kept.model_copy(
        update={"duplicate_count": kept.duplicate_count + sum(o.duplicate_count for o in others), "also_reported_by": sorted(publishers)}
    )


def cluster_near_duplicates(articles: list[NewsArticle], threshold: float) -> list[NewsArticle]:
    """Group headlines about the same story (TF-IDF cosine similarity) and keep one per group.

    The representative is the most recent article; it records how many articles and which
    other publishers covered the story, which later serves as an importance signal.
    """
    if len(articles) < 2:
        return articles
    ordered = sorted(articles, key=lambda a: a.published_at, reverse=True)
    try:
        matrix = TfidfVectorizer(stop_words="english").fit_transform(_title_key(a.title) for a in ordered)
    except ValueError:  # every headline consisted of stop words only
        return ordered
    similarity = linear_kernel(matrix)  # TF-IDF rows are L2-normalized, so this is cosine similarity
    assigned, representatives = set(), []
    for i, article in enumerate(ordered):
        if i in assigned:
            continue
        members = [j for j in range(i, len(ordered)) if j not in assigned and similarity[i, j] >= threshold]
        assigned.update(members)
        representatives.append(_merge(article, [ordered[j] for j in members[1:]]))
    return representatives


def preprocess_news(
    raw_articles: list[NewsArticle],
    symbol: str,
    company_name: str,
    settings: Settings = default_settings,
    now: datetime | None = None,
) -> tuple[list[NewsArticle], dict[str, int]]:
    """Clean, validate, deduplicate, filter, and cap the articles.

    Returns the cleaned articles and the article count after every step (the news funnel).
    """
    cutoff = (now or utc_now()) - timedelta(days=settings.news_lookback_days)
    counts = {"ingested": len(raw_articles)}

    # 1. Clean text; drop quote pages and items without a usable headline or a date in the lookback window.
    cleaned = []
    for article in raw_articles:
        publisher = clean_text(article.publisher)
        title = _strip_publisher(clean_text(article.title), publisher)
        if len(title.split()) < 4 or NOISE_PATTERN.search(title):
            continue
        if article.published_at is None or article.published_at < cutoff:
            continue
        cleaned.append(
            article.model_copy(update={"title": title, "summary": clean_text(article.summary), "publisher": publisher})
        )
    counts["valid_and_recent"] = len(cleaned)

    # 2. Exact duplicates: the same headline or URL, typically one story syndicated across feeds.
    unique: dict[str, NewsArticle] = {}
    seen_urls: dict[str, str] = {}
    for article in sorted(cleaned, key=lambda a: a.published_at, reverse=True):
        key = seen_urls.get(article.url) or _title_key(article.title)
        if key in unique:
            unique[key] = _merge(unique[key], [article])
        else:
            unique[key] = article
            if article.url:
                seen_urls[article.url] = key
    counts["after_exact_dedup"] = len(unique)

    # 3. Relevance: the item must actually be about this company.
    relevant = [a for a in unique.values() if is_relevant(a, symbol, company_name)]
    counts["relevant"] = len(relevant)

    # 4. Near duplicates: different wordings of the same story.
    clustered = cluster_near_duplicates(relevant, settings.near_duplicate_threshold)
    counts["after_near_dedup"] = len(clustered)

    # 5. Cap what goes to the LLM: widely covered stories first, then the most recent.
    selected = sorted(clustered, key=lambda a: (a.duplicate_count, a.published_at), reverse=True)
    selected = selected[: settings.max_news_articles]
    counts["selected_for_llm"] = len(selected)
    return selected, counts


# =========================================================== Stage 3: Classify
def _article_prompt(article: NewsArticle, company_name: str, symbol: str) -> str:
    lines = [f"Company: {company_name} ({symbol})", f"Headline: {article.title}"]
    if article.summary:
        lines.append(f"Summary: {article.summary[:600]}")
    return "\n".join(lines)


def _choice(value, allowed: list[str]) -> str:
    """Accept a label only if it is in the allowed set; otherwise mark it unknown."""
    value = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    return value if value in allowed else UNKNOWN


def classify_news(clean_articles: list[NewsArticle], llm, company_name: str, symbol: str) -> list[NewsArticle]:
    """Label each article with event type, relevance, impact, and time horizon."""
    prompts = [_article_prompt(article, company_name, symbol) for article in clean_articles]
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["event_type", "relevance", "impact", "time_horizon"],
        "properties": {
            "event_type": {"type": "string", "enum": EVENT_TYPES},
            "relevance": {"type": "string", "enum": RELEVANCE},
            "impact": {"type": "string", "enum": IMPACT},
            "time_horizon": {"type": "string", "enum": HORIZONS},
        },
    }
    answers = _json_answers(llm, _load_prompt("news_classifier"), prompts, "news_classification", schema)
    return [
        article.model_copy(
            update={
                "event_type": _choice(answer.get("event_type"), EVENT_TYPES),
                "relevance": _choice(answer.get("relevance"), RELEVANCE),
                "impact": _choice(answer.get("impact"), IMPACT),
                "time_horizon": _choice(answer.get("time_horizon"), HORIZONS),
            }
        )
        for article, answer in zip(clean_articles, answers)
    ]


# ============================================================ Stage 4: Extract
def _grounded(items, text: str, limit: int = 6) -> list[str]:
    """Keep only extracted strings that literally occur in the source text.

    This is the guard against hallucinated names, figures, and dates.
    """
    if not isinstance(items, list):
        return []
    haystack = text.lower()
    kept = [clean_text(i) for i in items if isinstance(i, (str, int, float)) and clean_text(i).lower() in haystack]
    return list(dict.fromkeys(k for k in kept if k))[:limit]


def _short_list(items, limit: int = 2) -> list[str]:
    if not isinstance(items, list):
        return []
    return [clean_text(i)[:160] for i in items if isinstance(i, str) and clean_text(i)][:limit]


def extract_news_events(
    classified_articles: list[NewsArticle], llm, company_name: str, symbol: str
) -> list[ExtractedNewsEvent]:
    """Extract a structured event from every article that is not low-relevance.

    A story covered by several publishers is kept even when the model labels it low:
    breadth of coverage is an importance signal that does not depend on the model.
    """
    material = [a for a in classified_articles if a.relevance != "low" or a.duplicate_count >= 2]
    material.sort(key=lambda a: (RELEVANCE + [UNKNOWN]).index(a.relevance))
    prompts = [_article_prompt(article, company_name, symbol) for article in material]
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["event", "entities", "figures", "dates", "risks", "catalysts"],
        "properties": {
            "event": {"type": "string"},
            "entities": {"type": "array", "items": {"type": "string"}},
            "figures": {"type": "array", "items": {"type": "string"}},
            "dates": {"type": "array", "items": {"type": "string"}},
            "risks": {"type": "array", "items": {"type": "string"}},
            "catalysts": {"type": "array", "items": {"type": "string"}},
        },
    }
    answers = _json_answers(llm, _load_prompt("news_extractor"), prompts, "news_extraction", schema)
    events = []
    for index, (article, answer) in enumerate(zip(material, answers), start=1):
        source_text = f"{article.title} {article.summary}"
        event_text = clean_text(answer.get("event"))
        events.append(
            ExtractedNewsEvent(
                id=f"E{index}",
                article_id=article.id,
                # If the model returned nothing usable, fall back to the headline itself.
                event=event_text or article.title,
                extraction_method="llm" if event_text else "headline_fallback",
                event_date=article.published_at.date().isoformat(),
                entities=_grounded(answer.get("entities"), source_text),
                figures=_grounded(answer.get("figures"), source_text),
                dates_mentioned=_grounded(answer.get("dates"), source_text),
                risks=_short_list(answer.get("risks")),
                catalysts=_short_list(answer.get("catalysts")),
                event_type=article.event_type,
                relevance=article.relevance,
                impact=article.impact,
                time_horizon=article.time_horizon,
                source=article.publisher,
                url=article.url,
                retrieved_at=article.retrieved_at,
            )
        )
    return events


# ========================================================== Stage 5: Summarize
def summarize_news(extracted_events: list[ExtractedNewsEvent], llm, company_name: str, symbol: str) -> dict:
    """Write an evidence-backed summary from the extracted events only.

    Citations such as [E3] are validated against the event list; if the model cites nothing
    valid, a deterministic summary built from the events is used instead.
    """
    if not extracted_events:
        return {"summary": f"No material recent news was identified for {company_name}.", "cited_event_ids": [], "method": "no_events"}

    event_lines = "\n".join(
        f"{e.id} | {e.event_date} | {e.source} | {e.event_type} | {e.impact} | {e.event}" for e in extracted_events
    )
    user = f"Company: {company_name} ({symbol})\n\nEvents (id | date | source | type | impact | event):\n{event_lines}"
    text = clean_text(_text_answer(llm, _load_prompt("news_summarizer"), user))

    valid_ids = {event.id for event in extracted_events}
    cited = list(dict.fromkeys(re.findall(r"E\d+", text)))
    invalid = [event_id for event_id in cited if event_id not in valid_ids]
    for event_id in invalid:  # remove citations of events that do not exist
        text = re.sub(rf"\[{event_id}\]", "", text)
    cited = [event_id for event_id in cited if event_id in valid_ids]

    method = "llm"
    if not cited:
        top = extracted_events[:5]
        text = " ".join(f"{e.event.rstrip('.')} [{e.id}]." for e in top)
        cited, method = [e.id for e in top], "deterministic_fallback"
    return {"summary": text, "cited_event_ids": cited, "method": method}

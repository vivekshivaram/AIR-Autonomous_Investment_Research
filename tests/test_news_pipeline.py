from datetime import datetime, timedelta, timezone

from src.config import Settings
from src.models import NewsArticle
from src.news_agent import run_news_agent
from src.news_pipeline import (
    UNKNOWN, classify_news, cluster_near_duplicates, extract_news_events, is_relevant, preprocess_news, summarize_news,
)
from src.models import SubagentInput
from tests.helpers import StubTools, fake_llm

NOW = datetime(2026, 10, 10, tzinfo=timezone.utc)


def article(n, title, days_ago=1, publisher="Reuters", url=None, **extra):
    return NewsArticle(
        id=f"A{n:03d}", title=title, publisher=publisher, url=url or f"https://example.com/{n}",
        published_at=NOW - timedelta(days=days_ago), **extra,
    )


def preprocess(articles, **settings):
    return preprocess_news(articles, "ACME", "Acme Corporation", Settings(**settings), now=NOW)


def test_exact_duplicates_across_feeds_are_merged():
    clean, counts = preprocess([
        article(1, "Acme Corp beats quarterly profit estimates - Reuters"),
        article(2, "Acme Corp beats quarterly profit estimates", publisher="Yahoo Finance"),
        article(3, "Acme Corp opens a new factory in Texas"),
    ])
    assert counts["valid_and_recent"] == 3 and counts["after_exact_dedup"] == 2
    merged = next(a for a in clean if "profit" in a.title)
    assert merged.duplicate_count == 2
    assert not merged.title.endswith("Reuters")  # publisher suffix stripped


def test_near_duplicates_are_clustered_but_distinct_stories_survive():
    clean, counts = preprocess([
        article(1, "Acme Corp to acquire Bolt Systems for $4.2 billion"),
        article(2, "Acme agrees to acquire Bolt Systems in $4.2 billion deal", publisher="CNBC"),
        article(3, "Acme Corp recalls 10,000 scooters over battery fault"),
    ])
    assert counts["relevant"] == 3 and counts["after_near_dedup"] == 2
    deal = next(a for a in clean if "Bolt" in a.title)
    assert deal.duplicate_count == 2 and len({deal.publisher, *deal.also_reported_by}) == 2


def test_irrelevant_old_noisy_and_undated_articles_are_dropped():
    clean, counts = preprocess([
        article(1, "Acme Corp launches a new cloud product"),
        article(2, "Stocks end the week higher as traders eye earnings"),  # does not mention the company
        article(3, "Acme Corp announced a share buyback last year", days_ago=90),  # outside the window
        article(4, "ACME Oct 2026 175.000 call stock price, news, quote and history"),  # quote page
        NewsArticle(id="A005", title="Acme Corp names a new chief financial officer"),  # no date
        article(6, "Acme soars"),  # too short to be informative
        article(7, "Acme Corporation $ACME Stock Acquired by Example Capital Partners"),  # templated holdings note
        article(8, "REG - Other PLC Acme Corporation - Holding(s) in Company"),  # exchange filing notice
    ])
    assert [a.id for a in clean] == ["A001"]
    assert counts == {"ingested": 8, "valid_and_recent": 2, "after_exact_dedup": 2, "relevant": 1, "after_near_dedup": 1, "selected_for_llm": 1}


def test_noise_filter_keeps_real_headlines_that_resemble_templates():
    clean, _ = preprocess([
        article(1, "Acme Corp stock price target raised by Example Bank to $200"),
        article(2, "Acme Corp agrees to be acquired by Bolt Systems"),
    ])
    assert {a.id for a in clean} == {"A001", "A002"}


def test_cap_prefers_widely_covered_then_recent_stories():
    clean, counts = preprocess([
        article(1, "Acme Corp wins a defense contract worth billions", days_ago=5),
        article(2, "Acme Corp wins a defense contract worth billions", days_ago=5, publisher="AP"),
        article(3, "Acme Corp launches a new cloud product", days_ago=1),
        article(4, "Acme Corp names a new chief financial officer", days_ago=3),
    ], max_news_articles=2)
    assert counts["selected_for_llm"] == 2
    assert [a.id for a in clean] == ["A001", "A003"]


def test_relevance_matches_ticker_or_company_name_but_not_feed_tags():
    assert is_relevant(article(1, "Why ACME shares jumped today"), "ACME", "Acme Corporation")
    assert is_relevant(article(2, "Acme unveils new robot line"), "ACME", "Acme Corporation")
    assert not is_relevant(article(3, "Rival firm unveils robot line"), "ACME", "Acme Corporation")


def test_relevance_handles_brands_joined_names_and_generic_first_words():
    assert is_relevant(article(1, "Google unveils a new Gemini model"), "GOOGL", "Alphabet Inc.")
    assert is_relevant(article(2, "ExxonMobil signs a new offshore deal"), "XOM", "Exxon Mobil Corporation")
    assert is_relevant(article(3, "Meta's ad revenue beats estimates"), "META", "Meta Platforms, Inc.")
    assert not is_relevant(article(4, "Central bank holds rates steady again"), "BAC", "Bank of America Corporation")


def test_single_article_needs_no_clustering():
    only = [article(1, "Acme Corp launches a new cloud product")]
    assert cluster_near_duplicates(only, 0.45) == only


def test_classify_accepts_valid_labels_and_marks_invalid_ones_unknown():
    llm = fake_llm(
        {"event_type": "Earnings", "relevance": "high", "impact": "positive", "time_horizon": "short-term"},
        {"event_type": "gossip", "relevance": "very high", "impact": "positive", "time_horizon": "short_term"},
        None,  # the model did not return JSON at all
    )
    articles = [article(n, f"Acme Corp headline number {n}") for n in (1, 2, 3)]
    first, second, third = classify_news(articles, llm, "Acme Corporation", "ACME")
    assert (first.event_type, first.relevance, first.time_horizon) == ("earnings", "high", "short_term")
    assert (second.event_type, second.relevance, second.impact) == (UNKNOWN, UNKNOWN, "positive")
    assert (third.event_type, third.relevance, third.impact) == (UNKNOWN, UNKNOWN, UNKNOWN)


def test_extract_skips_low_relevance_and_drops_ungrounded_values():
    classified = [
        article(1, "Acme Corp to acquire Bolt Systems for $4.2 billion", relevance="high", impact="positive", event_type="m_and_a"),
        article(2, "Five stocks to watch including Acme", relevance="low", impact="neutral", event_type="market_commentary"),
    ]
    llm = fake_llm({
        "event": "Acme Corp agreed to acquire Bolt Systems for $4.2 billion.",
        "entities": ["Acme Corp", "Bolt Systems", "Goldman Sachs"],  # Goldman Sachs is not in the headline
        "figures": ["$4.2 billion", "15%"],  # 15% is not in the headline
        "dates": ["Q3"],  # not in the headline
        "risks": ["Integration risk"], "catalysts": "not a list",
    })
    events = extract_news_events(classified, llm, "Acme Corporation", "ACME")
    assert len(events) == 1
    event = events[0]
    assert event.entities == ["Acme Corp", "Bolt Systems"]
    assert event.figures == ["$4.2 billion"] and event.dates_mentioned == []
    assert event.risks == ["Integration risk"] and event.catalysts == []
    assert (event.source, event.event_date, event.extraction_method) == ("Reuters", "2026-10-09", "llm")


def test_extract_keeps_widely_covered_stories_the_model_labelled_low():
    classified = [
        article(1, "Acme Corp barred from a federal program", relevance="low", duplicate_count=3),
        article(2, "Five stocks to watch including Acme", relevance="low"),
    ]
    events = extract_news_events(classified, fake_llm({"event": "Acme was barred."}), "Acme Corporation", "ACME")
    assert [e.article_id for e in events] == ["A001"]


def test_extract_falls_back_to_headline_when_model_output_is_unusable():
    classified = [article(1, "Acme Corp launches a new cloud product", relevance="medium")]
    event = extract_news_events(classified, fake_llm(None), "Acme Corporation", "ACME")[0]
    assert event.event == "Acme Corp launches a new cloud product"
    assert event.extraction_method == "headline_fallback"


def _events(count=2):
    classified = [article(n, f"Acme Corp event headline number {n}", relevance="high", impact="positive") for n in range(1, count + 1)]
    return extract_news_events(classified, fake_llm({"event": "Acme did something."}), "Acme Corporation", "ACME")


def test_summary_keeps_valid_citations_and_removes_invented_ones():
    llm = fake_llm("Acme did something [E1]. It also did more [E2][E9].")
    summary = summarize_news(_events(), llm, "Acme Corporation", "ACME")
    assert summary["cited_event_ids"] == ["E1", "E2"] and summary["method"] == "llm"
    assert "E9" not in summary["summary"]


def test_summary_without_citations_falls_back_to_the_events():
    summary = summarize_news(_events(), fake_llm("Acme had a great month."), "Acme Corporation", "ACME")
    assert summary["method"] == "deterministic_fallback"
    assert summary["cited_event_ids"] == ["E1", "E2"] and "[E1]" in summary["summary"]


def test_summary_with_no_events_makes_no_llm_claim():
    summary = summarize_news([], fake_llm("should not be used"), "Acme Corporation", "ACME")
    assert summary["method"] == "no_events" and "No material recent news" in summary["summary"]


def _feed(*titles):
    return {
        "articles": [
            {"title": t, "publisher": "Reuters", "url": f"https://example.com/{i}", "published_at": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(), "source_feed": "test"}
            for i, t in enumerate(titles)
        ],
        "feed_errors": {},
    }


def test_news_agent_runs_all_five_stages_and_keeps_their_outputs():
    tools = StubTools({"get_company_news": _feed(
        "Acme Corp to acquire Bolt Systems for $4.2 billion",
        "Acme Corp to acquire Bolt Systems for $4.2 billion",
        "Stocks end the week higher as traders eye earnings",
    )})
    llm = fake_llm(  # one answer per stage: classify, extract, summarize
        {"event_type": "m_and_a", "relevance": "high", "impact": "positive", "time_horizon": "long_term"},
        {"event": "Acme Corp agreed to acquire Bolt Systems.", "entities": ["Bolt Systems"], "risks": ["Integration risk"]},
        "Acme agreed to acquire Bolt Systems [E1].",
    )
    result = run_news_agent(SubagentInput(symbol="ACME", profile={"name": "Acme Corporation"}), tools, llm)
    assert result.status == "success"
    stages = result.visualization_data["stages"]  # the chain's state keeps the output of every stage
    assert {"raw", "clean", "classified", "events", "summary"} <= set(stages)
    assert (len(stages["raw"]), len(stages["clean"]), len(stages["events"])) == (3, 1, 1)
    assert result.visualization_data["funnel"] == {
        "ingested": 3, "valid_and_recent": 3, "after_exact_dedup": 2, "relevant": 1, "after_near_dedup": 1,
        "selected_for_llm": 1, "material_events": 1,
    }
    assert result.findings[0].evidence_ids == ["NEWS-1"]
    assert result.evidence[0].source == "Reuters" and result.evidence[0].source_url
    assert result.risks == ["Integration risk [E1]"]


def test_news_agent_fails_cleanly_when_no_feed_is_available():
    result = run_news_agent(SubagentInput(symbol="ACME"), StubTools(), fake_llm())
    assert result.status == "failed"
    assert result.warnings and result.limitations and not result.evidence

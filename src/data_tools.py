"""Data connectors. Each returns normalized, validated data and raises DataToolError otherwise.

Sources (none need an API key):
- Yahoo Finance via yfinance: company profile, prices, financial statements, news
- Google News RSS: second news feed, so duplicates across feeds can be detected
- FRED: macroeconomic series
- SEC EDGAR: recent filings

Every DataFrame carries ``attrs`` with its source and retrieval timestamp.
"""

from __future__ import annotations

import io
import logging
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import pandas as pd
import requests
import yfinance as yf
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_fixed

from .config import settings
from .utils import disk_cache, utc_now, utc_now_iso

# yfinance logs expected failures (e.g. unknown symbols) itself; we report them through tool results.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

SOURCE_YAHOO = "Yahoo Finance (yfinance)"
SOURCE_GOOGLE_NEWS = "Google News RSS"
SOURCE_FRED = "FRED, Federal Reserve Bank of St. Louis"
SOURCE_SEC = "SEC EDGAR"
MIN_PRICE_ROWS = 30
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "InvestmentResearchAgent research@example.com")

# Normalized column name -> Yahoo Finance line item, per statement.
LINE_ITEMS = {
    "income_stmt": {
        "revenue": "Total Revenue",
        "gross_profit": "Gross Profit",
        "operating_income": "Operating Income",
        "net_income": "Net Income",
    },
    "balance_sheet": {
        "total_debt": "Total Debt",
        "stockholders_equity": "Stockholders Equity",
        "cash": "Cash And Cash Equivalents",
    },
    "cashflow": {
        "operating_cash_flow": "Operating Cash Flow",
        "free_cash_flow": "Free Cash Flow",
    },
}


class DataToolError(Exception):
    """Raised when a data source returns nothing usable. Never retried, never cached."""


# Retry transient provider errors. A DataToolError means the data does not exist, so retrying cannot help.
_retry = retry(
    stop=stop_after_attempt(settings.max_retries), wait=wait_fixed(1),
    retry=retry_if_not_exception_type(DataToolError), reraise=True,
)


def normalize_symbol(symbol: str) -> str:
    """Sanitize a user-provided ticker before it reaches any provider."""
    cleaned = str(symbol).strip().upper()
    if not re.fullmatch(r"[A-Z0-9.\-^=]{1,12}", cleaned):
        raise DataToolError(f"Invalid symbol: {symbol!r}")
    return cleaned


def short_company_name(name: str) -> str:
    """'Microsoft Corporation' -> 'Microsoft'; used for news queries and relevance matching."""
    suffix = r"(,?\s+(&\s+co\.?|corporation|corp\.?|incorporated|inc\.?|ltd\.?|limited|plc|company|co\.?|holdings|group|n\.v\.|s\.a\.))+$"
    short = re.sub(suffix, "", name.strip(), flags=re.IGNORECASE)
    short = re.sub(r"^the\s+", "", short, flags=re.IGNORECASE)
    return re.sub(r"\.com$", "", short, flags=re.IGNORECASE).strip(" ,") or name


# Brands that headlines use instead of the legal company name.
BRAND_ALIASES = {"GOOGL": ("Google",), "GOOG": ("Google",), "META": ("Facebook", "Instagram")}
GENERIC_NAME_WORDS = {"bank", "first", "american", "united", "general", "national", "international", "new"}


def company_terms(symbol: str, company_name: str) -> list[str]:
    """Names to look for in news about the company: short name, its first word, and known brands.

    Headlines usually shorten the name ("JPMorgan", "Meta"); a first word too generic to
    identify a company ("Bank", "General") is left out.
    """
    short_name = short_company_name(company_name)
    first_word = short_name.split()[0]
    terms = [short_name, *BRAND_ALIASES.get(symbol, ())]
    if first_word != short_name and first_word.lower() not in GENERIC_NAME_WORDS:
        terms.append(first_word)
    return terms


# --------------------------------------------------------------------- profile
@disk_cache(ttl_hours=settings.statement_cache_hours)
@_retry
def get_company_profile(symbol: str) -> dict:
    """Validate the ticker and return basic company identity."""
    symbol = normalize_symbol(symbol)
    info = yf.Ticker(symbol).info or {}
    if not info.get("quoteType") or not (info.get("longName") or info.get("shortName")):
        raise DataToolError(f"Unknown symbol: {symbol}")
    return {
        "symbol": symbol,
        "name": info.get("longName") or info.get("shortName"),
        "quote_type": info.get("quoteType"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "country": info.get("country"),
        "currency": info.get("currency"),
        "financial_currency": info.get("financialCurrency"),
        "exchange": info.get("exchange"),
        "market_cap": info.get("marketCap"),
        "website": info.get("website"),
        "source": SOURCE_YAHOO,
        "retrieved_at": utc_now_iso(),
    }


@disk_cache(ttl_hours=settings.statement_cache_hours)
@_retry
def resolve_symbol(query: str) -> dict:
    """Find the stock ticker for a company name or ticker as a user would write it ('Google', 'tsla')."""
    quotes = yf.Search(query.strip(), max_results=8, timeout=settings.request_timeout).quotes or []
    match = next((quote for quote in quotes if quote.get("quoteType") == "EQUITY"), None)
    if match is None:
        raise DataToolError(f"No listed company found for {query!r}")
    return {
        "symbol": match["symbol"], "name": match.get("longname") or match.get("shortname"),
        "exchange": match.get("exchange"), "source": SOURCE_YAHOO, "retrieved_at": utc_now_iso(),
    }


# ---------------------------------------------------------------------- prices
def normalize_prices(raw: pd.DataFrame, symbol: str, period: str) -> pd.DataFrame:
    """Validate raw OHLCV data and return a clean, date-indexed DataFrame."""
    if raw is None or raw.empty or "Close" not in raw.columns:
        raise DataToolError(f"No price data returned for {symbol}")
    prices = raw[[c for c in ("Open", "High", "Low", "Close", "Volume") if c in raw.columns]].copy()
    prices.columns = [c.lower() for c in prices.columns]
    # Exchange-local timestamps -> plain calendar dates, so stock and benchmark align on a join.
    prices.index = pd.DatetimeIndex(pd.to_datetime(prices.index).tz_localize(None).normalize(), name="date")
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    valid = prices["close"].notna() & (prices["close"] > 0)
    prices = prices[valid]
    if len(prices) < MIN_PRICE_ROWS:
        raise DataToolError(f"Only {len(prices)} valid price rows for {symbol}; need at least {MIN_PRICE_ROWS}")
    prices.attrs = {
        "symbol": symbol,
        "period": period,
        "source": SOURCE_YAHOO,
        "retrieved_at": utc_now_iso(),
        "rows_dropped": int((~valid).sum()),
        "price_basis": "split- and dividend-adjusted",
    }
    return prices


@disk_cache(ttl_hours=settings.price_cache_hours)
@_retry
def get_price_history(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Daily adjusted OHLCV history for a stock."""
    symbol = normalize_symbol(symbol)
    raw = yf.Ticker(symbol).history(
        period=period, interval="1d", auto_adjust=True, timeout=settings.request_timeout
    )
    return normalize_prices(raw, symbol, period)


def get_benchmark_history(benchmark: str = "SPY", period: str = "1y") -> pd.DataFrame:
    """Daily adjusted history for the benchmark (an ETF or index ticker)."""
    return get_price_history(benchmark, period)


# ------------------------------------------------------------------ financials
def normalize_statements(raw: dict[str, pd.DataFrame], symbol: str) -> pd.DataFrame:
    """Merge the three statements into one DataFrame: one row per fiscal period, oldest first."""
    columns = {}
    for statement, items in LINE_ITEMS.items():
        frame = raw.get(statement)
        for name, label in items.items():
            if frame is not None and label in frame.index:
                columns[name] = pd.to_numeric(frame.loc[label], errors="coerce")
            else:
                columns[name] = pd.Series(dtype="float64")
    financials = pd.DataFrame(columns)
    financials.index = pd.DatetimeIndex(pd.to_datetime(financials.index), name="period_end")
    # Statements can disagree on how many periods they report; keep periods that have revenue.
    financials = financials.sort_index().dropna(subset=["revenue"])
    if len(financials) < 2:
        raise DataToolError(f"Fewer than two fiscal periods with revenue for {symbol}")
    financials.attrs = {
        "symbol": symbol,
        "source": SOURCE_YAHOO,
        "retrieved_at": utc_now_iso(),
        "missing_items": [c for c in financials.columns if financials[c].isna().all()],
    }
    return financials


@disk_cache(ttl_hours=settings.statement_cache_hours)
@_retry
def get_financial_statements(symbol: str) -> pd.DataFrame:
    """Key income-statement, balance-sheet, and cash-flow line items per fiscal year."""
    symbol = normalize_symbol(symbol)
    ticker = yf.Ticker(symbol)
    return normalize_statements({statement: getattr(ticker, statement) for statement in LINE_ITEMS}, symbol)


# ------------------------------------------------------------------------ news
def _fetch_yahoo_news(symbol: str) -> list[dict]:
    items = yf.Search(symbol, news_count=25, timeout=settings.request_timeout).news or []
    return [
        {
            "title": item.get("title", ""),
            "summary": item.get("summary", ""),
            "publisher": item.get("publisher", ""),
            "url": item.get("link", ""),
            "published_at": datetime.fromtimestamp(item["providerPublishTime"], timezone.utc).isoformat()
            if item.get("providerPublishTime")
            else None,
            "source_feed": "yahoo_finance",
        }
        for item in items
    ]


def _fetch_google_news(symbol: str, company_name: str, lookback_days: int) -> list[dict]:
    names = [short_company_name(company_name), *BRAND_ALIASES.get(symbol, ()), symbol]
    query = " OR ".join(f'"{name}"' for name in names) + f" when:{lookback_days}d"
    response = requests.get(
        "https://news.google.com/rss/search",
        params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=settings.request_timeout,
    )
    response.raise_for_status()
    articles = []
    for item in ET.fromstring(response.content).iter("item"):
        published = item.findtext("pubDate")
        articles.append(
            {
                "title": item.findtext("title", ""),
                "summary": "",  # the feed's description only repeats the headline
                "publisher": item.findtext("source", ""),
                "url": item.findtext("link", ""),
                "published_at": parsedate_to_datetime(published).astimezone(timezone.utc).isoformat()
                if published
                else None,
                "source_feed": "google_news",
            }
        )
    return articles


@disk_cache(ttl_hours=settings.news_cache_hours)
def get_company_news(symbol: str, company_name: str, lookback_days: int = 30) -> dict:
    """Recent headlines from both feeds. One failing feed is tolerated; both failing is an error."""
    symbol = normalize_symbol(symbol)
    feeds = {
        "yahoo_finance": lambda: _fetch_yahoo_news(symbol),
        "google_news": lambda: _fetch_google_news(symbol, company_name, lookback_days),
    }
    articles, feed_counts, feed_errors = [], {}, {}
    for name, fetch in feeds.items():
        try:
            items = _retry(fetch)()
            feed_counts[name] = len(items)
            articles.extend(items)
        except Exception as exc:
            feed_errors[name] = f"{type(exc).__name__}: {exc}"
    if not articles:
        raise DataToolError(f"No news retrieved for {symbol}. Feed errors: {feed_errors or 'none (empty feeds)'}")
    return {
        "articles": articles,
        "feed_counts": feed_counts,
        "feed_errors": feed_errors,
        "source": f"{SOURCE_YAHOO}; {SOURCE_GOOGLE_NEWS}",
        "retrieved_at": utc_now_iso(),
    }


# ------------------------------------------------------- optional: macro, filings
@disk_cache(ttl_hours=settings.statement_cache_hours)
@_retry
def get_macro_series(series_id: str, lookback_days: int = 365) -> pd.DataFrame:
    """One FRED series (e.g. DGS10, DCOILWTICO, CPIAUCSL) as a date-indexed 'value' column."""
    if not re.fullmatch(r"[A-Z0-9]{2,20}", series_id):
        raise DataToolError(f"Invalid FRED series id: {series_id!r}")
    start = (utc_now() - timedelta(days=lookback_days)).date().isoformat()
    response = requests.get(
        "https://fred.stlouisfed.org/graph/fredgraph.csv",
        params={"id": series_id, "cosd": start},
        timeout=settings.request_timeout,
    )
    response.raise_for_status()
    try:
        frame = pd.read_csv(io.StringIO(response.text))
        frame.columns = ["date", "value"]
    except Exception as exc:
        raise DataToolError(f"FRED returned no usable data for {series_id}") from exc
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")  # FRED marks holidays with '.'
    series = frame.dropna().assign(date=lambda d: pd.to_datetime(d["date"])).set_index("date")
    if series.empty:
        raise DataToolError(f"FRED series {series_id} has no observations since {start}")
    series.attrs = {"series_id": series_id, "source": SOURCE_FRED, "retrieved_at": utc_now_iso()}
    return series


@disk_cache(ttl_hours=settings.statement_cache_hours)
@_retry
def get_sec_filings(symbol: str) -> pd.DataFrame:
    """The company's most recent annual and quarterly SEC filings (10-K, 10-Q), newest first, with links to the documents."""
    symbol = normalize_symbol(symbol)
    headers = {"User-Agent": SEC_USER_AGENT}
    tickers = requests.get(
        "https://www.sec.gov/files/company_tickers.json", headers=headers, timeout=settings.request_timeout
    )
    tickers.raise_for_status()
    cik = next((row["cik_str"] for row in tickers.json().values() if row["ticker"] == symbol), None)
    if cik is None:
        raise DataToolError(f"{symbol} is not an SEC registrant")
    submissions = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik:010d}.json", headers=headers, timeout=settings.request_timeout
    )
    submissions.raise_for_status()
    recent = pd.DataFrame(submissions.json()["filings"]["recent"])
    filings = recent[recent["form"].isin(["10-K", "10-Q"])].head(4)
    if filings.empty:
        raise DataToolError(f"No 10-K or 10-Q filings found for {symbol}")
    accession = filings["accessionNumber"].str.replace("-", "")
    filings = pd.DataFrame(
        {
            "form": filings["form"],
            "filing_date": filings["filingDate"],
            "report_date": filings["reportDate"],
            "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/" + accession + "/" + filings["primaryDocument"],
        }
    ).reset_index(drop=True)
    filings.attrs = {"symbol": symbol, "cik": int(cik), "source": SOURCE_SEC, "retrieved_at": utc_now_iso()}
    return filings

# Engineer 2 Plan: Data Tools and Specialist Subagents

## Objective

Build reliable data connectors and the Market, Earnings, and News specialist subagents, including the complete news Prompt Chaining workflow.

## Main Deliverables

- Market and benchmark data tools.
- Financial-statement data tools.
- News retrieval tool.
- Optional filing and macro-context tools.
- Market Subagent.
- Earnings Subagent.
- News Subagent.
- Five-stage news Prompt Chain.
- Data validation, caching, and error handling.

## Five-Day Plan

### Day 1: Data Connectors

- Implement company-profile lookup.
- Implement stock and benchmark price retrieval.
- Implement financial-statement retrieval.
- Normalize returned formats into DataFrames or structured objects.
- Add retrieval timestamps and source metadata.

### Day 2: Market and Earnings Subagents

- Calculate returns, benchmark performance, and rolling volatility.
- Analyze revenue, profit, margin, and cash-flow trends.
- Return findings, evidence, risks, and limitations.
- Add unit tests for deterministic calculations.

### Day 3: News Ingestion and Preprocessing

- Retrieve recent company news.
- Clean text and normalize publication dates.
- Remove duplicates and low-relevance records.
- Cluster similar articles when practical.
- Capture article counts at each processing stage.

### Day 4: News Classification, Extraction, and Summary

- Classify event type, relevance, potential impact, and time horizon.
- Extract events, dates, entities, figures, risks, catalysts, and sources.
- Generate the final news summary from extracted evidence.
- Preserve intermediate outputs for notebook demonstration.

### Day 5: Reliability and Visualization Data

- Add caching, timeouts, retries, and graceful failures.
- Test at least three stock symbols.
- Prepare clean data for all required charts.
- Complete API and schema documentation.
- Integrate all subagents with Engineer 1's registry.

## Acceptance Criteria

- Market and financial tools return normalized, validated data.
- Each subagent follows the agreed output schema.
- All five news-chain stages run and display intermediate output.
- Duplicate and irrelevant articles are visibly reduced.
- Numeric calculations are reproducible and tested.
- Missing data produces a warning and limitation.
- Every major finding contains supporting evidence or a source reference.

## Dependencies

- Engineer 1 provides shared schemas and registry contracts.
- Engineer 3 provides visualization requirements and report evidence schema.

## Handoff Contract

Engineer 2 exposes stable Python functions for Engineer 1 to register and execute. Specialist outputs include summaries, findings, evidence, risks, limitations, and visualization-ready data.

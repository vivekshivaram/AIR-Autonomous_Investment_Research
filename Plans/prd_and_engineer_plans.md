# Investment Research Agent

## Product Requirements Document and Engineering Plans

## 1. Product Overview

### Product Title

**Agentic AI for Autonomous Investment Research**

### Product Description

A Python-based agentic AI system that accepts a stock symbol and research question, creates a research plan, selects relevant specialist subagents, gathers market, financial, and news data, and produces an evidence-backed investment research report.

The system evaluates its own report, improves weak sections, and stores brief lessons that can influence later runs.

> The output is intended for educational research and is not personalized investment advice.

## 2. Product Objectives

- Demonstrate **Prompt Chaining**, **Routing**, and **Evaluator–Optimizer** workflows.
- Demonstrate **Planning**, **Dynamic Tool Use**, **Self-Reflection**, and **Learning Across Runs**.
- Generate a structured report using market, earnings, and news evidence.
- Display intermediate decisions, tool calls, quality scores, and memory usage.
- Include clear comments, documentation, tables, and visualizations.
- Deliver an executable Python Jupyter Notebook supported by modular Python files.

## 3. Primary User Flow

1. User enters a stock symbol and research question.
2. Planner retrieves relevant lessons from previous runs.
3. Planner decomposes the request into research tasks.
4. Planner routes tasks to selected subagents.
5. Selected subagents retrieve and analyze data.
6. News Subagent runs the prompt chain.
7. Report Generator combines specialist evidence.
8. Evaluator scores the initial report.
9. Optimizer improves weak sections when required.
10. System displays the final report and visualizations.
11. Reflection is stored as memory for later runs.

## 4. System Components

### Planner and Coordinator

- Creates the research plan.
- Performs routing without requiring a separate Router Agent.
- Selects only the necessary subagents.
- Maintains shared execution state.
- Coordinates task execution and result aggregation.

### Market Subagent

- Retrieves historical stock and benchmark prices.
- Calculates returns and rolling volatility.
- Compares stock performance against a benchmark.
- Adds macroeconomic context only when relevant.

### Earnings Subagent

- Retrieves company financial statements.
- Analyzes revenue, profit, margins, and cash flow.
- Identifies financial strengths, risks, and limitations.

### News Subagent

Implements the required Prompt Chaining workflow:

1. Ingest news.
2. Preprocess and deduplicate articles.
3. Classify events.
4. Extract structured facts.
5. Summarize material developments.

### Evaluator–Optimizer

- Evaluates grounding, completeness, consistency, recency, risk coverage, clarity, and uncertainty.
- Produces structured feedback.
- Refines only weak sections.
- Stops after reaching the quality threshold or maximum refinement count.

### Memory

- Stores compact lessons in SQLite.
- Retrieves relevant lessons before planning.
- Influences later plans without being treated as current financial evidence.

## 5. Data Requirements

### Data Sources

- Yahoo Finance through `yfinance` for prices and financial statements.
- NewsAPI or Yahoo Finance news for recent articles.
- SEC EDGAR as an optional source for company filings.
- FRED as an optional source for conditional macroeconomic context.

### Expected Data Volume per Symbol

- Approximately 250 daily price observations for one year.
- Approximately 10 to 30 recent news articles.
- Approximately 3 to 5 years of annual or quarterly financial data.
- One or more SQLite records per completed agent run.

### Core Variables

- Date, open, high, low, close, adjusted close, and volume.
- Revenue, net income, margins, assets, liabilities, and cash flow.
- News title, description, source, publication date, category, relevance, and impact.
- Tool name, execution status, duration, and retrieval timestamp.
- Evaluation criteria, score, feedback, and refinement status.

## 6. Functional Requirements

### FR-1: Planning

The system shall generate an ordered research plan based on the stock symbol, research question, available tools, and relevant memory.

### FR-2: Routing

The Planner shall assign each research task to one or more relevant subagents and record the selection reason. It shall also record skipped subagents and reasons.

### FR-3: Dynamic Tool Use

The system shall select tools based on the plan and task requirements rather than calling every tool unconditionally.

### FR-4: Prompt Chaining

The News Subagent shall execute and display all five stages: Ingest, Preprocess, Classify, Extract, and Summarize.

### FR-5: Specialist Analysis

The Market, Earnings, and News Subagents shall return structured findings, evidence, risks, limitations, and status.

### FR-6: Report Generation

The system shall produce a report containing an executive summary, specialist findings, risks, catalysts, confidence, limitations, and sources.

### FR-7: Evaluator–Optimizer

The system shall score the initial report, provide feedback, refine weak sections, and show before-and-after results.

### FR-8: Self-Reflection

The system shall document successful steps, failures, evidence gaps, quality issues, and lessons after each run.

### FR-9: Learning Across Runs

The system shall save useful lessons and demonstrate that a later run retrieves and applies at least one lesson.

### FR-10: Visualization

The notebook shall visualize market performance, financial trends, news processing, routing decisions, evaluation improvement, and repeated-run quality.

## 7. Non-Functional Requirements

- The main notebook must run from top to bottom.
- Python functions must include useful docstrings and reasoning comments.
- Agent outputs must use consistent structured schemas.
- Tool failures must produce warnings instead of fabricated results.
- API keys must be loaded from environment variables.
- Refinement cycles must be limited.
- Sources and retrieval timestamps must be preserved.
- The implementation must be understandable to an evaluator without inspecting every source file.

## 8. Success Criteria

- All three workflow patterns are executed and demonstrated.
- All four agent functions are executed and demonstrated.
- At least four routing examples are shown, including one multi-subagent route.
- The tool-call log proves conditional tool selection.
- The news pipeline displays intermediate output from all five stages.
- The evaluator identifies meaningful weaknesses in the initial report.
- The refined report improves the quality score.
- A second run retrieves memory and changes planning or execution.
- Required visualizations are present and explained.
- The notebook completes without manual code changes.

## 9. Out of Scope

- Automated trading or order execution.
- Personalized buy or sell recommendations.
- Portfolio optimization.
- Real-time streaming data.
- Full production authentication and multi-user access.
- Large-scale vector database infrastructure.
- Model fine-tuning.

---

# Engineer 1 Plan: Planner and Agent Orchestration

## Objective

Build the central Planner and Coordinator that decomposes requests, routes tasks, executes selected subagents, and maintains observable agent state.

## Main Deliverables

- Shared agent state and structured interfaces.
- Research-plan generator.
- Task routing and subagent-selection logic.
- Subagent registry and execution coordinator.
- Tool registry integration.
- Routing and tool-call logs.
- Evidence aggregation interface.
- End-to-end notebook orchestration.

## Five-Day Plan

### Day 1: Models and Interfaces

- Define `AgentState`, `ResearchPlan`, `ResearchStep`, and `RoutingDecision`.
- Define standard subagent input and output schemas.
- Define status, warning, and error structures.
- Add architecture and execution-flow documentation.

### Day 2: Planner

- Implement research-question interpretation.
- Implement task decomposition.
- Generate ordered steps, dependencies, and completion criteria.
- Include relevant stored lessons in planner input.
- Validate generated plans before execution.

### Day 3: Routing and Registries

- Implement routing inside the Planner.
- Create Market, Earnings, and News subagent registry.
- Create approved tool registry.
- Record selected and skipped subagents with reasons.
- Demonstrate single-agent and multi-agent routing cases.

### Day 4: Orchestration

- Execute selected subagents in the correct order.
- Capture tool-call status, duration, and result summaries.
- Aggregate specialist evidence.
- Handle partial failures and skipped tasks.
- Connect orchestrator output to the Report Generator.

### Day 5: Integration and Testing

- Integrate the complete notebook flow.
- Test narrow and comprehensive research questions.
- Verify that routing is conditional rather than fixed.
- Add plan, routing, and execution visualizations.
- Finalize comments, docstrings, and Engineer 1 tests.

## Acceptance Criteria

- Planner creates a structured and valid research plan.
- Narrow questions select only relevant subagents.
- Comprehensive research selects all required subagents.
- Every selected and skipped subagent has a reason.
- Tool calls and subagent executions are logged.
- Partial failures do not terminate the entire workflow.
- Planner output is visible and understandable in the notebook.

## Dependencies

- Engineer 2 provides callable data tools and specialist interfaces.
- Engineer 3 provides Report Generator, Evaluator, and Memory interfaces.

## Handoff Contract

Engineer 1 passes aggregated specialist results to Engineer 3. Engineer 1 receives tool and subagent functions from Engineer 2 and memory results from Engineer 3.

---

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

---

# Engineer 3 Plan: Report, Evaluation, Memory, and Visualization

## Objective

Build the final research-report pipeline, evaluator–optimizer workflow, self-reflection, persistent memory, and project visualizations.

## Main Deliverables

- Evidence-backed Report Generator.
- Quality rubric and evaluator.
- Targeted optimizer and stopping conditions.
- Run reflection.
- SQLite memory.
- Second-run learning demonstration.
- Financial and agent-workflow visualizations.
- Notebook presentation and final documentation.

## Five-Day Plan

### Day 1: Report and Evidence Model

- Define the final report structure.
- Define evidence and citation fields.
- Implement the initial Report Generator.
- Separate observed facts from interpretations.
- Add confidence and limitation sections.

### Day 2: Evaluator

- Implement the quality rubric.
- Score grounding, completeness, consistency, recency, risk coverage, clarity, and uncertainty.
- Generate structured issues and corrective actions.
- Display the initial score and evaluation details.

### Day 3: Optimizer and Reflection

- Implement targeted section refinement.
- Add quality threshold and maximum-cycle stopping conditions.
- Re-evaluate refined reports.
- Implement structured run reflection.
- Preserve evaluation history.

### Day 4: Persistent Memory

- Create the SQLite memory schema.
- Save compact symbol-specific and general lessons.
- Retrieve relevant memory before planning.
- Demonstrate that a second run uses a saved lesson.
- Prevent stored memory from being treated as current evidence.

### Day 5: Visualization and Submission Quality

- Implement market, financial, news, routing, and evaluation charts.
- Add written interpretation below each chart.
- Show quality changes across refinement cycles and repeated runs.
- Review notebook comments, Markdown, labels, and execution order.
- Export the executed notebook to HTML as a backup artifact.

## Acceptance Criteria

- Initial and refined reports are both visible.
- The evaluator provides criterion-level scores and useful feedback.
- The optimizer corrects identified weaknesses.
- The refined score improves or the lack of improvement is explained.
- Reflection records successes, failures, gaps, and lessons.
- Memory persists between runs and affects a later plan or workflow.
- Charts have titles, labels, units, legends, and short interpretations.

## Dependencies

- Engineer 1 provides aggregated state, plan, routing, and tool logs.
- Engineer 2 provides evidence and visualization-ready specialist outputs.

## Handoff Contract

Engineer 3 receives the aggregated `AgentState`, produces the report and evaluation history, writes reflection to memory, and returns the final report and visualization outputs to the notebook.

---

# Shared Integration Plan

## Shared Interface Freeze

By the end of Day 1, all engineers agree on:

- Agent state fields.
- Research-plan schema.
- Routing-decision schema.
- Tool-result schema.
- Subagent-result schema.
- Evidence-item schema.
- Evaluation-result schema.
- Memory-item schema.

## Daily Integration

- Integrate all completed branches at the end of each day.
- Run one smoke test using the same stock symbol.
- Log interface mismatches immediately.
- Keep notebook cells small and independently testable.
- Avoid changing shared schemas without notifying all engineers.

## Final Deliverables

- `investment_research_agent.ipynb`
- `architecture.md`
- `prd_and_engineer_plans.md`
- Modular Python source files under `src/`
- Automated tests under `tests/`
- `requirements.txt`
- `.env.example`
- `README.md`
- SQLite memory database created during execution
- Optional executed notebook HTML or short report

## Definition of Done

- All three workflows are implemented and visibly demonstrated.
- All four agent functions are implemented and visibly demonstrated.
- The notebook runs from start to finish in a clean environment.
- Data or API failures are handled without fabricated output.
- Intermediate agent decisions are inspectable.
- Comments explain assumptions and non-obvious decisions.
- Visualizations support the analysis and scoring criteria.
- Each engineer's component passes agreed acceptance tests.

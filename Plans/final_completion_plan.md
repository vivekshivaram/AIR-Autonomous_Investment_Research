# Final Completion Plan: Engineer 3 and Remaining Gaps

## Purpose

This plan finishes Engineer 3's work (report, evaluator-optimizer, reflection, memory, visualizations, notebook and documentation) and closes every gap between the course requirements, the PRD and `main`. The project is complete when every verification gate at the end passes.

## Scope and Decisions

- Base: `main` at `2b906a2`. Work branch: `feature/engineer-3-report-eval-memory`, merged through one pull request.
- Engineer 3 owns every item in this plan, including the gaps in Engineer 1 and Engineer 2 areas.
- Commit rule: Engineer 3 commits only add new files. Changes to existing files (teammates' modules, the shared notebook, the docs) go into separate follow-up commits grouped by owner, so each teammate can review their part on its own.
- LLM: Gemini 3.8 Flash (`gemini-3.8-flash`) through Google AI Studio's OpenAI-compatible endpoint. The key lives in a local `.env` as `GEMINI_API_KEY` and never enters git, notebook output or logs. Groq, Hugging Face and local Qwen stay selectable. The AI Studio project is on Tier 3, so no request throttle is needed; the client keeps the SDK's retries for `429` errors.
- Grading weights: agent functions 33.8%, workflow patterns 33.8%, comments and visualizations 32.4%. Every item below maps to one of these.
- Rubric: eight criteria, following the README, which scores risk coverage and catalyst coverage separately. `engineer_3_plan.md` is aligned in the documentation commit.
- Evaluator: hybrid. Seven criteria are checked in code against the evidence. Clarity is judged by the LLM, with a heuristic fallback when no LLM is available.
- The output stays educational research, not personalised investment advice.

## Requirement Coverage

| Requirement | Grade area | Where | Status on `main` |
|---|---|---|---|
| Plans research steps | Agent functions | `planner.py` | Done (Engineer 1) |
| Uses tools dynamically (APIs, datasets, retrieval) | Agent functions | `tool_registry.py`, `data_tools.py`, memory tools | APIs done; retrieval in this plan (A6, B2) |
| Self-reflects on output quality | Agent functions | `evaluator.py`, `reflection.py` | This plan (A3, A5) |
| Learns across runs | Agent functions | `memory.py` with planner hints | This plan (A6, A7) |
| Prompt chaining | Workflow patterns | `news_pipeline.py`, `news_agent.py` | Done (Engineer 2) |
| Routing | Workflow patterns | `planner.py` | Done (Engineer 1) |
| Evaluator-Optimizer | Workflow patterns | `report_generator.py`, `evaluator.py`, `optimizer.py` | This plan (A2 to A4) |
| Comments and docstrings | Comments and visualizations | all of `src/` | 80 of 165 public functions and classes have no docstring (B5) |
| Visualizations with interpretations | Comments and visualizations | `visualizations.py`, `orchestration_viz.py`, `evaluation_viz.py` | 10 charts in use, 4 required charts missing, no interpretations (A8, B3, B6) |
| Data sources | Technology | `data_tools.py` | Yahoo Finance, Google News, FRED and SEC EDGAR done. NewsAPI, Kaggle and Alpha Vantage are optional and not used |

## Part A: Engineer 3 Deliverables (new files only)

### A1. Report and evaluation schemas: `src/report_models.py`

Dataclasses with `to_dict` and `from_dict`, in the style of `models.py`. A separate file keeps `models.py` untouched.

- `ReportStatement`: `text`, `kind` (`fact` or `interpretation`), `evidence_ids`, `risk_ids`, `catalyst_ids`.
- `ReportSection`: `key`, `title`, `status` (`complete`, `out_of_scope` or `unavailable`), `statements`, `note`.
- `ResearchReport`: `symbol`, `question`, `version`, `generated_by` (`llm` or `template`), `sections`, `confidence` (level and reasons), `lessons_applied` (planning context only), `created_at`, `disclaimer`, plus `to_markdown()` and `cited_ids()`.
- `CriterionScore`: `name`, `weight`, `score` (0 to 100), `details`.
- `ReportIssue`: `criterion`, `section`, `severity`, `message`, `corrective_action`, `refs`.
- `EvaluationResult`: `cycle`, `report_version`, `overall_score`, `criteria`, `issues`, `weak_sections`, `decision`, `stop_reason`, `judge_mode`, `evaluated_at`.
- `RunReflection` and `Lesson` (see A5 and A6).

### A2. Report generator: `src/report_generator.py`, `prompts/report_generator.txt`

- Entry point `run_report_pipeline(state, *, llm=None, settings=None, tools=None) -> AgentState`, which `build_runtime()` already detects. Called without arguments, it creates its own LLM client and settings, so it works before the wiring commit (B2).
- Pipeline order: evidence pack, first draft, evaluator-optimizer loop (A4), reflection (A5), memory (A6), saved outputs (A9). A failure in any stage becomes a warning, and the aggregated evidence stays intact.
- Evidence pack from `state.aggregated`: findings with evidence IDs, numbered risks (`R1`...) and catalysts (`C1`...), limitations, coverage, data periods, and news event tags (`E1`...) mapped to their `NEWS-n` evidence IDs.
- The eleven README sections: executive summary, company overview, market performance, financial analysis, news and events, macro context (only when the macro tool ran), risks, catalysts, conflicting or missing evidence, confidence and limitations, sources.
- First draft: the LLM writes statements for each section under a JSON schema and may cite only IDs from the pack. The draft is a brief first pass of about four statements per section, and the evaluator decides what is missing. Citations of unknown IDs are removed, as in the news summarizer.
- Built in code rather than by the LLM: the company overview, out-of-scope notes for skipped specialists (with the routing reason), the confidence level (from coverage, evidence count and errors) and the sources list (source, URL, as-of date and retrieval time of every cited item).
- Template fallback without an LLM: findings become fact statements word for word.
- Lessons with a `report_guidance` hint are added to the prompt as writing guidance, never as facts.
- Markdown rendering: facts end with their `[ID]` citations, and interpretations are labelled.

### A3. Evaluator: `src/evaluator.py`, `prompts/evaluator.txt`

| Criterion | Weight | How it is scored |
|---|---|---|
| Evidence grounding | 20 | Share of fact statements that cite at least one valid evidence ID. Citations of unknown IDs or of memory count as failures; interpretations must be labelled |
| Completeness | 15 | Required sections present; each completed specialist's key findings cited; skipped or failed specialists explained; a price analysis names what drove the move or says the drivers were not analysed (a disclosed gap earns partial credit) |
| Numeric consistency | 15 | Every number in a fact statement matches a value in its cited evidence after unit normalisation (%, percentage points, bn, x). Years, dates, window lengths and IDs are ignored |
| Source recency | 10 | Cited evidence inside freshness limits (prices at most 7 days old, news inside the lookback window, latest fiscal year at most 15 months old) and the data dates stated in each section |
| Risk coverage | 10 | Share of specialist risks (`R1`...) covered in the risks section |
| Catalyst coverage | 10 | Share of specialist catalysts (`C1`...) covered; when news did not run, the report must say that catalysts were not assessed |
| Clarity | 10 | LLM judge scoring structure, plain language, concision and repetition from 1 to 5, scaled to 100; fallback heuristics on sentence length, duplicate sentences and section length |
| Uncertainty disclosure | 10 | Confidence level with reasons; every limitation and coverage gap from aggregation appears; interpretations labelled |

- Overall score: the weighted sum, out of 100.
- Decision: accept when the overall score is at least 80 (`Settings.quality_threshold`) and no criterion is below 60; otherwise refine. (The floor was 50 in the first version; see the implementation notes.)
- Every issue names its section and a concrete fix, for example "add risk R2 to the risks section, citing MKT-7" or "the 17.8% in financial analysis matches no cited evidence; cite ERN-2 or remove it".
- One `EvaluationResult` per cycle is appended to `state.evaluation_history`, with a snapshot of the report it scored.

### A4. Optimizer: `src/optimizer.py`, `prompts/optimizer.txt`

- Refines only the sections that have issues, worst first.
- The LLM receives the section, its issues and fixes, and the IDs it may cite. Statements that already passed every check are locked and put back if the rewrite drops what they cite.
- Fallback without an LLM: add the missing risks, catalysts and limitations from the pack, and drop numbers that match no evidence.
- Loop: evaluate the first draft, refine, evaluate again. Stop when the decision is accept, after `Settings.max_refinement_cycles` (2) cycles, or when a cycle gains less than 2 points. The best-scoring version becomes the final report and the stop reason is recorded, so a missing improvement is explained.
- State: `draft_report` holds the first draft, `final_report` the best version, `evaluation_history` every cycle.

### A5. Reflection: `src/reflection.py`

- Built in code from the state and the evaluation history: steps that worked, failed or skipped steps with reasons, tool problems, evidence gaps, first-draft issues fixed and still open, the score change, and lessons.
- Lesson rules:

| Trigger | Scope | Planner or writer hint |
|---|---|---|
| Price analysis ran without news, and the evaluator flagged unexplained price drivers | Symbol | `require_subagents: ["news_agent"]` |
| A selected specialist failed | General | `extra_criteria` for that specialist: state the coverage gap explicitly |
| Numeric or citation issues in the first draft | General | `report_guidance`: quote figures exactly and cite the ID beside each one |
| Risks or catalysts missing from the first draft | General | `report_guidance`: list every specialist risk and catalyst with its ID |
| Latest annual statements older than 9 months | Symbol | `extra_criteria` for `earnings_agent`: state the fiscal-year end |

- General lessons never add specialists, so a lesson learned on one symbol cannot change routing for another.

### A6. Memory: `src/memory.py`

- SQLite at `Settings.db_path` (`agent_memory.db`, already gitignored). `AGENT_MEMORY_DB` or `use_database(path, reset=False)` points it at another file.
- Tables:
  - `runs`: run ID, symbol, question, scope, selected specialists, first and final scores, refinement cycles, LLM provider and model, lessons applied, reflection JSON, timestamps.
  - `lessons`: lesson ID (`MEM-0001`...), scope, symbol, topic tags, text, hint JSON, source run, creation time, times retrieved, times applied, last use, active flag.
  - `lesson_usage`: run ID, lesson ID, retrieved, applied, timestamp.
- `store_lesson(...)` merges duplicates (same text, scope and symbol) and keeps at most 20 lessons per symbol.
- `retrieve_lessons(symbol, question)` returns the top five in the `MemoryItem` shape the planner already reads (`memory_id`, `lesson`, `scope`, `symbol`, `score`, `hint`), ranked by symbol match, topic overlap with the question, recency and past use. Every retrieval is recorded.
- `record_run(...)` runs after reflection and marks the lessons that the plan applied.
- Memory is never evidence. Lesson IDs are not evidence IDs, the evaluator fails any citation outside the evidence set, and tests check that memory never appears in aggregated evidence or in report sources.

### A7. Learning-across-runs demonstration

1. The notebook keeps its memory in its own file under `outputs/`. The file is cleared at the start of the notebook and again right before this demonstration, so every execution tells the same story. Normal runs keep using `agent_memory.db`.
2. Run 1: MSFT, "Analyze one-year price performance". Only the market specialist runs, the evaluator flags unexplained price drivers, and reflection stores the MSFT lesson.
3. The lesson rows are read back from SQLite through a new connection, which shows that they persist on disk.
4. Run 2: the same question on a fresh runtime. The plan adds the news specialist with the reason "Added by lesson MEM-0001", `lessons_applied` lists the lesson, the report now covers catalysts, and the two runs' scores are compared.
5. Optional run 3: a different symbol, where the general writing lessons improve the first draft. This shows memory changing execution as well as planning.

### A8. Evaluation and workflow charts: `src/evaluation_viz.py`

- Criterion scores before and after refinement (grouped bars with the threshold line).
- Overall score by refinement cycle (line with the threshold and the stop reason).
- Quality across repeated runs (first and final score for each run, with the lessons applied noted).
- Issue count by criterion, first draft against final report (optional).
- Agent workflow diagram for one run: which specialists ran, the refinement cycles, and memory reads and writes. This covers the "architecture and workflow diagrams" checklist item.
- Caption helpers that write a short interpretation from the data for every chart in the notebook, including the Engineer 1 and 2 charts, so the text stays correct when live data changes.

### A9. Run outputs: `src/run_outputs.py`

- `outputs/reports/initial_report.md` and `outputs/reports/final_report.md`, plus copies named by run ID.
- `outputs/charts/*.png`.
- `outputs/logs/latest_run.json`: plan, routing, tool calls, execution log, evaluations, reflection and memories used, without DataFrames or secrets.

### A10. Tests (offline, with a fake LLM and stub tools)

- `test_report_generator.py`: all sections present, every fact cites valid IDs, skipped and failed specialists explained, invalid citations removed, memory never cited.
- `test_evaluator.py`: score arithmetic, the threshold and floor decision, an uncited claim, a wrong number, a missing risk, stale evidence.
- `test_optimizer.py`: each stop rule, best version kept, locked statements preserved, the fallback adding missing risks.
- `test_reflection.py`: each lesson rule produces the right scope and hint.
- `test_memory.py`: schema, store and retrieve, ranking, duplicate merging, persistence across connections, usage records.
- `test_report_pipeline.py`: end to end with stubs; a second run applies the stored lesson and its plan changes.

## Part B: Follow-up Commits (existing files)

### B1. Gemini provider (Engineer 1: `llm_client.py`, `config.py`, `.env.example`; new `src/llm_cache.py`)

- `GeminiClient` on `https://generativelanguage.googleapis.com/v1beta/openai/`, configured by `GEMINI_API_KEY`, `GEMINI_MODEL=gemini-3.8-flash` and `LLM_PROVIDER=gemini`.
- No `temperature` is sent, because Gemini 3 and later removed the sampling parameters. Thinking depth is set through `reasoning_effort` (`GEMINI_REASONING_EFFORT`, starting at `low`).
- JSON: try `response_format` with the schema; if Gemini rejects it, retry with the schema in the prompt and parse the reply, as the Hugging Face client already does.
- Retry `429 RESOURCE_EXHAUSTED` with backoff (the SDK's retries, `GEMINI_MAX_RETRIES`). A request throttle was planned for the free tier and dropped because the project is on Tier 3.
- An on-disk cache of LLM responses keyed by model, messages and schema, so re-running the notebook does not spend quota and returns the same text.

### B2. Report pipeline and memory wiring (Engineer 1: `bootstrap.py`, `orchestrator.py`, `tests/test_orchestrator.py`)

- Pass the runtime's LLM, settings and tool registry into the report pipeline.
- Call memory retrieval and storage through `ToolSession`, so `memory_retrieval` and `memory_storage` appear in the tool-call log.
- The integration table reports the report pipeline and memory as `real`.

### B3. Chart fixes (Engineer 2: `visualizations.py`; Engineer 1: `orchestration_viz.py`)

- Add the daily-return distribution the README requires, computed from the existing performance data.
- Add missing axis labels and legends (routing matrix axes, tool-call status colours).
- Remove `plot_subagent_calls`, which has been unused since the LangGraph runner was dropped.

### B4. Restore Engineer 2's tests (Engineer 2: `tests/`)

- Port `test_market_agent.py`, `test_earnings_agent.py`, `test_news_pipeline.py` and `test_tools.py` from `feature/engineer-2-data-subagents`, about 45 tests.
- Replace the LangChain fakes in `helpers.py` with a plain fake LLM and stub tools, and drop the test for the removed local tool-calling model.

### B5. Docstrings and reasoning comments (Engineers 1 and 2: `src/`)

- Docstrings for the definitions a reader meets first: `build_runtime`, `Runtime`, `Planner.create_plan`, the shared models, the registries, the LLM clients and the stubs.
- Reasoning comments where the logic is not obvious. No behaviour changes.

### B6. Notebook (shared: `investment_research_agent.ipynb`)

- Fixes: two sections are numbered 5; the note after the comprehensive run describes the earlier narrow run; each of the six specialist charts appears twice; the hand-written `DEMO-1` lesson gives way to the real memory section.
- Final structure:
  0. Overview, how to run, architecture diagram, runtime status
  1. Planning
  2. Routing
  3. Dynamic tool use and orchestration
  4. Market and financial analysis
  5. News prompt chain, stage by stage
  6. Partial failure handling
  7. Initial report
  8. Evaluation
  9. Optimization and the refined report
  10. Self-reflection
  11. Learning across runs
  12. Summary and requirement checklist
- Every chart is followed by its generated caption and a short Markdown interpretation.
- Markdown cells in plain, human prose.
- One clean top-to-bottom execution in a fresh environment, with the outputs saved in the notebook.

### B7. Documentation (shared: `README.md`, `architecture.md`, `Plans/`)

- README: current status, a file tree that matches the repo, Gemini setup, the scoring checklist ticked with the notebook section that proves each item, and the output files.
- `architecture.md`: an Engineer 3 section (report pipeline, rubric, stop rules, memory schema, learning loop) with a Mermaid diagram.
- `engineer_3_plan.md` aligned to the eight criteria, and the progress list below ticked.
- Optional HTML export of the executed notebook (team plan, Day 5).

## Implementation Order

Development order differs from commit order in one place: live runs need the Gemini client (B1) from the start, but it is committed after the Engineer 3 commits.

0. Setup: create the branch, install `openai` and `pytest` into `Course5/.venv`, add the Gemini key to `.env`, draft B1 and smoke-test it.
1. A1 schemas and A2 report generator, with tests.
2. A3 evaluator, with tests.
3. A4 optimizer and A5 reflection, with tests.
4. A6 memory, A7 learning demo and the end-to-end test.
5. A8 charts and A9 outputs.
6. B2 to B5: wiring, chart fixes, restored tests, docstrings.
7. B6 notebook and clean run, B7 documentation, pull request.

Commit sequence on the branch:

1. Report schemas and report generator
2. Evaluator
3. Optimizer and reflection
4. SQLite memory
5. Evaluation charts and run outputs
6. Follow-up: Gemini provider and LLM cache (Engineer 1 files)
7. Follow-up: report pipeline and memory wiring (Engineer 1 files)
8. Follow-up: chart fixes (Engineer 1 and 2 files)
9. Follow-up: restored Engineer 2 tests
10. Follow-up: docstrings and comments (Engineer 1 and 2 files)
11. Follow-up: notebook restructure and clean execution
12. Follow-up: README, architecture and plan updates

## Verification Gates

- All tests pass: the existing 35, the new Engineer 3 tests and the restored Engineer 2 tests.
- Gemini smoke test: the planner and news-classifier schemas both return valid JSON from `gemini-3.8-flash`.
- `build_runtime()` reports every component as `real`.
- A comprehensive MSFT run shows the first draft, criterion scores with issues, at least one refinement cycle, the refined report, and either a higher score or a recorded reason why not.
- Every citation in the final report resolves to aggregated evidence, and no statement cites memory.
- The second run retrieves the stored lesson and applies it, and its plan differs from the first run's.
- `memory_retrieval` and `memory_storage` appear in the tool-call log.
- Every chart has a title, axis labels with units, a legend where needed, and an interpretation.
- The notebook runs top to bottom in a fresh environment without manual edits.
- `outputs/` holds both reports, the charts and the run log, and no API key appears in any output.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Gemini's OpenAI compatibility layer is still in beta and may reject strict JSON schemas | Schema-in-prompt fallback, the existing deterministic fallbacks, and a smoke test before any other work |
| Quota: limits apply per project and are shown in AI Studio, and a full notebook run makes about 100 LLM calls | Tier 3 limits, 429 retries, the response cache and low thinking effort |
| Gemini 3 has no temperature control, so the text varies between runs | The response cache for re-runs; scores and decisions come from code checks |
| The LLM's first draft already passes, leaving the optimizer little to do | Strict coverage and citation checks; the notebook explains a no-refinement outcome; the learning demo's first run always has a disclosed gap |
| Live market and news data change daily | Captions computed from the data, and the run date printed |
| The notebook is open in PyCharm while a scripted run saves it | Check for a PyCharm kernel and the file's modification time before writing, and reload from disk afterwards |

## Progress

- [x] Phase 0: setup (branch, `openai` and `pytest` in `Course5/.venv`, `.env`, Gemini client)
- [x] Gemini smoke test with the live key (planner, classifier, report-draft and clarity schemas all accepted by `response_format`, no fallback needed)
- [x] A1 and A2: schemas and report generator
- [x] A3: evaluator
- [x] A4 and A5: optimizer and reflection
- [x] A6 and A7: memory and learning demo (end-to-end test passes offline)
- [x] A8 and A9: charts and outputs
- [x] B1 to B5: Gemini client and cache, wiring, chart fixes, restored tests, docstrings (158 tests pass)
- [x] B6: notebook restructured and executed top to bottom on live data with Gemini 3.8 Flash, in a fresh environment built from `requirements.txt`
- [x] B7: documentation
- [ ] Pull request opened and reviewed by Engineers 1 and 2

## Implementation Notes

- The report schemas live in `src/report_models.py` and the evidence pack in `src/report_evidence.py`, so the writer, evaluator and optimizer share one structure without import cycles.
- Statements carry a kind: `fact`, `interpretation` or `disclosure`. Disclosures (scope, data dates, gaps) need no evidence, which keeps honest statements about the analysis from failing the grounding check.
- Issues the run cannot fix, such as price drivers when the plan skipped news, are marked not actionable, so the optimizer does not spend a cycle on them; reflection turns them into lessons instead.
- The numeric check also accepts figures from the specialists' own finding sentences for the cited evidence, because those sentences hold computed values (prices, moving averages) that are not stored as evidence values. An explicit sign must agree with signed evidence.
- `AgentState` gained `reflection` and `output_files` in the wiring commit; the report pipeline sets them either way.
- The first live run showed a lenient evaluator: Gemini's first draft scored 95 and was accepted although it printed raw figures (USD 331839000000.0) and a double negative, which only the LLM judge noticed (clarity 3 of 5, exactly the old floor of 50). The run also exposed false positives: spelled-out dates read as figures, ratios derived from cited figures, and the disclaimer "does not guarantee future returns" read as advice. The evaluator now skips spelled-out dates, accepts ratios and percentage-point gaps derived from cited figures, ignores negated advisory words, combines the LLM judgement with rule-based readability checks (taking the lower score), and requires every criterion to reach 60. Section notes are written for the reader, and exchange codes are spelled out. The writer prompt was not changed.
- A third live run showed a flaw in the lock: when the judge raised a section-level wording issue, every statement in that section was locked, so a rewrite that rephrased or merged bullets got the originals appended back and created near-duplicates (the score fell from 95.5 to 95.0, and the loop correctly kept version 0). A locked statement now comes back only if what it cites disappears. The same run showed that the clarity prompt contained the report's generation time, which defeated the response cache; the judge now reads the content only.
- Engineer 2's tests were ported with a plain fake LLM in place of the LangChain fakes. Three tests that targeted APIs removed in the merge (the unscoped registry's `call`/`log` and the LangGraph local model) were dropped, and the SEC filing test was split in two because the integrated earnings agent calls SEC filings only when the plan selects them.

## Open Items

- Due date, to schedule the teammates' review.

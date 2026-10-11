# Engineer 1 and Engineer 2 Integration Plan

## Scope and Decisions

- Base: `feature/vivek`; integrate `origin/feature/engineer-2-data-subagents` on a dedicated integration branch.
- Keep Engineer 1's Planner, Orchestrator, shared AgentState, scoped tools, routing, and Engineer 3 handoff.
- Integrate Engineer 2's real data connectors, deterministic specialists, optional macro/filing tools, five-stage news chain, and charts.
- Default to Groq/Hugging Face APIs; retain local Qwen as an explicitly selected, lazily loaded alternative.
- Report generation, evaluation/optimization, reflection, and SQLite memory remain Engineer 3's responsibility. Missing modules must remain visible.
- Use existing system Python and existing Engineer 1 tests; do not add tests or a virtual environment.
- Create a local merge commit only after validation. Do not push.

## Implementation Order

1. Fetch both branches, record their revisions, and perform a noncommitting merge on the integration branch.
2. Resolve shared-file conflicts in favor of the orchestration interfaces, merging connector configuration and dependencies deliberately.
3. Isolate specialist-specific models and adapt specialist calls to `(state, ToolSession, step)` without bypassing tool selection or logging.
4. Normalize specialist findings, warnings, catalysts, evidence IDs, provenance, and visualization payloads into shared state.
5. Inject the selected API/local LLM into news stages; keep the single Planner/Orchestrator as the default entry point.
6. Incorporate real-data charts and all five news-stage outputs into the existing notebook using notebook-aware editing.
7. Update configuration examples and usage documentation, install required non-GPU dependencies, and validate.

## Verification Gates

- Existing Engineer 1 tests pass; the integrated real-data workflow is exercised through the existing notebook without adding tests.
- Real implementations register with stubs disabled; API mode never loads a local checkpoint.
- Narrow requests call only their selected specialists and permitted tools; comprehensive requests execute all three.
- Optional macro/SEC tools run only when selected; data-source failure produces explicit partial/failed coverage without stopping independent work.
- Finding evidence references remain resolvable after aggregation; original source URLs, dates, units, and retrieval timestamps survive.
- News preserves ingest, preprocess, classify, extract, and summarize intermediate output and chart data.
- Notebook cells are syntactically valid; live provider checks are attempted without exposing credentials, and environmental blockers are recorded.
- Engineer 3 integration status remains `missing` until those modules are supplied.
- The alternate Engineer 2 LangGraph runner and tests tied to its incompatible unscoped registry are excluded; Engineer 1 remains the sole orchestration entrypoint.
- Review the final diff and commit the resolved merge locally without pushing.

## Progress

- [x] User confirmed scope, default backend, and local merge commit.
- [x] Merge and resolve conflicts.
- [x] Adapt runtime, specialists, and evidence.
- [x] Integrate notebook and documentation.
- [x] Validate: 35 Engineer 1 tests passed; all `src` modules compiled; all 9 notebook code cells compiled; comprehensive real-data notebook run completed with all three specialists successful and both news feeds available.
- [x] Review, stage, and create local merge commit `ab5c99a`; do not push.
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

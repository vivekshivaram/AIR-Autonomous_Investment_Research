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

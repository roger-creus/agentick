# Leaderboard

Scripts for creating, validating, running, viewing, and comparing leaderboard
submissions.

## Prerequisites

```bash
uv sync              # base install for create/validate/view/compare
uv sync --extra all  # run_evaluation.py uses suite loader and numpy
```

## Scripts

- **create_submission.py** -- Generate a `submission.yaml` template with fields
  for agent name, author, agent class, observation mode, target suites, and
  hardware metadata. Edit the template before submitting.
- **validate_submission.py** -- Check a submission YAML for required fields,
  valid Python syntax in the adapter code, and forbidden patterns (shell
  execution, dynamic eval). Reports errors and optional missing fields.
- **run_evaluation.py** -- Run a full local evaluation from a submission YAML.
  Loads the agent adapter, iterates over a task suite, and saves per-task
  results (mean reward, success rate) to JSON.
- **view_results.py** -- Display a formatted summary of evaluation results:
  overall statistics, per-task breakdown, capability analysis by category
  (navigation, reasoning, memory, etc.), and performance distribution.
- **compare_agents.py** -- Load two or more evaluation result JSON files and
  produce an overall comparison table, task-by-task success rates, head-to-head
  win/loss record, and best/worst tasks per agent.

## Running

```bash
# Create a submission template
uv run python examples/leaderboard/create_submission.py

# Validate the submission
uv run python examples/leaderboard/validate_submission.py submission.yaml

# Run local evaluation
uv run python examples/leaderboard/run_evaluation.py submission.yaml --suite core

# View results
uv run python examples/leaderboard/view_results.py evaluation_results/evaluation_results.json

# Compare two agents
uv run python examples/leaderboard/compare_agents.py results1.json results2.json
```

## Typical Workflow

1. Create a submission template with `create_submission.py`.
2. Fill in your agent details and adapter code.
3. Validate with `validate_submission.py`.
4. Run local evaluation with `run_evaluation.py`.
5. View results with `view_results.py`.
6. Compare against baselines with `compare_agents.py`.

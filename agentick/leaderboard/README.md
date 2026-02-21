# Leaderboard

Standardized evaluation, scoring, and ranking system for comparing agents on Agentick benchmark suites.

## Overview

The leaderboard system runs agents on fixed benchmark suites with locked seeds, computes normalized scores, and maintains a serverless JSON-file database of results. It supports API-based, HuggingFace, local-weights, code, Docker, and git-repo agent submissions.

## Modules

### `evaluator.py`
- **`LeaderboardEvaluator`** -- Main evaluation engine. Loads agents from `SubmissionSpec`, runs them on a `BenchmarkSuite`, tracks costs, verifies integrity, and produces `EvaluationResult` objects.

### `submission.py`
- **`SubmissionSpec`** (Pydantic model) -- Defines agent identity (name, author, tags, license), agent type (`api`, `huggingface`, `local_weights`, `code`, `docker`, `git_repo`), observation mode, and connection config. Loads from YAML via `SubmissionSpec.from_yaml()`.

### `result.py`
- **`EpisodeResult`** -- Per-episode data: task, difficulty, seed, return, steps, success, optional trajectory.
- **`EvaluationResult`** -- Complete evaluation output: per-task scores, aggregate Agentick score with CI, submission metadata.

### `scoring.py`
- **`TaskScore`** -- Score for one task (mean return, success rate, normalized score with CI).
- **`CapabilityScore`** -- Average normalized score across tasks in a capability category.
- **`AggregateScore`** -- Overall Agentick score, equally weighted by capability.
- **`compute_score_from_results()`** -- Converts raw episode results into the scoring hierarchy.

### `rankings.py`
- **`compute_rankings()`** -- Sorts evaluation results by `agentick_score`, assigns ranks, and annotates with CI and per-task breakdowns.

### `database.py`
- **`LeaderboardDatabase`** -- Serverless JSON-file storage. Organizes entries, rankings, and history under a configurable `data_dir`. Methods: `add_entry()`, query, and ranking persistence.

### `suites.py`
- **`BenchmarkSuite`** -- Dataclass defining a suite: task list, difficulties, seeds per task, and scoring config.
- **`OFFICIAL_SUITES`** -- Predefined suites. `CORE_TASKS` covers the original 27 tasks.
- **`ScoringConfig`** -- Scoring parameters exported at package level.

### `seeds.py`
- **`generate_seeds_from_name()`** -- Deterministic seed generation using SHA-256 hash of the suite name, producing uniformly distributed seeds in `[0, 2^31)`.

### `integrity.py`
- **`compute_result_hash()`** -- SHA-256 hash of deterministic result fields (scores, episodes) for tamper detection.
- **`verify_reproducibility()`** -- Re-runs subset of episodes and checks consistency.
- **`verify_result()`** -- Validates a saved result file.

### `comparison.py`
- **`compare_entries()`** -- Pairwise comparison of two `EvaluationResult` objects with per-task deltas and aggregate score difference.

### `cost_tracker.py`
- **`CostTracker`** -- Tracks API token usage and cost. Ships with per-model pricing tables for OpenAI, Anthropic, and Google models.

### `baselines.py`
- **`run_random_baseline()`** -- Runs a random agent on a task to establish floor performance for score normalization.

### `experiment.py`
- **`ExperimentConfig`** -- Lightweight dataclass for quick evaluations: task list, agent type, seed count, render mode, video/trace recording.

### `cli.py`
- **`cmd_evaluate()`** -- CLI entry point. Loads a `SubmissionSpec` from YAML, runs `LeaderboardEvaluator`, optionally verifies reproducibility. Integrates with `agentick` CLI via argparse.

## Usage

```python
from agentick.leaderboard.submission import SubmissionSpec
from agentick.leaderboard.evaluator import LeaderboardEvaluator

submission = SubmissionSpec.from_yaml("my_agent.yaml")
evaluator = LeaderboardEvaluator(verbose=True)
result = evaluator.evaluate(submission=submission, suite="core")
print(result.agentick_score)
```

```bash
# CLI
uv run agentick leaderboard evaluate --submission my_agent.yaml --suite core
```

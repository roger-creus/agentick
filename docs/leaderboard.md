# Leaderboard

**[View the live leaderboard](https://roger-creus.github.io/agentick/board/)**

## What Can You Submit?

You don't need to evaluate on the full benchmark to submit. We accept:

- **Full benchmark** — all tasks, all 4 difficulties (appears in the overall ranking + category breakdown + per-task view)
- **Partial results** — any subset of tasks and/or difficulties (appears in the per-task breakdown only, with `–` for missing entries)

Even a single task at a single difficulty is welcome. Every data point helps the community understand where current agents stand.

## Submission Flow

1. **Run evaluation** using the official eval seeds (25 seeds per task-difficulty pair):

```bash
uv run python -m agentick.experiments.run --config your_config.yaml
```

2. **Validate** your results:

```bash
uv run python scripts/validate_submission.py results/<your_run>/
```

3. **Submit** — email the generated zip to `roger.creus-castanyer@mila.quebec`

Include in the email: agent name, your name/affiliation, agent type (rl/llm/vlm/hybrid), observation mode, model name, and whether weights are open.

## Scoring Methodology

Each task-difficulty pair gets a **normalized score** (ONS):

```
ONS = (agent_return - random_baseline) / (oracle_return - random_baseline)
```

Scores are clipped to [0, 1]. The **Agentick Score** is the mean across all evaluated task-difficulty pairs. 95% bootstrap confidence intervals are computed over the per-episode results.

## Category Scores

Scores are also computed per category (only when all tasks in a category are fully evaluated):

| Category | Tasks |
|---|---|
| Navigation | 8 |
| Planning | 9 |
| Reasoning | 8 |
| Memory | 4 |
| Generalization | 3 |
| Multi-Agent | 5 |

## Evaluation Seeds

Seeds are deterministic and derived from a SHA-256 hash of `"{task_name}::{difficulty}::eval"`. See the [Experiments](experiments.md) page for details on seed generation and results format.

# Leaderboard

**[View the live leaderboard](https://roger-creus.github.io/agentick/board/)**

## Submission Flow

1. **Run evaluation** on all 37 tasks, all 4 difficulties, using the official eval seeds (25 per task-difficulty = 3800 total episodes):

```bash
uv run python -m agentick.experiments.run --config your_config.yaml
```

2. **Validate** your results:

```bash
uv run python scripts/validate_submission.py results/<your_run>/
```

3. **Submit** — email the generated zip to `roger.creus-castanyer@mila.quebec`

## Scoring Methodology

Each task-difficulty pair gets a **normalized score**:

```
score = (agent_return - random_baseline) / (oracle_return - random_baseline)
```

Scores are clipped to [0, 1]. The **Agentick Score** is the mean across all 37 tasks x 4 difficulties = 148 task-difficulty pairs. 95% bootstrap confidence intervals are computed over the per-episode results.

## Category Scores

Scores are also computed per category:

| Category | Tasks |
|---|---|
| Navigation | 8 |
| Planning | 9 |
| Reasoning | 9 |
| Memory | 4 |
| Generalization | 3 |
| Multi-Agent | 5 |

## Evaluation Seeds

Seeds are deterministic and derived from a SHA-256 hash of `"{task_name}::{difficulty}::eval"`. See the [Experiments](experiments.md) page for details on seed generation and results format.

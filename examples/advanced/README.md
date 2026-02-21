# Advanced

These examples go beyond the basics: building custom tasks and reward functions,
playing tasks interactively, and running parallel evaluations.

## Prerequisites

```bash
uv sync            # base install is sufficient for most scripts
uv sync --extra all  # parallel_eval.py uses the oracle baseline
```

## Scripts

- **custom_task.py** -- Build a complete custom Gymnasium environment
  (CollectGems) from scratch with walls, gems, lava, and an exit. Shows entity
  placement, reward logic, registration with `gym.register`, and a test episode.
- **custom_reward.py** -- Wrap any Agentick task with a `CustomRewardWrapper`
  that applies exploration bonuses, distance shaping, efficiency bonuses,
  curiosity rewards, or a combination of several components.
- **human_play.py** -- Play any task interactively in the terminal with WASD or
  arrow keys. Useful for understanding task difficulty and collecting human
  demonstrations. Accepts `--task` and `--difficulty` arguments.
- **parallel_eval.py** -- Evaluate agents across multiple tasks in parallel
  using `ProcessPoolExecutor`. Includes demos for multi-task evaluation, agent
  comparison (oracle vs random), and a sequential-vs-parallel speedup benchmark.

## Running

```bash
uv run python examples/advanced/custom_task.py
uv run python examples/advanced/custom_reward.py
uv run python examples/advanced/human_play.py --task GoToGoal-v0 --difficulty easy
uv run python examples/advanced/parallel_eval.py
```

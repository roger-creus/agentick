# Experiments

Run systematic evaluations with YAML configs or the Python API.

## YAML Config

```yaml
# my_experiment.yaml
name: gpt4o-language
description: "GPT-4o on navigation tasks"
agent:
  type: llm
  hyperparameters:
    backend: openai
    model: gpt-4o
    harness: markovian_zero_shot
    observation_modes: [language]
    api_key_env: OPENAI_API_KEY
    max_tokens: 100
    temperature: 0.0
tasks: [GoToGoal-v0, MazeNavigation-v0]
difficulties: [easy, medium]
split: eval          # "eval" for benchmarking, "train" for training
n_seeds: 25          # Seeds per (task, difficulty)
n_episodes: 1        # Episodes per seed
render_modes: [language]
record_trajectories: true
output_dir: results/gpt4o
```

Run it:

```bash
uv run python -m agentick.experiments.run --config my_experiment.yaml
```

## Python API

```python
from agentick.experiments.runner import ExperimentRunner
from agentick.experiments.config import ExperimentConfig

config = ExperimentConfig.from_yaml("my_experiment.yaml")
runner = ExperimentRunner(config)
results = runner.run()
```

## Benchmark Suites

Use predefined suites instead of listing tasks:

```yaml
tasks: "full"           # All tasks
tasks: "navigation"     # 8 navigation tasks
tasks: "planning"       # 9 planning tasks
tasks: "reasoning"      # 9 reasoning tasks
tasks: "memory"         # 4 memory tasks
tasks: "generalization" # 3 generalization tasks
tasks: "multi_agent"    # 5 multi-agent tasks
```

## Seed System

Seeds are generated **per task per difficulty** from the `split` field using SHA-256 hashing. No need to list explicit seeds:

- `split: eval` — 25 deterministic eval seeds per (task, difficulty). Used for benchmarking and leaderboard.
- `split: train` — 2000 train seeds per (task, difficulty). Used for RL/SFT training.

**Never train on eval seeds.**

```python
from agentick.leaderboard.seeds import generate_task_seeds, get_train_seeds, get_eval_seeds

eval_seeds = get_eval_seeds("GoToGoal-v0", "medium")    # 25 seeds
train_seeds = get_train_seeds("GoToGoal-v0", "medium")  # 2000 seeds
seeds = generate_task_seeds("GoToGoal-v0", "medium", "eval", 10)  # Custom count
```

7 official suites, all using per-task eval seeds:

| Suite | Tasks |
|---|---|
| `agentick-full-v2` | 38 |
| `agentick-navigation-v2` | 8 |
| `agentick-planning-v2` | 9 |
| `agentick-reasoning-v2` | 9 |
| `agentick-memory-v2` | 4 |
| `agentick-generalization-v2` | 3 |
| `agentick-multiagent-v2` | 5 |

## Scoring

Results are normalized to [0, 1]:

```
score = (agent_return - random_baseline) / (optimal_return - random_baseline)
```

Where `random_baseline` is the expected return of a random agent and `optimal_return` comes from oracle performance. Scores are aggregated per-capability and overall.

## Results Format

Results are saved to `output_dir/{name}_{timestamp}/`:

```
results/gpt4o_20260302_120000/
├── config.yaml          # Experiment config
├── metadata.json        # Runtime metadata (agent, platform, git hash)
├── summary.json         # Aggregate results
├── figures/             # Auto-generated plots
└── per_task/
    ├── GoToGoal-v0/
    │   ├── metrics.json
    │   └── episodes/
    │       └── diff_easy_seed_0_ep_0.json
    └── MazeNavigation-v0/
        ├── metrics.json
        └── episodes/
```

**metadata.json** includes: `agentick_version`, `python_version`, `platform`, `git_hash`, `agent_name`, `agent_type`, `model`, `backend`, `observation_modes`, `harness`.

**per_task/{task}/metrics.json** contains per-difficulty episode data and aggregate metrics (mean_return, success_rate, mean_length).

## Example Configs

Pre-built configs in `examples/experiments/configs/`:

```bash
# Run a predefined config
uv run python -m agentick.experiments.run --config examples/experiments/configs/random_agent.yaml
```

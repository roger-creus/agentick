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

## Seed System

Seeds are generated **per task per difficulty** from the `split` field. No need to list explicit seeds:

- `split: eval` — uses the 25 deterministic eval seeds per (task, difficulty)
- `split: train` — uses train-split seeds (for RL/SFT training)

See [Evaluation Seeds](seeds.md) for details.

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
tasks: "full"           # All 38 tasks
tasks: "navigation"     # 8 navigation tasks
tasks: "planning"       # 9 planning tasks
tasks: "reasoning"      # 9 reasoning tasks
tasks: "memory"         # 4 memory tasks
tasks: "generalization" # 3 generalization tasks
tasks: "multi_agent"    # 5 multi-agent tasks
```

## Scoring

Results are normalized to [0, 1]:

```
score = (agent_return - random_baseline) / (optimal_return - random_baseline)
```

Where `random_baseline` is the expected return of a random agent and `optimal_return` comes from oracle performance. Scores are aggregated per-capability and overall.

## Output

Results are saved to `output_dir/{name}_{timestamp}/`:

```
results/gpt4o_20260302_120000/
├── config.yaml          # Experiment config
├── metadata.json        # Runtime metadata
├── summary.json         # Aggregate results
└── per_task/
    ├── GoToGoal-v0/
    │   ├── metrics.json
    │   └── episodes/
    └── MazeNavigation-v0/
        ├── metrics.json
        └── episodes/
```

## Example Configs

Pre-built configs in `examples/experiments/configs/`:

```bash
ls examples/experiments/configs/
# claude_sonnet_ascii.yaml, gpt4o_language.yaml,
# qwen3_4b_ascii_markov.yaml, ppo_pixels_dense.yaml, ...
```

Run a predefined config:

```bash
uv run python -m agentick.experiments.run --config examples/experiments/configs/qwen3_4b_ascii_markov.yaml
```

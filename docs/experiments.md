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
n_episodes: 10
n_seeds: 3
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

print(f"Tasks evaluated: {len(results.task_results)}")
for task, metrics in results.task_results.items():
    print(f"  {task}: success={metrics['success_rate']:.1%}")
```

Or define programmatically:

```python
from agentick.experiments.config import ExperimentConfig, AgentConfig

config = ExperimentConfig(
    name="random-baseline",
    agent=AgentConfig(type="random"),
    tasks=["GoToGoal-v0", "MazeNavigation-v0"],
    difficulties=["easy"],
    n_episodes=10,
    n_seeds=3,
)
runner = ExperimentRunner(config)
results = runner.run()
```

## Benchmark Suites

Use predefined suites instead of listing tasks:

```yaml
tasks: "full"    # All 38 tasks
tasks: "quick"   # 5 representative tasks
tasks: "core"    # 25 core tasks
```

## Scoring

Results are normalized to [0, 1]:

```
score = (agent_return - random_baseline) / (optimal_return - random_baseline)
```

Where `random_baseline` is the expected return of a random agent and `optimal_return` comes from oracle performance. Scores are aggregated per-capability and overall.

## Reproducibility

Use locked seeds for reproducible evaluation:

```yaml
seeds: [42, 123, 456]   # Explicit seeds
n_seeds: 10              # Or auto-generate N seeds
```

Same config + same seeds = identical results. See [Evaluation Seeds](seeds.md) for the official seed sets.

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
# oracle_agent.yaml, random_agent.yaml, claude_sonnet_ascii.yaml,
# gpt4o_language.yaml, qwen3_4b_ascii_markov.yaml, ...
```

Run a predefined config:

```bash
uv run python examples/experiments/run_predefined.py --config examples/experiments/configs/random_agent.yaml
```

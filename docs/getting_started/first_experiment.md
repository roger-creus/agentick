# Your First Experiment

Run systematic evaluations across multiple tasks with metrics collection and analysis.

## Quick Start

Create `my_experiment.yaml`:

```yaml
name: my_first_experiment
description: "Random agent on easy tasks"

agent:
  type: random

tasks:
  - GoToGoal-v0
  - MazeNavigation-v0
  - KeyDoorPuzzle-v0

num_seeds: 3
episodes_per_seed: 10
render_mode: ascii
```

Run it:

```python
from agentick.leaderboard.experiment import ExperimentRunner

runner = ExperimentRunner(
    config_path="my_experiment.yaml",
    output_dir="results"
)

results = runner.run()
print(f"Saved to: results/")
```

## Configuration Options

### Agent Types

```yaml
# Random baseline
agent:
  type: random

# Oracle (optimal)
agent:
  type: oracle

# Custom agent code
agent:
  adapter: |
    def get_action(obs, env):
        # Your logic here
        return env.action_space.sample()
  model: "my-model"
```

### Task Selection

```yaml
# List specific tasks
tasks:
  - GoToGoal-v0
  - KeyDoorPuzzle-v0

# Or use a suite
tasks: core  # Predefined suite
```

### Evaluation Settings

```yaml
num_seeds: 5              # Reproducibility
episodes_per_seed: 10     # Episodes per seed
render_mode: ascii        # Observation format
timeout: 100              # Max steps per episode
```

## Analyzing Results

Results are saved as JSON:

```python
import json

with open("results/summary.json") as f:
    results = json.load(f)

# Overall metrics
print(f"Success rate: {results['success_rate']:.1%}")
print(f"Mean reward: {results['mean_reward']:.2f}")

# Per-task breakdown
for task_result in results['tasks']:
    task = task_result['task']
    success = task_result['success_rate']
    print(f"{task}: {success:.1%}")
```

## Comparing Agents

Run multiple experiments:

```python
from agentick.leaderboard.experiment import ExperimentRunner

agents = {
    "random": {"type": "random"},
    "oracle": {"type": "oracle"},
}

for name, agent_config in agents.items():
    runner = ExperimentRunner(
        config_path=f"{name}_config.yaml",
        output_dir=f"results/{name}"
    )
    runner.run()
```

Compare results:

```python
import json
from pathlib import Path

results = {}
for agent_dir in Path("results").iterdir():
    with open(agent_dir / "summary.json") as f:
        results[agent_dir.name] = json.load(f)

for name, data in results.items():
    print(f"{name}: {data['success_rate']:.1%} success")
```

## Visualizing Results

Use the plotting utilities:

```python
from agentick.visualization.experiment_plots import ExperimentPlotter

plotter = ExperimentPlotter("results")
plotter.plot_all(output_dir="plots")
```

Generates:
- Per-task scores
- Success rate comparison
- Score distributions
- Capability radar charts

## Example Workflows

### Quick Test

```yaml
name: quick_test
agent: {type: random}
tasks: [GoToGoal-v0]
num_seeds: 1
episodes_per_seed: 3
```

### Full Evaluation

```yaml
name: full_eval
agent: {type: oracle}
tasks: core  # All core tasks
num_seeds: 10
episodes_per_seed: 10
```

### Multi-Difficulty

```yaml
name: difficulty_scaling
agent: {type: random}
tasks: [GoToGoal-v0]
difficulty: [easy, medium, hard]
num_seeds: 5
episodes_per_seed: 10
```

## Next Steps

- [RL Agents](../agents/rl_agents.md) - Train your own agents
- [LLM Agents](../agents/llm_agents.md) - Evaluate language models
- [Leaderboard](../leaderboard/submitting.md) - Submit results

# Your First Experiment

Complete walkthrough of running a real experiment with experiment configs, multiple tasks, and result analysis.

## Overview

An "experiment" in Agentick is a systematic evaluation of an agent across multiple tasks, difficulties, and seeds with metrics collection and result analysis. This guide walks you through:

1. Creating an experiment configuration
2. Running the experiment
3. Analyzing and visualizing results
4. Comparing agents

## Part 1: Create an Experiment Configuration

Experiment configurations are YAML files that define what to evaluate. Create a file `my_first_experiment.yaml`:

```yaml
# Basic experiment metadata
name: my_first_experiment
description: "My first Agentick experiment - random agent on easy tasks"

# Which agent to evaluate
agent:
  type: random
  hyperparameters: {}

# Which tasks to evaluate (suite name or list of tasks)
tasks: quick  # "quick" suite: GoToGoal, KeyDoor, Sokoban, PreciseNav, PhysicsDiscovery

# Which difficulties to test
difficulties:
  - easy

# How many episodes per task per difficulty
n_episodes: 10

# How many different seeds for reproducibility
n_seeds: 3

# Which observation mode to use
render_modes:
  - ascii

# Metrics to compute
metrics:
  - mean_return
  - success_rate
  - mean_length

# Whether to save episode trajectories
record_trajectories: true

# Output directory
output_dir: results

# Optional: tags for filtering experiments
tags:
  - baseline
  - random-agent
```

Save this as `my_first_experiment.yaml`.

### Understanding Configuration Options

**agent** - Which agent to evaluate:
```yaml
# Random agent (baseline)
agent:
  type: random
  hyperparameters: {}

# Oracle agent (knows optimal solution)
agent:
  type: oracle
  hyperparameters: {}

# Custom agent
agent:
  type: custom
  hyperparameters:
    model_path: "path/to/model.pt"
    learning_rate: 0.001
```

**tasks** - Tasks to evaluate:
```yaml
# Built-in suite
tasks: quick  # or "full", "navigation", "memory", "reasoning", etc.

# Or specific task list
tasks:
  - GoToGoal-v0
  - KeyDoorPuzzle-v0
  - SokobanPush-v0
```

**difficulties** - Problem difficulty levels:
```yaml
difficulties:
  - easy      # Small problems, simple
  - medium    # Medium-sized with some complexity
  - hard      # Large, complex problems
  - expert    # Very challenging
```

**metrics** - What to measure:
```yaml
metrics:
  - mean_return          # Average episode reward
  - success_rate         # Fraction of successful episodes
  - mean_length          # Average episode length
  - std_return           # Standard deviation of returns
  - median_return        # Median episode reward
  - min_return           # Minimum episode reward
  - max_return           # Maximum episode reward
  - action_efficiency    # Success per action
  - exploration_efficiency  # Exploration vs exploitation
```

## Part 2: Run the Experiment Programmatically

Load the config and run the experiment:

```python
from agentick.experiments import ExperimentConfig, ExperimentRunner

# Load configuration
config = ExperimentConfig.from_yaml("my_first_experiment.yaml")

# Validate configuration (optional but recommended)
errors = config.validate_config()
if errors:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("✓ Configuration is valid")

# Create and run the experiment
runner = ExperimentRunner()
results = runner.run(config)

# Print summary
print("\n" + "=" * 60)
print("EXPERIMENT COMPLETE")
print("=" * 60)
print(f"Output directory: {results.output_dir}")
print(f"Results saved to: {results.output_dir}")
```

## Part 3: Understand the Output Structure

Experiments create an output directory with this structure:

```
results/my_first_experiment_20250212_143022/
├── config.yaml                 # Exact config used
├── metadata.json               # Git hash, versions, timestamps
├── summary.json                # Aggregate metrics across all tasks
└── per_task/
    ├── GoToGoal-v0/
    │   ├── metrics.json        # Metrics for this task
    │   └── episodes/
    │       ├── seed_0_ep_0.json
    │       ├── seed_0_ep_1.json
    │       └── ...
    ├── KeyDoorPuzzle-v0/
    │   ├── metrics.json
    │   └── episodes/
    └── ...
```

### Examining Results

```python
import json
from pathlib import Path

results_dir = "results/my_first_experiment_20250212_143022"

# Load summary
with open(f"{results_dir}/summary.json") as f:
    summary = json.load(f)

print("Overall Performance:")
print(f"  Success Rate: {summary['success_rate']:.1%}")
print(f"  Mean Return: {summary['mean_return']:.3f}")
print(f"  Mean Length: {summary['mean_length']:.1f}")

# Per-task results
for task_name in summary.get('per_task', {}).keys():
    task_metrics = summary['per_task'][task_name]['aggregate_metrics']
    print(f"\n{task_name}:")
    print(f"  Success Rate: {task_metrics['success_rate']:.1%}")
    print(f"  Mean Return: {task_metrics['mean_return']:.3f}")

# Per-episode details
with open(f"{results_dir}/per_task/GoToGoal-v0/metrics.json") as f:
    task_metrics = json.load(f)

# View metrics by difficulty
for difficulty, diff_data in task_metrics['per_difficulty'].items():
    print(f"\nGoToGoal-v0 ({difficulty}):")
    print(f"  Episodes: {len(diff_data['episodes'])}")
    print(f"  Mean Return: {diff_data['metrics']['mean_return']:.3f}")
    print(f"  Success Rate: {diff_data['metrics']['success_rate']:.1%}")
```

## Part 4: Create a Multi-Difficulty Experiment

Let's create a more complex experiment testing multiple difficulties:

```yaml
name: multi_difficulty_experiment
description: "Evaluate random agent across easy/medium/hard"

agent:
  type: random
  hyperparameters: {}

tasks: quick

difficulties:
  - easy
  - medium
  - hard

n_episodes: 20
n_seeds: 5
render_modes:
  - ascii

metrics:
  - mean_return
  - success_rate
  - mean_length
  - std_return

record_trajectories: true
output_dir: results
```

Run it:

```python
from agentick.experiments import ExperimentConfig, ExperimentRunner

config = ExperimentConfig.from_yaml("multi_difficulty_experiment.yaml")
runner = ExperimentRunner()
results = runner.run(config)

# Analyze by difficulty
import json
from pathlib import Path

results_dir = results.output_dir

for task_dir in Path(results_dir).glob("per_task/*/"):
    task_name = task_dir.name
    with open(f"{task_dir}/metrics.json") as f:
        metrics = json.load(f)

    print(f"\n{task_name}:")
    for difficulty, diff_data in metrics['per_difficulty'].items():
        m = diff_data['metrics']
        print(f"  {difficulty:6s}: Success={m['success_rate']:5.1%}, "
              f"Return={m['mean_return']:7.3f}")
```

## Part 5: Comparison with Multiple Agents

Create configs for different agents and compare them:

**random_agent.yaml:**
```yaml
name: baseline_random
agent:
  type: random
  hyperparameters: {}
tasks: quick
difficulties: [easy, medium]
n_episodes: 20
n_seeds: 3
metrics: [mean_return, success_rate, mean_length]
output_dir: results
```

**oracle_agent.yaml:**
```yaml
name: baseline_oracle
agent:
  type: oracle
  hyperparameters: {}
tasks: quick
difficulties: [easy, medium]
n_episodes: 20
n_seeds: 3
metrics: [mean_return, success_rate, mean_length]
output_dir: results
```

Run both:

```python
from agentick.experiments import ExperimentConfig, ExperimentRunner
import json
from pathlib import Path

runner = ExperimentRunner()

# Run both experiments
for config_file in ["random_agent.yaml", "oracle_agent.yaml"]:
    print(f"\nRunning {config_file}...")
    config = ExperimentConfig.from_yaml(config_file)
    results = runner.run(config)

# Compare results
results_dirs = sorted(Path("results").glob("*"))
experiments = {}

for result_dir in results_dirs[-2:]:  # Last 2 experiments
    with open(f"{result_dir}/summary.json") as f:
        summary = json.load(f)

    config_name = result_dir.name.split("_")[0]
    experiments[config_name] = summary

# Side-by-side comparison
print("\n" + "=" * 70)
print("AGENT COMPARISON")
print("=" * 70)
print(f"{'Agent':<20} {'Success Rate':<15} {'Mean Return':<15}")
print("-" * 70)

for agent_name, summary in experiments.items():
    sr = summary.get('success_rate', 0)
    mr = summary.get('mean_return', 0)
    print(f"{agent_name:<20} {sr:>13.1%} {mr:>14.3f}")

# Per-task comparison
print("\n" + "=" * 70)
print("PER-TASK SUCCESS RATES")
print("=" * 70)

all_tasks = set()
for summary in experiments.values():
    all_tasks.update(summary.get('per_task', {}).keys())

for task_name in sorted(all_tasks):
    print(f"\n{task_name}:")
    print(f"{'Agent':<20} Success Rate")
    print("-" * 35)

    for agent_name, summary in experiments.items():
        if task_name in summary.get('per_task', {}):
            sr = summary['per_task'][task_name].get('success_rate', 0)
            print(f"  {agent_name:<18} {sr:>8.1%}")
```

## Part 6: Analyzing Episode Trajectories

```python
import json
from pathlib import Path
from collections import defaultdict

results_dir = "results/multi_difficulty_experiment_20250212_143022"

# Load episode data for a specific task/difficulty
episodes_dir = Path(results_dir) / "per_task" / "GoToGoal-v0" / "episodes"

episode_data = []
for episode_file in sorted(episodes_dir.glob("*.json")):
    with open(episode_file) as f:
        episode_data.append(json.load(f))

# Analyze statistics by seed
by_seed = defaultdict(list)
for ep in episode_data:
    seed_idx = ep['seed_idx']
    by_seed[seed_idx].append(ep)

print("Performance by Seed:")
for seed_idx in sorted(by_seed.keys()):
    episodes = by_seed[seed_idx]
    returns = [e['total_reward'] for e in episodes]
    successes = [e['success'] for e in episodes]

    print(f"  Seed {seed_idx}: "
          f"Mean Return={sum(returns)/len(returns):.3f}, "
          f"Success Rate={sum(successes)/len(successes):.1%}, "
          f"Episodes={len(episodes)}")

# Find best and worst episodes
best_ep = max(episode_data, key=lambda e: e['total_reward'])
worst_ep = min(episode_data, key=lambda e: e['total_reward'])

print(f"\nBest episode: Return={best_ep['total_reward']:.3f}, "
      f"Length={best_ep['episode_length']}, "
      f"Success={best_ep['success']}")
print(f"Worst episode: Return={worst_ep['total_reward']:.3f}, "
      f"Length={worst_ep['episode_length']}, "
      f"Success={worst_ep['success']}")

# Success distribution
successes = sum(1 for e in episode_data if e['success'])
print(f"\nOverall Success Rate: {successes}/{len(episode_data)} ({successes/len(episode_data):.1%})")
```

## Part 7: Generate Plots

If you have visualization dependencies installed (`pip install agentick[viz]`):

```python
import matplotlib.pyplot as plt
import json
from pathlib import Path
from collections import defaultdict

results_dir = "results/multi_difficulty_experiment_20250212_143022"

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Experiment Results', fontsize=16, fontweight='bold')

# Plot 1: Success rate by task
task_dir = Path(results_dir) / "per_task"
tasks = [d.name for d in task_dir.iterdir() if d.is_dir()]
success_rates = []

for task in sorted(tasks):
    with open(f"{task_dir}/{task}/metrics.json") as f:
        metrics = json.load(f)
    sr = metrics['aggregate_metrics']['success_rate']
    success_rates.append(sr)

axes[0, 0].bar(range(len(tasks)), success_rates)
axes[0, 0].set_xticks(range(len(tasks)))
axes[0, 0].set_xticklabels(tasks, rotation=45, ha='right')
axes[0, 0].set_ylabel('Success Rate')
axes[0, 0].set_title('Success Rate by Task')
axes[0, 0].set_ylim([0, 1])

# Plot 2: Mean return by task
mean_returns = []
for task in sorted(tasks):
    with open(f"{task_dir}/{task}/metrics.json") as f:
        metrics = json.load(f)
    mr = metrics['aggregate_metrics']['mean_return']
    mean_returns.append(mr)

axes[0, 1].bar(range(len(tasks)), mean_returns)
axes[0, 1].set_xticks(range(len(tasks)))
axes[0, 1].set_xticklabels(tasks, rotation=45, ha='right')
axes[0, 1].set_ylabel('Mean Return')
axes[0, 1].set_title('Mean Return by Task')

# Plot 3: Difficulty progression (using first task)
with open(f"{task_dir}/{tasks[0]}/metrics.json") as f:
    metrics = json.load(f)

difficulties = sorted(metrics['per_difficulty'].keys())
difficulty_sr = []
for diff in difficulties:
    sr = metrics['per_difficulty'][diff]['metrics']['success_rate']
    difficulty_sr.append(sr)

axes[1, 0].plot(difficulties, difficulty_sr, marker='o', linewidth=2, markersize=8)
axes[1, 0].set_ylabel('Success Rate')
axes[1, 0].set_xlabel('Difficulty')
axes[1, 0].set_title(f'{tasks[0]} - Success Rate by Difficulty')
axes[1, 0].set_ylim([0, 1])
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Episode length distribution
episodes_dir = Path(results_dir) / "per_task" / tasks[0] / "episodes"
episode_lengths = []

for episode_file in episodes_dir.glob("*.json"):
    with open(episode_file) as f:
        ep = json.load(f)
        episode_lengths.append(ep['episode_length'])

axes[1, 1].hist(episode_lengths, bins=20, edgecolor='black')
axes[1, 1].set_xlabel('Episode Length')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title(f'{tasks[0]} - Episode Length Distribution')

plt.tight_layout()
plt.savefig(f"{results_dir}/analysis.png", dpi=150, bbox_inches='tight')
print(f"Saved plot to {results_dir}/analysis.png")
plt.show()
```

## Part 8: Quick Experiment Template

Copy and customize this template:

```yaml
# experiment_template.yaml
name: my_experiment
description: "Description of what I'm testing"

agent:
  type: random  # or 'oracle', 'custom', etc.
  hyperparameters: {}

tasks: quick  # or list specific tasks
difficulties: [easy, medium]
n_episodes: 20
n_seeds: 3

render_modes:
  - ascii

metrics:
  - mean_return
  - success_rate
  - mean_length

record_trajectories: true
output_dir: results
tags: [my-tag]
```

Then run it:

```python
from agentick.experiments import ExperimentConfig, ExperimentRunner

config = ExperimentConfig.from_yaml("experiment_template.yaml")
runner = ExperimentRunner()
results = runner.run(config)

print(f"Results saved to: {results.output_dir}")
```

## Best Practices

### 1. Start Small

Run a quick test first:

```yaml
name: quick_test
tasks: quick
difficulties: [easy]
n_episodes: 5
n_seeds: 1
```

Then scale up once you know it works.

### 2. Use Descriptive Names

```yaml
name: gpt4_vs_claude_on_navigation_hard
description: "Compare GPT-4 and Claude on hard navigation tasks"
```

### 3. Include Metadata

```yaml
tags:
  - llm-comparison
  - navigation
  - v1.0
```

### 4. Reproducibility

Always use explicit seeds:

```yaml
n_seeds: 5
seeds: [42, 123, 456, 789, 999]  # Explicit seeds
```

### 5. Monitor Progress

For long experiments, save intermediate results:

```python
runner = ExperimentRunner()
results = runner.run(config, save_intermediate=True)
```

## Troubleshooting

### "Task not found: GoToGoal-v0"

List available tasks:

```python
import agentick
print(agentick.list_tasks())
```

### "Invalid difficulty: 'super_hard'"

Valid difficulties are: `easy`, `medium`, `hard`, `expert`

### Out of memory during experiment

Reduce `n_episodes` or `n_seeds`:

```yaml
n_episodes: 5   # Instead of 50
n_seeds: 1      # Instead of 5
```

### Results directory is huge

Disable trajectory recording:

```yaml
record_trajectories: false  # Don't save individual episodes
```

## Next Steps

1. **[Analyze Results](../experiments/analysis.md)** - Advanced analysis and visualization
2. **[Reproducibility](../experiments/reproducibility.md)** - Reproduce exact results
3. **[RL Agents](../agents/rl_agents.md)** - Train your own agent instead of using random
4. **[LLM Agents](../agents/llm_agents.md)** - Evaluate language models

## Complete Example Script

Here's a full example you can run immediately:

```python
#!/usr/bin/env python3
"""Complete first experiment example."""

from agentick.experiments import ExperimentConfig, ExperimentRunner
import json
from pathlib import Path

# Create a config
config = ExperimentConfig(
    name="my_first_complete_experiment",
    description="Random agent on quick task suite",
    agent={"type": "random", "hyperparameters": {}},
    tasks="quick",
    difficulties=["easy", "medium"],
    n_episodes=10,
    n_seeds=3,
    render_modes=["ascii"],
    metrics=["mean_return", "success_rate", "mean_length"],
    record_trajectories=True,
    output_dir="results"
)

# Run the experiment
print("Starting experiment...")
runner = ExperimentRunner()
results = runner.run(config)

# Load and display summary
summary_file = Path(results.output_dir) / "summary.json"
with open(summary_file) as f:
    summary = json.load(f)

print(f"\nExperiment Complete!")
print(f"Results saved to: {results.output_dir}")
print(f"\nSummary:")
print(f"  Overall Success Rate: {summary['success_rate']:.1%}")
print(f"  Overall Mean Return: {summary['mean_return']:.3f}")
print(f"  Overall Mean Length: {summary['mean_length']:.1f}")

# Show per-task results
print(f"\nPer-Task Results:")
for task_name, task_data in summary.get('per_task', {}).items():
    m = task_data.get('aggregate_metrics', {})
    print(f"  {task_name}:")
    print(f"    Success: {m.get('success_rate', 0):.1%}")
    print(f"    Return:  {m.get('mean_return', 0):.3f}")
```

Save as `run_first_experiment.py` and execute:

```bash
python run_first_experiment.py
```

Congratulations! You've now run a complete Agentick experiment!

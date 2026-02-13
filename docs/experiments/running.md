# Running Experiments

Execute systematic evaluations on Agentick tasks with comprehensive configuration options.

## Quick Start

### Using YAML Configuration

The simplest way to run experiments is with YAML configs:

```python
from agentick.leaderboard.experiment import ExperimentRunner

# Run experiment from YAML config
runner = ExperimentRunner(
    config_path="configs/quick/sanity_check.yaml",
    output_dir="results"
)
results = runner.run()

print(f"Total episodes: {len(results)}")
```

### Using Programmatic Configuration

You can also define experiments programmatically:

```python
from agentick.leaderboard.experiment import ExperimentRunner, ExperimentConfig

# Define experiment
config = ExperimentConfig(
    name="my_experiment",
    tasks=["GoToGoal-v0", "MazeNavigation-v0"],
    agent_type="random",
    num_seeds=3,
    episodes_per_seed=5,
    render_mode="rgb_array",
    timeout=100
)

# Run
runner = ExperimentRunner(config=config, output_dir="results")
results = runner.run()

# Results is a list of episode dictionaries
for result in results:
    print(f"Task: {result['task']}, Reward: {result['total_reward']}")
```

## YAML Configuration Format

Create a YAML file with your experiment configuration:

```yaml
# configs/my_experiment.yaml
name: my_experiment
description: Testing random agent on navigation tasks
tasks:
  - GoToGoal-v0
  - MazeNavigation-v0
agent_type: random
num_seeds: 3
episodes_per_seed: 5
render_mode: rgb_array
timeout: 100
record_video: true
record_traces: true
```

### Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | required | Unique experiment identifier |
| `description` | str | "" | Experiment description |
| `tasks` | list[str] | ["GoToGoal-v0"] | List of task IDs to evaluate |
| `agent_type` | str | "random" | Agent type: "random" or "greedy" |
| `num_seeds` | int | 3 | Number of random seeds |
| `episodes_per_seed` | int | 1 | Episodes per seed |
| `render_mode` | str | "rgb_array" | Render mode for environments |
| `timeout` | int | 100 | Max steps per episode |
| `record_video` | bool | true | Record videos of sample episodes |
| `record_traces` | bool | true | Save interaction traces |

## Programmatic Configuration Reference

```python
from agentick.leaderboard.experiment import ExperimentConfig

config = ExperimentConfig(
    name="my_experiment",
    description="Optional description",
    tasks=["GoToGoal-v0", "MazeNavigation-v0"],
    agent_type="random",  # or "greedy"
    num_seeds=3,
    episodes_per_seed=5,
    render_mode="rgb_array",
    timeout=100,
    record_video=True,
    record_traces=True,
)
```

### Difficulty Levels

Valid difficulty levels for all tasks:

- `easy`: Smallest grid, fewest obstacles, most steps
- `medium`: Standard configuration
- `hard`: Larger grid, more obstacles, fewer steps
- `expert`: Large grid, many obstacles, minimal steps

### Available Metrics

Metrics automatically computed during experiments:

- `mean_return`: Average episode return
- `std_return`: Standard deviation of returns
- `median_return`: Median episode return
- `min_return`: Minimum episode return
- `max_return`: Maximum episode return
- `success_rate`: Proportion of successful episodes (0-1)
- `mean_length`: Average episode length in steps
- `action_efficiency`: Ratio of optimal steps to agent's steps
- `exploration_efficiency`: Coverage of state space per step

## Agent Configuration

### Agent Types and Hyperparameters

```python
# Random agent (baseline)
AgentConfig(type="random")

# Oracle/optimal agent (knows optimal policy)
AgentConfig(type="oracle")

# PPO reinforcement learning
AgentConfig(
    type="ppo",
    hyperparameters={
        "learning_rate": 0.0003,
        "batch_size": 32,
        "n_steps": 2048,
        "entropy_coeff": 0.01,
        "value_coeff": 0.5,
    }
)

# DQN reinforcement learning
AgentConfig(
    type="dqn",
    hyperparameters={
        "learning_rate": 0.0001,
        "epsilon": 0.1,
        "replay_buffer_size": 10000,
        "target_update_freq": 1000,
    }
)

# Language Model agent
AgentConfig(
    type="llm",
    hyperparameters={
        "model": "gpt-4",
        "temperature": 0.7,
        "max_context_length": 4000,
    }
)

# Vision-Language Model agent
AgentConfig(
    type="vlm",
    hyperparameters={
        "model": "gpt-4-vision",
        "temperature": 0.5,
        "vision_detail": "high",
    }
)
```

## Running Experiments

### Sequential Execution (Python)

```python
from agentick.experiments import ExperimentRunner
from agentick.experiments.config import ExperimentConfig

config = ExperimentConfig(
    name="baseline_eval",
    agent={"type": "ppo"},
    tasks=["GoToGoal-v0", "Maze-v0"],
    n_episodes=20,
    n_seeds=5,
)

runner = ExperimentRunner(config)
results = runner.run()  # Sequential: runs tasks one by one

print(f"Mean return: {results.summary['mean_return']:.3f}")
print(f"Success rate: {results.summary['success_rate']:.2%}")
```

### Parallel Execution

```python
runner = ExperimentRunner(config)
results = runner.run(n_parallel=4)  # Run 4 tasks simultaneously
```

### Resuming Interrupted Experiments

Experiments automatically create checkpoints. Resume from the last checkpoint:

```python
runner = ExperimentRunner(config)
results = runner.run(resume_from="results/baseline_eval_20240115_143022")
```

## Configuration Files (YAML)

### Creating a Configuration File

Create `configs/experiment.yaml`:

```yaml
# Experiment metadata
name: ppo_benchmark
description: PPO performance on navigation tasks

# Agent configuration
agent:
  type: ppo
  hyperparameters:
    learning_rate: 0.0003
    batch_size: 32
    n_steps: 2048
    entropy_coeff: 0.01

# Tasks to evaluate
tasks:
  - GoToGoal-v0
  - Maze-v0
  - MultiGoal-v0

# Evaluation settings
difficulties: [easy, medium, hard]
n_episodes: 20
n_seeds: 5
render_modes: [ascii, language]
reward_mode: dense

# Data recording
record_trajectories: true
record_videos: false
record_observations: false

# Metrics to compute
metrics:
  - mean_return
  - success_rate
  - mean_length
  - std_return

# Organization
tags:
  - navigation
  - benchmark
  - ppo

# Output
output_dir: results
```

### Loading and Running YAML Config

```python
from agentick.experiments.config import load_config
from agentick.experiments import ExperimentRunner

config = load_config("configs/experiment.yaml")
runner = ExperimentRunner(config)
results = runner.run(n_parallel=4)
```

### Config Inheritance

Base config `configs/base.yaml`:

```yaml
agent:
  type: ppo
  hyperparameters:
    learning_rate: 0.0003
    batch_size: 32

n_seeds: 5
render_modes: [ascii]
record_trajectories: true
```

Variant `configs/navigation.yaml`:

```yaml
base_config: base.yaml

name: navigation_benchmark
tasks:
  - GoToGoal-v0
  - Maze-v0
  - MultiGoal-v0
difficulties: [easy, medium, hard]
n_episodes: 20

tags:
  - navigation
  - benchmark
```

## Predefined Experiment Suites

### Quick Suite (5 tasks, 2 seeds)

```python
config = ExperimentConfig(
    name="quick_test",
    agent={"type": "ppo"},
    tasks="quick",  # Predefined suite
    n_episodes=5,
    n_seeds=2,
)
```

Tasks: GoToGoal-v0, CollectKeys-v0, AvoidObstacles-v0, MemoryPath-v0, DoorKey-v0

### Navigation Suite

```python
config = ExperimentConfig(
    name="nav_benchmark",
    agent={"type": "ppo"},
    tasks="navigation",
)
```

Tasks: GoToGoal-v0, AvoidObstacles-v0, Maze-v0, MultiGoal-v0

### Memory Suite

```python
config = ExperimentConfig(
    name="memory_benchmark",
    agent={"type": "ppo"},
    tasks="memory",
)
```

Tasks: MemoryPath-v0, MemorySequence-v0, MemoryPairs-v0

### Reasoning Suite

```python
config = ExperimentConfig(
    name="reasoning_benchmark",
    agent={"type": "ppo"},
    tasks="reasoning",
)
```

Tasks: BlocksWorld-v0, PushBlock-v0, LightsOut-v0, GraphColoring-v0

### Full Suite (All Tasks)

```python
config = ExperimentConfig(
    name="full_benchmark",
    agent={"type": "ppo"},
    tasks="full",
)
```

## Multi-Agent Comparisons

### Sequential Agent Comparison

```python
from agentick.experiments import ExperimentRunner
from agentick.experiments.config import ExperimentConfig

agents = ["random", "ppo", "dqn"]

results_dict = {}
for agent_type in agents:
    config = ExperimentConfig(
        name=f"benchmark_{agent_type}",
        agent={"type": agent_type},
        tasks="navigation",
        n_seeds=5,
        tags=["benchmark", agent_type],
    )

    runner = ExperimentRunner(config)
    results = runner.run(n_parallel=4)
    results_dict[agent_type] = results

    print(f"{agent_type}: {results.summary['success_rate']:.2%}")
```

## Ablation Studies

### Systematic Ablations

```python
base_config = ExperimentConfig(
    name="ablation_study",
    agent={
        "type": "ppo",
        "hyperparameters": {
            "learning_rate": 0.0003,
            "entropy_coeff": 0.01,
            "value_coeff": 0.5,
            "batch_size": 32,
        }
    },
    tasks="navigation",
    n_seeds=5,
)

ablations = [
    {"entropy_coeff": 0.0},     # No entropy regularization
    {"value_coeff": 0.0},       # No value function
    {"learning_rate": 0.001},   # 3x higher LR
    {"batch_size": 64},         # 2x batch size
]

for ablation_params in ablations:
    config_dict = base_config.model_dump()
    config_dict["agent"]["hyperparameters"].update(ablation_params)
    config_dict["name"] = f"ablation_{list(ablation_params.keys())[0]}"

    config = ExperimentConfig(**config_dict)
    runner = ExperimentRunner(config)
    results = runner.run()

    print(f"{list(ablation_params.keys())[0]}: {results.summary['success_rate']:.2%}")
```

## Output Directory Structure

### Results Layout

```
results/
└── baseline_navigation_20240115_143022/
    ├── config.yaml              # Original configuration
    ├── metadata.json            # Execution metadata
    ├── summary.json             # Aggregate results
    ├── per_task/
    │   ├── GoToGoal-v0/
    │   │   ├── metrics.json     # Task-level metrics
    │   │   └── episodes/
    │   │       ├── seed_0_ep_0.json
    │   │       ├── seed_0_ep_1.json
    │   │       └── ...
    │   ├── Maze-v0/
    │   └── ...
    └── .checkpoint.json         # Checkpoint for resuming
```

### Summary Results (summary.json)

```json
{
  "mean_return": 0.85,
  "std_return": 0.12,
  "median_return": 0.88,
  "min_return": 0.45,
  "max_return": 1.0,
  "success_rate": 0.92,
  "mean_length": 42.3,
  "total_time_seconds": 3600.5,
  "total_episodes": 2000
}
```

### Per-Task Metrics

Each task has metrics.json:

```json
{
  "task_name": "GoToGoal-v0",
  "per_difficulty": {
    "easy": {
      "difficulty": "easy",
      "episodes": [
        {
          "seed": 42,
          "episode_idx": 0,
          "return": 1.0,
          "length": 10,
          "success": true
        }
      ],
      "metrics": {
        "mean_return": 0.95,
        "success_rate": 0.98,
        "mean_length": 12.5
      }
    }
  },
  "aggregate_metrics": {
    "mean_return": 0.90,
    "success_rate": 0.95,
    "mean_length": 15.0
  }
}
```

### Episode Trajectory Format

Each episode in `seed_X_ep_Y.json`:

```json
{
  "seed": 42,
  "seed_idx": 0,
  "episode_idx": 0,
  "steps": [
    {
      "step": 1,
      "action": 2,
      "reward": 0.0,
      "terminated": false,
      "truncated": false
    },
    {
      "step": 2,
      "action": 0,
      "reward": 0.0,
      "terminated": false,
      "truncated": false
    }
  ],
  "total_reward": 1.0,
  "episode_length": 10,
  "success": true
}
```

### Metadata (metadata.json)

```json
{
  "timestamp": "2024-01-15T14:30:22.123456",
  "config_name": "baseline_navigation",
  "git_hash": "abc123def456...",
  "agentick_version": "0.1.0",
  "python_version": "3.10.12",
  "platform": "Linux-5.15.0-139-generic",
  "cpu_count": 8
}
```

## Advanced Features

### Custom Metrics

Specify which metrics to compute:

```python
config = ExperimentConfig(
    name="custom_metrics",
    agent={"type": "ppo"},
    tasks=["GoToGoal-v0"],
    metrics=[
        "mean_return",
        "success_rate",
        "std_return",
        "action_efficiency",
        "exploration_efficiency",
    ]
)
```

### Multi-Modal Observations

Record multiple observation formats:

```python
config = ExperimentConfig(
    name="multimodal_experiment",
    agent={"type": "vlm"},
    tasks=["GoToGoal-v0"],
    render_modes=[
        "ascii",           # Text representation
        "language",        # Natural language description
        "language_structured",  # Structured language
        "rgb_array",       # Pixel observations
    ],
    record_observations=True,  # Save all modalities
)
```

## Configuration Validation

### Validation Before Running

```python
config = ExperimentConfig(
    name="experiment",
    agent={"type": "ppo"},
    tasks=["GoToGoal-v0"],
)

errors = config.validate_config()
if errors:
    for error in errors:
        print(f"Configuration error: {error}")
else:
    runner = ExperimentRunner(config)
    results = runner.run()
```

## Best Practices

1. **Meaningful experiment names**: Include agent type and key details
   - Good: `ppo_lr_0.0003_navigation_benchmark`
   - Bad: `exp1`, `test`

2. **Use tags for organization**: Group related experiments
   ```python
   tags=["navigation", "benchmark", "ppo", "paper"]
   ```

3. **Save and version configs**: Commit YAML configs to version control

4. **Use parallelization wisely**:
   - Set `n_parallel` based on available CPU cores
   - Reduce parallelization if memory is limited

5. **Monitor output directory**: Verify structure after runs

6. **Resume on failure**: Leverage checkpoint system for long runs

7. **Document assumptions**: Use `description` field

8. **Standard seeds**: Use consistent `n_seeds` across experiments

## Troubleshooting

### Out of Memory
```python
# Reduce parallelization
runner.run(n_parallel=1)

# Reduce batch size or episodes
config.n_episodes = 5
```

### Slow Experiments
```python
# Quick testing: 2 seeds, parallel execution
config.n_seeds = 2
runner.run(n_parallel=8)

# Scale up for final results
config.n_seeds = 5
runner.run(n_parallel=4)
```

### Configuration Errors
```python
# Always validate before running
errors = config.validate_config()
assert not errors, f"Config errors: {errors}"
```

### Interrupted Experiments
```python
# Find checkpoint directory
import glob
dirs = glob.glob("results/experiment_*/")
latest = max(dirs, key=os.path.getctime)

# Resume from checkpoint
runner.run(resume_from=latest)
```

# Reproducibility

Ensure experiments can be exactly reproduced across time and systems.

## Core Principles

Agentick is designed for **bit-perfect reproducibility**:

1. **Deterministic Environments**: Same seed produces identical episodes
2. **Config Versioning**: All settings saved with results
3. **Dependency Tracking**: Package versions recorded
4. **Git Integration**: Code version captured automatically
5. **System Metadata**: Hardware/OS info logged
6. **Config Hashing**: Detect configuration changes

## How Reproducibility is Ensured

### Random Seed Management

```python
from agentick.experiments.config import ExperimentConfig
from agentick.experiments import ExperimentRunner

# Option 1: Auto-generate seeds (reproducible with fixed base seed)
config = ExperimentConfig(
    name="reproducible_exp",
    agent={"type": "ppo"},
    tasks=["GoToGoal-v0"],
    n_seeds=5,
    seeds=None,  # Auto-generates from seed 42
)

# Option 2: Explicit seeds (most reproducible)
config = ExperimentConfig(
    name="reproducible_exp",
    agent={"type": "ppo"},
    tasks=["GoToGoal-v0"],
    n_seeds=5,
    seeds=[42, 123, 456, 789, 1011],
)
```

### Auto-Generated Seed Generation

When `seeds=None`, seeds are generated deterministically:

```python
import numpy as np

# Agentick's seed generation (from runner.py)
rng = np.random.default_rng(42)  # Fixed base seed
seeds = rng.integers(0, 1_000_000, size=5).tolist()
# Always produces: [123456, 456789, ...] for same n_seeds
```

### Configuration Hashing

Every config is hashed to track changes:

```python
from agentick.experiments.config import ExperimentConfig
import hashlib

config = ExperimentConfig(
    name="reproducible_exp",
    agent={"type": "ppo"},
    tasks=["GoToGoal-v0"],
)

# Config is automatically hashed
config_dict = config.model_dump()
config_hash = hashlib.sha256(str(sorted(config_dict.items())).encode()).hexdigest()
print(f"Config hash: {config_hash}")
```

### Metadata Recording

All execution metadata is captured:

```python
from agentick.experiments import ExperimentRunner

runner = ExperimentRunner(config)
results = runner.run()

# Access metadata
metadata = results.metadata
print(f"Timestamp: {metadata['timestamp']}")
print(f"Git commit: {metadata['git_hash']}")
print(f"Agentick version: {metadata['agentick_version']}")
print(f"Python version: {metadata['python_version']}")
print(f"Platform: {metadata['platform']}")
print(f"CPU count: {metadata['cpu_count']}")
```

## Reproducing Experiments

### Reproduce from Saved Results

The simplest way to reproduce is from saved results:

```python
from agentick.experiments.reproduce import reproduce

# Load original config and re-run
new_results = reproduce("results/baseline_navigation_20240115_143022")

# This automatically:
# 1. Loads config.yaml
# 2. Uses same seeds
# 3. Runs identical experiment
```

### Reproduce from Config File

```python
from agentick.experiments.config import load_config
from agentick.experiments import ExperimentRunner

# Load config with base config resolution
config = load_config("configs/paper/main_results.yaml")

# Run with exact same settings
runner = ExperimentRunner(config)
results = runner.run()
```

### Reproduce with Different System Settings

```python
# Change parallelization but keep everything else identical
config = load_config("configs/paper/main_results.yaml")

# Run with different n_parallel but same seeds
runner = ExperimentRunner(config)
results = runner.run(n_parallel=8)  # Different parallel workers

# Results should be identical except for timing
```

## Verifying Reproducibility

### Compare Two Runs

```python
from agentick.experiments.reproduce import verify

# Compare original and reproduced runs
report = verify(
    original_dir="results/baseline_navigation_20240115_143022",
    reproduced_dir="results/baseline_navigation_20240115_150000",
    rtol=1e-2,  # Relative tolerance (1%)
    atol=1e-3,  # Absolute tolerance
)

print(f"Config match: {report['config_match']}")
print(f"Summary match: {report['summary_match']}")
print(f"Per-task match: {report['per_task_match']}")

if report['differences']:
    for diff in report['differences']:
        print(f"  - {diff}")

print(f"Verification passed: {report['passed']}")
```

### Difference Report

```python
from agentick.experiments.reproduce import diff

# Get detailed difference report
diff_report = diff(
    results_dir_a="results/baseline_navigation_old",
    results_dir_b="results/baseline_navigation_new",
    output_path="diffs/comparison.json",
)

print("Configuration differences:")
for key, change in diff_report['config_diff'].items():
    print(f"  {key}: {change['a']} -> {change['b']}")

print("\nResults differences:")
for metric, values in diff_report['summary_diff'].items():
    if values:
        print(f"  {metric}: {values['a']} -> {values['b']} ({values['change']})")
```

### Seed Management

#### Auto-Generated Seeds (Reproducible)

```python
# The same n_seeds always produces same seeds
config1 = ExperimentConfig(name="exp1", agent={"type": "ppo"}, tasks=["GoToGoal-v0"], n_seeds=5)
config2 = ExperimentConfig(name="exp2", agent={"type": "ppo"}, tasks=["GoToGoal-v0"], n_seeds=5)

# Both will generate same seeds internally
runner1 = ExperimentRunner(config1)
runner2 = ExperimentRunner(config2)
# runner1.config.seeds == runner2.config.seeds (after generation)
```

#### Explicit Seeds (Most Reproducible)

```python
# Store seeds in config for guaranteed reproducibility
config = ExperimentConfig(
    name="reproducible_exp",
    agent={"type": "ppo"},
    tasks=["GoToGoal-v0"],
    n_seeds=5,
    seeds=[42, 123, 456, 789, 1011],  # Explicit seeds
)

# Can be saved in YAML
# config.to_yaml("configs/paper/main_results.yaml")

# Later, load and run
config = load_config("configs/paper/main_results.yaml")
runner = ExperimentRunner(config)
results = runner.run()  # Uses exact same seeds
```

## Seeding Across Components

### Environment Seeding

```python
import agentick

# Each environment step is deterministic given seed
env = agentick.make("GoToGoal-v0", seed=42)
obs, info = env.reset(seed=42)

# Same seed = same initial state
obs2, info2 = env.reset(seed=42)
assert np.array_equal(obs, obs2)
```

### Agent Seeding

```python
from agentick.experiments.config import AgentConfig

# PPO agent inherits seed
agent_config = AgentConfig(
    type="ppo",
    hyperparameters={
        "seed": 42,  # Agent uses this seed for initialization
        "learning_rate": 0.0003,
    }
)

config = ExperimentConfig(
    name="seeded_ppo",
    agent=agent_config,
    tasks=["GoToGoal-v0"],
    seeds=[42, 123, 456],  # Episode seeds
)
```

## Configuration Validation for Reproducibility

### Check Seed Consistency

```python
config = ExperimentConfig(
    name="exp",
    agent={"type": "ppo"},
    tasks=["GoToGoal-v0"],
    n_seeds=5,
    seeds=[1, 2, 3],  # Wrong number of seeds!
)

errors = config.validate_config()
# Will report: "len(seeds)=3 but n_seeds=5"
```

### Verify Config Hash Matches

```python
import hashlib
import json
from agentick.experiments.config import load_config

# Load config
config1 = load_config("configs/experiment.yaml")

# Hash it
hash1 = hashlib.sha256(json.dumps(config1.model_dump(), sort_keys=True).encode()).hexdigest()

# Load again and verify hash matches
config2 = load_config("configs/experiment.yaml")
hash2 = hashlib.sha256(json.dumps(config2.model_dump(), sort_keys=True).encode()).hexdigest()

assert hash1 == hash2, "Config file changed!"
```

## Common Reproducibility Issues

### Issue 1: Forgotten Seed in Config

```python
# WRONG - no explicit seeds
config = ExperimentConfig(
    name="not_reproducible",
    agent={"type": "ppo"},
    tasks=["GoToGoal-v0"],
)

# RIGHT - explicit seeds
config = ExperimentConfig(
    name="reproducible",
    agent={"type": "ppo"},
    tasks=["GoToGoal-v0"],
    seeds=[42, 123, 456, 789, 1011],
)
```

### Issue 2: Different Python/NumPy Versions

```python
# Version info saved in metadata
results = runner.run()
metadata = results.metadata

print(f"Python: {metadata['python_version']}")  # 3.10.12
print(f"Platform: {metadata['platform']}")      # Linux-5.15.0

# Different versions may give slightly different results due to
# floating point differences, NumPy changes, etc.
```

### Issue 3: Parallel vs Sequential Execution

```python
# Sequential (guaranteed reproducible)
runner.run(n_parallel=1)

# Parallel (usually reproducible, but order of updates may differ)
runner.run(n_parallel=4)

# Results should be identical but may have floating point variations
# due to different summation order in parallel aggregation
```

### Issue 4: Checkpoint Resumption

```python
# If interrupted and resumed, must use same n_parallel
config = ExperimentConfig(name="exp", ...)

# First run with n_parallel=4
runner = ExperimentRunner(config)
results1 = runner.run(n_parallel=4)  # Interrupted at 50%

# Resume with same n_parallel
results2 = runner.run(n_parallel=4, resume_from="results/exp_timestamp")

# Do NOT resume with different n_parallel - results may differ
```

## Reproducibility Best Practices

### 1. Always Use Explicit Seeds

```python
config = ExperimentConfig(
    name="my_experiment",
    agent={"type": "ppo"},
    tasks=["GoToGoal-v0"],
    n_seeds=5,
    seeds=[42, 123, 456, 789, 1011],  # ALWAYS explicit
)

config.to_yaml("configs/my_experiment.yaml")
```

### 2. Save Configuration with Results

```python
runner = ExperimentRunner(config)
results = runner.run()

# Config is automatically saved as config.yaml in results directory
# Verify it's there
import os
assert os.path.exists(f"{results.output_dir}/config.yaml")
```

### 3. Document System Information

```python
# Always record when running
results = runner.run()
metadata = results.metadata

print(f"Run information:")
print(f"  Date: {metadata['timestamp']}")
print(f"  Git commit: {metadata['git_hash']}")
print(f"  Agentick version: {metadata['agentick_version']}")
print(f"  Python {metadata['python_version']} on {metadata['platform']}")
```

### 4. Use Version Control for Configs

```bash
# Commit configs to git
git add configs/
git commit -m "Add experiment configs for paper"

# Document Git hash in paper
# "All experiments used commit abc123def456..."
```

### 5. Create Reproducibility Report

```python
def create_reproducibility_report(original_dir, new_dir, output_file):
    """Create report comparing two runs."""
    from agentick.experiments.reproduce import verify

    report = verify(original_dir, new_dir)

    with open(output_file, 'w') as f:
        f.write("# Reproducibility Report\n\n")
        f.write(f"Config match: {report['config_match']}\n")
        f.write(f"Summary match: {report['summary_match']}\n")
        f.write(f"Per-task match: {report['per_task_match']}\n")
        f.write(f"Overall passed: {report['passed']}\n\n")

        if report['differences']:
            f.write("## Differences Found\n")
            for diff in report['differences']:
                f.write(f"- {diff}\n")

    return report
```

## Sharing Reproducible Experiments

### Complete Reproducibility Package

```bash
# Directory structure for sharing
paper_experiments/
├── configs/
│   ├── base.yaml
│   ├── ppo_benchmark.yaml
│   ├── dqn_benchmark.yaml
│   └── README.md
├── scripts/
│   ├── run_all_experiments.py
│   └── verify_reproducibility.py
├── results/
│   ├── ppo_benchmark_20240115_143022/
│   │   ├── config.yaml
│   │   ├── metadata.json
│   │   └── ...
│   └── README.md
└── README.md
```

### Reproducibility README

```markdown
# Experiment Reproducibility Guide

## Quick Reproduction

```python
from agentick.experiments.config import load_config
from agentick.experiments import ExperimentRunner

config = load_config("configs/ppo_benchmark.yaml")
runner = ExperimentRunner(config)
results = runner.run()
```

## System Requirements

- Python 3.10+
- agentick==0.1.0
- Other requirements in requirements.txt

## Expected Results

- Mean return: 0.82 ± 0.05
- Success rate: 0.88 ± 0.03
- Total time: ~60 minutes on 8-core CPU

## Verification

```python
from agentick.experiments.reproduce import verify

verify(
    "results/ppo_benchmark_20240115_143022",
    "results/your_run",
)
```

## Troubleshooting

See troubleshooting section in docs/reproducibility.md
```

## Advanced Reproducibility

### Determinism Verification

```python
def verify_determinism(config, n_trials=3):
    """Verify experiment is deterministic."""
    from agentick.experiments import ExperimentRunner

    results_list = []

    for trial in range(n_trials):
        runner = ExperimentRunner(config)
        results = runner.run()
        results_list.append(results.summary)

    # Check all summaries match
    for i in range(1, len(results_list)):
        for key in results_list[0].keys():
            if isinstance(results_list[0][key], float):
                assert abs(results_list[0][key] - results_list[i][key]) < 1e-10, \
                    f"{key} differs: {results_list[0][key]} vs {results_list[i][key]}"

    print("✓ Experiment is deterministic")
    return True
```

### Cross-Platform Reproducibility

```python
# Different platforms may give slightly different results due to:
# 1. Different floating point implementations
# 2. Different NumPy/SciPy versions
# 3. Different hardware (CPU, GPU)

# Use tolerances for comparison
from agentick.experiments.reproduce import verify

verify(
    original_dir,
    reproduced_dir,
    rtol=1e-5,  # 0.001% relative tolerance
    atol=1e-7,  # 1e-7 absolute tolerance
)
```

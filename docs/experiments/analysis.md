# Analyzing Experiments

Comprehensive statistical analysis and comparison of experiment results.

## Loading Results

### Load Single Experiment

```python
from agentick.experiments.runner import ExperimentResults

# Load from results directory
results = ExperimentResults.load("results/baseline_navigation_20240115_143022")

# Access components
print(f"Config name: {results.config.name}")
print(f"Tasks: {list(results.per_task_results.keys())}")
print(f"Summary metrics: {results.summary}")
```

### Load Multiple Experiments

```python
from agentick.experiments.registry import ExperimentRegistry

registry = ExperimentRegistry("results")

# List all experiments
experiments = registry.list_experiments()
for exp in experiments:
    print(f"{exp['name']}: {exp['timestamp']}")

# Load latest run of experiment
latest = registry.load_latest("baseline_navigation")

# Load by tag
navigation_exps = registry.list_experiments(tag="navigation")
```

## Basic Metrics

### Metrics Computed During Experiments

Each experiment automatically computes these metrics:

```python
# Access summary metrics
summary = results.summary
print(f"Mean return: {summary['mean_return']:.3f}")
print(f"Std return: {summary['std_return']:.3f}")
print(f"Success rate: {summary['success_rate']:.2%}")
print(f"Mean length: {summary['mean_length']:.1f}")
```

### Per-Task Breakdown

```python
# Analyze individual tasks
for task_name, task_results in results.per_task_results.items():
    agg_metrics = task_results.get('aggregate_metrics', {})
    print(f"\n{task_name}:")
    print(f"  Mean return: {agg_metrics.get('mean_return', 0):.3f}")
    print(f"  Success rate: {agg_metrics.get('success_rate', 0):.2%}")

    # Per-difficulty breakdown
    for difficulty, diff_results in task_results['per_difficulty'].items():
        metrics = diff_results.get('metrics', {})
        print(f"  [{difficulty}] return={metrics.get('mean_return', 0):.3f}")
```

### Per-Seed and Episode-Level Data

```python
# Access individual episodes
for task_name, task_results in results.per_task_results.items():
    for difficulty, diff_results in task_results['per_difficulty'].items():
        episodes = diff_results['episodes']

        # Per-seed analysis
        for seed_idx in range(len(set(ep['seed'] for ep in episodes))):
            seed_episodes = [ep for ep in episodes if ep.get('seed_idx') == seed_idx]
            returns = [ep['return'] for ep in seed_episodes]
            print(f"Seed {seed_idx}: mean={np.mean(returns):.3f}, std={np.std(returns):.3f}")

        # All episodes
        for episode in episodes:
            print(f"Episode {episode['episode_idx']}: return={episode['return']}, success={episode['success']}")
```

## Statistical Tests

### Bootstrap Confidence Intervals

```python
from agentick.analysis.statistics import bootstrap_ci

# Compute CI for mean return
results = results.summary
returns = [ep['return'] for episodes in results for ep in episodes]

ci_result = bootstrap_ci(
    returns,
    statistic_fn=np.mean,
    n_bootstrap=10000,
    ci=0.95,
)

print(f"Mean: {ci_result.point_estimate:.3f}")
print(f"95% CI: [{ci_result.ci_lower:.3f}, {ci_result.ci_upper:.3f}]")
```

### Welch's t-test (Unequal Variances)

```python
from agentick.analysis.statistics import welch_t_test

# Compare two agents
agent_a_returns = [...]  # Returns from agent A
agent_b_returns = [...]  # Returns from agent B

t_result = welch_t_test(agent_a_returns, agent_b_returns)

print(f"t-statistic: {t_result.t_statistic:.3f}")
print(f"p-value: {t_result.p_value:.4f}")
print(f"Significant: {t_result.significant}")
```

### Mann-Whitney U Test (Non-parametric)

```python
from agentick.analysis.statistics import mann_whitney_u

mw_result = mann_whitney_u(agent_a_returns, agent_b_returns)

print(f"U-statistic: {mw_result.u_statistic:.3f}")
print(f"p-value: {mw_result.p_value:.4f}")
print(f"Significant (α=0.05): {mw_result.significant}")
```

### Permutation Test

```python
from agentick.analysis.statistics import permutation_test

perm_result = permutation_test(
    agent_a_returns,
    agent_b_returns,
    n_permutations=10000,
    random_seed=42,
)

print(f"Observed difference: {perm_result.observed_diff:.3f}")
print(f"p-value: {perm_result.p_value:.4f}")
print(f"Significant: {perm_result.significant}")
```

## Effect Sizes

### Cohen's d (Parametric Effect Size)

```python
from agentick.analysis.statistics import cohens_d

effect = cohens_d(agent_a_returns, agent_b_returns)

print(f"Cohen's d: {effect.d:.3f}")
print(f"Interpretation: {effect.interpretation}")  # negligible, small, medium, large

# Interpretation thresholds:
# |d| < 0.2: negligible
# 0.2 <= |d| < 0.5: small
# 0.5 <= |d| < 0.8: medium
# |d| >= 0.8: large
```

### Cliff's Delta (Non-parametric Effect Size)

```python
from agentick.analysis.statistics import cliff_delta

effect = cliff_delta(agent_a_returns, agent_b_returns)

print(f"Cliff's delta: {effect.delta:.3f}")
print(f"Interpretation: {effect.interpretation}")

# Interpretation thresholds:
# |delta| < 0.147: negligible
# 0.147 <= |delta| < 0.33: small
# 0.33 <= |delta| < 0.474: medium
# |delta| >= 0.474: large
```

## Comparing Agents

### Pairwise Agent Comparison

```python
from agentick.analysis.comparisons import compare_agents

# Prepare results from two agents
results_a = {
    "GoToGoal-v0": [0.95, 0.92, 0.98],  # Returns for each seed
    "Maze-v0": [0.85, 0.88, 0.90],
    "MultiGoal-v0": [0.75, 0.78, 0.80],
}

results_b = {
    "GoToGoal-v0": [0.88, 0.90, 0.92],
    "Maze-v0": [0.92, 0.95, 0.94],
    "MultiGoal-v0": [0.82, 0.85, 0.88],
}

comparison = compare_agents(results_a, results_b)

print(f"Winner: {comparison.winner}")  # 'agent_a', 'agent_b', or 'tie'
print(f"Tasks won by A: {comparison.wins_a}")
print(f"Tasks won by B: {comparison.wins_b}")
print(f"Ties: {comparison.ties}")

# Per-task analysis
for task in comparison.tasks:
    task_comp = comparison.per_task[task]
    print(f"\n{task}:")
    print(f"  A mean: {task_comp['mean_a']:.3f}")
    print(f"  B mean: {task_comp['mean_b']:.3f}")
    print(f"  Difference: {task_comp['diff']:.3f}")
    print(f"  Cohen's d: {task_comp['cohens_d']['d']:.3f} ({task_comp['cohens_d']['interpretation']})")
    print(f"  p-value: {task_comp['permutation']['p']:.4f}")
```

### Multiple Agent Comparison (Friedman Test)

```python
from agentick.analysis.comparisons import compare_multiple

# Results from 3+ agents
results_dict = {
    "agent_random": {
        "GoToGoal-v0": [0.10, 0.12, 0.08],
        "Maze-v0": [0.05, 0.08, 0.06],
        "MemoryPath-v0": [0.15, 0.12, 0.18],
    },
    "agent_ppo": {
        "GoToGoal-v0": [0.95, 0.92, 0.98],
        "Maze-v0": [0.85, 0.88, 0.90],
        "MemoryPath-v0": [0.75, 0.78, 0.80],
    },
    "agent_dqn": {
        "GoToGoal-v0": [0.88, 0.90, 0.92],
        "Maze-v0": [0.92, 0.95, 0.94],
        "MemoryPath-v0": [0.82, 0.85, 0.88],
    },
}

comparison = compare_multiple(results_dict)

# Friedman test (rank-based)
friedman = comparison['friedman']
print(f"Friedman χ²: {friedman['statistic']:.3f}")
print(f"p-value: {friedman['p_value']:.4f}")
print(f"Significant difference: {friedman['significant']}")

# Rankings (lower is better)
rankings = comparison['rankings']
for agent, rank in sorted(rankings.items(), key=lambda x: x[1]):
    print(f"{agent}: rank {rank:.2f}")

# Pairwise Nemenyi post-hoc test
critical_diff = comparison['critical_difference']
print(f"\nCritical difference: {critical_diff:.3f}")
for pair, pairwise in comparison['pairwise'].items():
    if pairwise['significant']:
        print(f"{pair}: SIGNIFICANT (diff={pairwise['rank_diff']:.3f})")
```

## Learning Curves

### Computing Learning Curves

```python
from agentick.analysis.learning_curves import compute_learning_curve

# Episode returns over time for each seed
episode_returns = [
    [0.1, 0.2, 0.3, 0.45, 0.55, 0.65, 0.75, 0.80, 0.85, 0.88],  # Seed 0
    [0.05, 0.15, 0.25, 0.40, 0.50, 0.60, 0.70, 0.78, 0.82, 0.85],  # Seed 1
    [0.12, 0.22, 0.35, 0.50, 0.58, 0.68, 0.76, 0.82, 0.86, 0.90],  # Seed 2
]

curve = compute_learning_curve(
    episode_returns,
    window_size=2,  # Smoothing window
    ci=0.95,
)

# Access results
mean_curve = curve['mean_curve']
ci_lower = curve['ci_lower']
ci_upper = curve['ci_upper']
n_episodes = curve['n_episodes']

print(f"Final mean: {mean_curve[-1]:.3f}")
print(f"Final 95% CI: [{ci_lower[-1]:.3f}, {ci_upper[-1]:.3f}]")
```

### Convergence Analysis

```python
from agentick.analysis.learning_curves import estimate_convergence_point

# When does learning plateau?
convergence = estimate_convergence_point(
    curve=mean_curve,
    threshold=0.95,  # 95% of final performance
    window=10,
)

if convergence['convergence_episode'] is not None:
    print(f"Reached 95% performance at episode: {convergence['convergence_episode']}")
    print(f"Final performance: {convergence['final_performance']:.3f}")
else:
    print("Did not reach 95% of final performance")
```

### Plateau Detection

```python
from agentick.analysis.learning_curves import plateau_detection

plateau = plateau_detection(
    curve=mean_curve,
    window=10,
    threshold=0.01,  # 1% improvement threshold
)

if plateau['plateau_detected']:
    print(f"Learning plateaued at episode: {plateau['plateau_start']}")
    print(f"Plateau value: {plateau['plateau_value']:.3f}")
else:
    print("Learning did not plateau (still improving)")
```

### Sample Efficiency

```python
from agentick.analysis.learning_curves import compute_sample_efficiency

curves = {
    "agent_ppo": mean_curve_ppo,
    "agent_dqn": mean_curve_dqn,
    "agent_random": mean_curve_random,
}

efficiency = compute_sample_efficiency(curves)

print("Convergence speed ranking (lower is better):")
for rank in efficiency['ranked']:
    print(f"  {rank['agent']}: {rank['episodes']} episodes to 95%")
```

## Specialized Metrics

### Normalized Score

```python
from agentick.analysis.metrics import normalized_score

score = normalized_score(
    returns=[0.85, 0.88, 0.90, 0.92],
    optimal=1.0,
    random_baseline=0.0,
    ci=0.95,
)

print(f"Score: {score['score']:.3f}")
print(f"95% CI: [{score['ci_lower']:.3f}, {score['ci_upper']:.3f}]")
```

### Agentick Score (Aggregate)

```python
from agentick.analysis.metrics import agentick_score

per_task_scores = {
    "GoToGoal-v0": 0.95,
    "Maze-v0": 0.85,
    "MemoryPath-v0": 0.75,
}

aggregate = agentick_score(per_task_scores)

print(f"Aggregate score: {aggregate['aggregate_score']:.3f}")
print(f"95% CI: [{aggregate['ci_lower']:.3f}, {aggregate['ci_upper']:.3f}]")
```

### Capability Profile

```python
from agentick.analysis.metrics import capability_profile

task_capability_map = {
    "GoToGoal-v0": "navigation",
    "Maze-v0": "navigation",
    "MemoryPath-v0": "memory",
    "MemorySequence-v0": "memory",
}

profile = capability_profile(per_task_scores, task_capability_map)

for capability, scores in profile.items():
    print(f"{capability}:")
    print(f"  Score: {scores['score']:.3f}")
    print(f"  95% CI: [{scores['ci_lower']:.3f}, {scores['ci_upper']:.3f}]")
    print(f"  N tasks: {scores['n_tasks']}")
```

### Consistency Across Seeds

```python
from agentick.analysis.metrics import consistency_score

per_seed_returns = [0.95, 0.92, 0.98, 0.91, 0.93]

consistency = consistency_score(per_seed_returns)

print(f"Consistency: {consistency['consistency']:.3f}")  # 0-1
print(f"Mean: {consistency['mean']:.3f}")
print(f"Std: {consistency['std']:.3f}")
print(f"CV: {consistency['cv']:.3f}")
```

### Action Efficiency

```python
from agentick.analysis.metrics import action_efficiency

efficiency = action_efficiency(agent_steps=25, optimal_steps=15)

print(f"Efficiency: {efficiency['efficiency']:.2f}")  # 15/25 = 0.6
print(f"Excess steps: {efficiency['excess_steps']}")  # 10
```

### Difficulty Scaling

```python
from agentick.analysis.metrics import difficulty_scaling

per_difficulty = {
    "easy": 0.95,
    "medium": 0.85,
    "hard": 0.70,
    "expert": 0.50,
}

scaling = difficulty_scaling(per_difficulty)

print(f"Scaling slope: {scaling['scaling_slope']:.3f}")  # Negative is good
print(f"Scaling R²: {scaling['scaling_r2']:.3f}")  # Goodness of fit
```

## Ablation Analysis

### Structured Ablation Study

```python
from agentick.analysis.comparisons import ablation_analysis

baseline = {
    "GoToGoal-v0": [0.95, 0.92, 0.98],
    "Maze-v0": [0.85, 0.88, 0.90],
}

ablations = {
    "no_entropy": {
        "GoToGoal-v0": [0.85, 0.80, 0.88],
        "Maze-v0": [0.70, 0.75, 0.78],
    },
    "no_value": {
        "GoToGoal-v0": [0.70, 0.68, 0.75],
        "Maze-v0": [0.60, 0.62, 0.65],
    },
}

results = ablation_analysis(baseline, ablations)

print("Ablation impact ranking (most to least critical):")
for item in results['ranked_by_impact']:
    print(f"  {item['name']}: {item['mean_diff']:.3f} (relative: {item['relative_performance']:.1%})")
```

## Multiple Comparison Correction

### Holm-Bonferroni Correction

```python
from agentick.analysis.statistics import holm_bonferroni

p_values = [0.001, 0.005, 0.01, 0.02, 0.05]

correction = holm_bonferroni(p_values)

print("Original p-values:", correction.original_p_values)
print("Adjusted p-values:", correction.adjusted_p_values)
print("Significant tests:", correction.significant)
```

### Benjamini-Hochberg FDR Correction

```python
from agentick.analysis.statistics import benjamini_hochberg

correction = benjamini_hochberg(p_values, fdr=0.05)

print("FDR-controlled adjusted p-values:", correction.adjusted_p_values)
print("Significant tests:", correction.significant)
```

## Outlier Detection

### IQR Method

```python
from agentick.analysis.statistics import iqr_outlier_detection

returns = [0.1, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 5.0]

outlier_result = iqr_outlier_detection(returns, iqr_multiplier=1.5)

print(f"Number of outliers: {outlier_result.n_outliers}")
print(f"Outlier indices: {outlier_result.outlier_indices}")
print(f"Bounds: [{outlier_result.lower_bound:.3f}, {outlier_result.upper_bound:.3f}]")
```

## Complete Analysis Example

```python
import numpy as np
from agentick.experiments.registry import ExperimentRegistry
from agentick.analysis.comparisons import compare_agents
from agentick.analysis.learning_curves import compute_learning_curve, estimate_convergence_point
from agentick.analysis.statistics import cohens_d, bootstrap_ci

# Load experiments
registry = ExperimentRegistry("results")
ppo_results = registry.load_latest("ppo_benchmark")
dqn_results = registry.load_latest("dqn_benchmark")

# Extract returns by task
ppo_returns = {}
dqn_returns = {}

for task, task_results in ppo_results.per_task_results.items():
    episodes = task_results['per_difficulty']['medium']['episodes']
    ppo_returns[task] = [ep['return'] for ep in episodes]

for task, task_results in dqn_results.per_task_results.items():
    episodes = task_results['per_difficulty']['medium']['episodes']
    dqn_returns[task] = [ep['return'] for ep in episodes]

# Compare agents
print("=== Agent Comparison ===")
comparison = compare_agents(ppo_returns, dqn_returns)
print(f"Winner: {comparison.winner}")
print(f"PPO wins: {comparison.wins_a}, DQN wins: {comparison.wins_b}, Ties: {comparison.ties}")

# Learning curves
print("\n=== Learning Curves ===")
# Collect learning curves from all seeds
ppo_curves = []
for seed_idx in range(5):
    # Gather episodes for this seed
    seed_episodes = [ep for ep in episodes if ep.get('seed_idx') == seed_idx]
    ppo_curves.append([ep['return'] for ep in seed_episodes])

curve_stats = compute_learning_curve(ppo_curves)
convergence = estimate_convergence_point(curve_stats['mean_curve'], threshold=0.95)

if convergence['convergence_episode']:
    print(f"PPO converges at episode: {convergence['convergence_episode']}")
else:
    print("PPO did not converge to 95% performance")

# Effect size
print("\n=== Effect Sizes ===")
effect = cohens_d(ppo_returns['GoToGoal-v0'], dqn_returns['GoToGoal-v0'])
print(f"Cohen's d for GoToGoal: {effect.d:.3f} ({effect.interpretation})")

# Confidence intervals
print("\n=== Confidence Intervals ===")
all_ppo_returns = [r for returns in ppo_returns.values() for r in returns]
ci = bootstrap_ci(all_ppo_returns)
print(f"PPO overall mean: {ci.point_estimate:.3f}")
print(f"95% CI: [{ci.ci_lower:.3f}, {ci.ci_upper:.3f}]")
```

## Output and Reporting

### Summary Statistics Table

```python
summary_stats = {
    "agent": [],
    "task": [],
    "mean_return": [],
    "std_return": [],
    "success_rate": [],
    "episodes_to_convergence": [],
}

for agent_name, results in {"ppo": ppo_results, "dqn": dqn_results}.items():
    for task, task_results in results.per_task_results.items():
        agg = task_results['aggregate_metrics']
        summary_stats["agent"].append(agent_name)
        summary_stats["task"].append(task)
        summary_stats["mean_return"].append(agg.get('mean_return', 0))
        summary_stats["std_return"].append(agg.get('std_return', 0))
        summary_stats["success_rate"].append(agg.get('success_rate', 0))

import pandas as pd
df = pd.DataFrame(summary_stats)
print(df.to_string())
```

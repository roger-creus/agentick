# Scoring

Agentick uses rigorous statistical scoring to fairly compare agent capabilities across diverse tasks. This guide explains the normalization methodology, capability aggregation, and leaderboard ranking.

## Overview

Scoring transforms raw episode returns into interpretable metrics:
1. **Normalize** each task score using random and optimal baselines
2. **Aggregate** task scores into per-capability scores
3. **Combine** capabilities into overall Agentick score
4. **Report** with confidence intervals for statistical rigor

## Normalized Scoring Formula

The fundamental scoring formula normalizes agent performance to [0, 1] scale:

$$\text{Score} = \frac{\text{Agent Return} - \text{Random Baseline}}{\text{Optimal Return} - \text{Random Baseline}}$$

Where:
- **Agent Return**: Mean episode return of agent
- **Random Baseline**: Expected return of random agent
- **Optimal Return**: Theoretical or empirical maximum

### Mathematical Notation

Let:
- $R_a$ = agent's mean return over episodes
- $R_r$ = random agent's mean return
- $R_o$ = oracle/optimal return

Then:
$$s = \max(0, \min(1, \frac{R_a - R_r}{R_o - R_r}))$$

Clamped to [0, 1] to handle edge cases.

### Why This Formula?

**1. Scale Invariance**: Tasks with different reward ranges are comparable
```
Task A: agent=50, random=0, optimal=100
  Score_A = (50 - 0) / (100 - 0) = 0.50

Task B: agent=1500, random=500, optimal=3000
  Score_B = (1500 - 500) / (3000 - 500) = 0.571

Both interpreted on [0, 1] scale despite different magnitudes
```

**2. Effort-Neutral**: Random agent always scores 0.0, optimal always 1.0
```
Random agent: (R_r - R_r) / (R_o - R_r) = 0
Optimal agent: (R_o - R_r) / (R_o - R_r) = 1.0
```

**3. Linear Interpolation**: Score reflects position between random and optimal

## Per-Task Scoring with Examples

### Example 1: Navigation Task (GoToGoal-v0, Medium)

```python
from agentick.leaderboard.scoring import compute_task_score

# Run agent for 10 episodes
episode_returns = [1.0, 0.85, 0.9, 0.8, 0.95, 0.88, 0.92, 0.87, 0.91, 0.89]
success_flags = [True] * 10  # All episodes successful

task_score = compute_task_score(
    task_name="GoToGoal-v0",
    difficulty="medium",
    episode_returns=episode_returns,
    random_baseline=0.04,  # Random agent success ~4%
    optimal_return=1.0,     # Perfect task completion
    success_flags=success_flags
)

print(f"Mean Return: {task_score.mean_return:.3f}")
print(f"Normalized Score: {task_score.normalized_score:.3f}")
print(f"95% CI: {task_score.normalized_score_ci}")
# Output:
# Mean Return: 0.898
# Normalized Score: 0.858  # (0.898 - 0.04) / (1.0 - 0.04)
# 95% CI: (0.825, 0.891)
```

### Example 2: Memory Task (KeyDoorPuzzle-v0, Medium)

```python
# Memory task has sparser success (harder to solve)
episode_returns = [0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.0]
success_flags = [False, False, False, True, False, False, True, False, False, False]

task_score = compute_task_score(
    task_name="KeyDoorPuzzle-v0",
    difficulty="medium",
    episode_returns=episode_returns,
    random_baseline=0.05,   # Rarely solved randomly
    optimal_return=1.0,      # Perfect solution
    success_flags=success_flags
)

print(f"Mean Return: {task_score.mean_return:.3f}")
print(f"Success Rate: {task_score.success_rate:.1%}")
print(f"Normalized Score: {task_score.normalized_score:.3f}")
# Output:
# Mean Return: 0.300
# Success Rate: 20.0%
# Normalized Score: 0.289  # (0.3 - 0.05) / (1.0 - 0.05)
```

### Handling Edge Cases

**Case 1: No gap between random and optimal (trivial task)**
```python
# If optimal ≈ random, task is either too easy or broken
if abs(optimal_return - random_baseline) < 1e-9:
    # Agent is either at optimal or at random baseline
    normalized_score = 1.0 if agent_return >= optimal_return else 0.0
```

**Case 2: Agent exceeds optimal (shouldn't happen, but cap it)**
```python
# Some tasks might have loose optimal estimates
normalized_score = min(normalized_score, 1.0)
```

## Baseline Computation

### Random Baseline

Random agent takes random actions each step:

```python
from agentick.leaderboard.baselines import run_random_baseline

random_result = run_random_baseline(
    task_name="GoToGoal-v0",
    difficulty="medium",
    n_episodes=50,      # 50 episodes to get stable estimate
    seeds=list(range(50))
)

print(f"Random Mean: {random_result['mean_return']:.3f}")
print(f"Random Std: {random_result['std_return']:.3f}")
print(f"Success Rate: {random_result['success_rate']:.1%}")
```

### Optimal Baseline

Optimal agent uses oracle solver (e.g., BFS for navigation):

```python
from agentick.leaderboard.baselines import run_oracle_baseline

oracle_result = run_oracle_baseline(
    task_name="GoToGoal-v0",
    difficulty="medium",
    n_episodes=20,
    seeds=list(range(20))
)

print(f"Oracle Mean: {oracle_result['mean_return']:.3f}")
print(f"Success Rate: {oracle_result['success_rate']:.1%}")
```

**Note**: Oracle may fail on intractable tasks (e.g., complex planning). Falls back to greedy heuristic:

```python
from agentick.leaderboard.baselines import run_greedy_baseline

greedy_result = run_greedy_baseline(
    task_name="SokobanPush-v0",
    difficulty="medium",
    n_episodes=20
)

# If oracle fails, use greedy estimate
if oracle_result is None:
    estimated_optimal = greedy_result["mean_return"] * 1.5
```

### Computing All Baselines

```python
from agentick.leaderboard.baselines import compute_baselines_for_suite

baselines = compute_baselines_for_suite(
    suite_name="agentick-full-v1",
    output_dir="leaderboard_data/baselines",
    run_random=True,
    run_greedy=True,
    run_oracle=True,
    n_episodes_random=50,
    n_episodes_oracle=20
)

# Returns dict: {task_name: {"random_baseline": float, "optimal_return": float}}
```

## Per-Capability Aggregation

Tasks group into **capability categories**. Scores within each category average to produce capability scores.

### Task-to-Capability Mapping

```python
from agentick.leaderboard.scoring import TASK_CAPABILITY_MAP

TASK_CAPABILITY_MAP = {
    # Navigation (5 tasks)
    "GoToGoal-v0": "navigation",
    "MazeNavigation-v0": "navigation",
    "FogOfWarExploration-v0": "navigation",
    "DynamicObstacles-v0": "navigation",
    "MultiGoalRoute-v0": "navigation",

    # Memory (4 tasks)
    "KeyDoorPuzzle-v0": "memory",
    "SequenceMemory-v0": "memory",
    "DelayedGratification-v0": "memory",
    "BacktrackPuzzle-v0": "memory",

    # Reasoning (5 tasks)
    "SokobanPush-v0": "reasoning",
    "CausalChain-v0": "reasoning",
    "SymbolMatching-v0": "reasoning",
    "RuleInduction-v0": "reasoning",
    "SwitchCircuit-v0": "reasoning",

    # Control (4 tasks)
    "PreciseNavigation-v0": "control",
    "TimingChallenge-v0": "control",
    "ChaseEvade-v0": "control",
    "Herding-v0": "control",

    # Skill (5 tasks)
    "ToolUse-v0": "skill",
    "RecipeAssembly-v0": "skill",
    "MultiRoomEscape-v0": "skill",
    "EmergentStrategy-v0": "skill",
    "ResourceManagement-v0": "skill",

    # Combinatorial (4 tasks)
    "LightsOut-v0": "combinatorial",
    "GraphColoring-v0": "combinatorial",
    "TileSorting-v0": "combinatorial",
    "PackingPuzzle-v0": "combinatorial",

    # World Model (3 tasks)
    "EnvironmentShift-v0": "worldmodel",
    "PhysicsDiscovery-v0": "worldmodel",
    "RuleDiscoveryNavigation-v0": "worldmodel",

    # Adversarial (3 tasks)
    "DeceptiveReward-v0": "adversarial",
    "DistributionShift-v0": "adversarial",
    "NoisyObservation-v0": "adversarial",

    # Meta-learning (2 tasks)
    "FewShotAdaptation-v0": "meta",
    "TaskInterference-v0": "meta",

    # Multi-agent (2 tasks)
    "CooperativeTransport-v0": "multiagent",
    "CompetitiveTag-v0": "multiagent",
}
```

### Computing Capability Scores

```python
from agentick.leaderboard.scoring import compute_capability_scores

task_scores = {
    "GoToGoal-v0": TaskScore(..., normalized_score=0.85),
    "MazeNavigation-v0": TaskScore(..., normalized_score=0.72),
    "FogOfWarExploration-v0": TaskScore(..., normalized_score=0.68),
    "DynamicObstacles-v0": TaskScore(..., normalized_score=0.81),
    "MultiGoalRoute-v0": TaskScore(..., normalized_score=0.74),
    # ... other tasks
}

capability_scores = compute_capability_scores(task_scores)

print(f"Navigation: {capability_scores['navigation'].mean_normalized_score:.3f}")
print(f"  Tasks: {capability_scores['navigation'].tasks}")
print(f"  Per-task scores: {capability_scores['navigation'].task_scores}")
print(f"  95% CI: {capability_scores['navigation'].normalized_score_ci}")

# Output:
# Navigation: 0.760
#   Tasks: ['GoToGoal-v0', 'MazeNavigation-v0', ...]
#   Per-task scores: {'GoToGoal-v0': 0.85, 'MazeNavigation-v0': 0.72, ...}
#   95% CI: (0.712, 0.808)
```

## Suite-Level Scoring

### Full Benchmark Scoring

$$\text{Agentick Score} = \text{Mean}(\text{All Capability Scores})$$

**Key property**: Each capability weighted equally, regardless of task count.

```python
from agentick.leaderboard.scoring import compute_aggregate_score

aggregate = compute_aggregate_score(task_scores)

print(f"Agentick Score: {aggregate.agentick_score:.3f}")
print(f"95% CI: {aggregate.agentick_score_ci}")

print("\nPer-Capability Breakdown:")
for cap_name, cap_score in aggregate.per_capability.items():
    print(f"  {cap_name}: {cap_score.mean_normalized_score:.3f}")
```

Example output:
```
Agentick Score: 0.673 ± 0.038

Per-Capability Breakdown:
  navigation: 0.760 ± 0.046
  memory: 0.582 ± 0.063
  reasoning: 0.451 ± 0.072
  control: 0.718 ± 0.055
  skill: 0.644 ± 0.058
  combinatorial: 0.528 ± 0.081
  worldmodel: 0.623 ± 0.067
  adversarial: 0.385 ± 0.095
  meta: 0.412 ± 0.108
  multiagent: 0.506 ± 0.092
```

### Category-Specific Suites

Score only subset of capabilities:

```python
# Navigation-only suite
navigation_tasks = {k: v for k, v in task_scores.items()
                   if TASK_CAPABILITY_MAP.get(k) == "navigation"}
navigation_aggregate = compute_aggregate_score(navigation_tasks)

print(f"Navigation Suite Score: {navigation_aggregate.agentick_score:.3f}")
```

## Confidence Intervals and Statistical Significance

All scores reported with 95% confidence intervals using bootstrap resampling.

### Bootstrap Methodology

```python
from agentick.leaderboard.scoring import bootstrap_confidence_interval

# Episode returns from agent
episode_returns = [0.95, 0.88, 0.92, 0.87, 0.91, ...]

# Normalize each return
normalized_returns = [
    normalize_score(ret, random_baseline, optimal_return)
    for ret in episode_returns
]

# Bootstrap CI
ci_lower, ci_upper = bootstrap_confidence_interval(
    normalized_returns,
    n_bootstrap=1000,      # Resample 1000 times
    confidence=0.95,       # 95% CI
    statistic="mean"       # Resample and compute mean
)

print(f"Score: {np.mean(normalized_returns):.3f} ± {(ci_upper - ci_lower) / 2:.3f}")
```

### Interpretation

95% CI of (0.725, 0.891) means:
- Point estimate: 0.808
- We're 95% confident the true score is between 0.725 and 0.891
- Wider CI = more uncertainty (fewer episodes or high variance)
- Narrower CI = more confident estimate

## Score Interpretation Guide

### What Different Scores Mean

**Score 0.0-0.2**: Random agent performance
- Agent barely beats random baseline
- Task not well-suited to agent architecture
- Fundamental capability gap

**Score 0.2-0.4**: Below average performance
- Agent understands task partially
- May solve easy cases, fails on hard
- Indicates capability gap or insufficient training

**Score 0.4-0.6**: Average performance
- Agent solves typical task cases
- Performance partway between random and optimal
- Reasonable capability, room for improvement

**Score 0.6-0.8**: Above average performance
- Agent demonstrates strong capability
- Solves most task variants
- Minor gaps remain (complex cases, hard difficulty)

**Score 0.8-1.0**: Excellent performance
- Agent near-optimal on this capability
- Handles diverse task variants well
- Represents true mastery

### Per-Capability Interpretation

```
Navigation: 0.85     ★★★★★  Excellent spatial reasoning
Memory: 0.62         ★★★☆☆  Good short-term memory
Reasoning: 0.48      ★★☆☆☆  Struggles with abstract reasoning
Control: 0.72        ★★★★☆  Good low-level control
Skill: 0.64          ★★★☆☆  Can compose behaviors
Combinatorial: 0.53  ★★★☆☆  Basic constraint satisfaction
World Model: 0.62    ★★★☆☆  Understands some dynamics
Adversarial: 0.39    ★★☆☆☆  Vulnerable to distribution shift
Meta: 0.41           ★★☆☆☆  Limited few-shot adaptation
Multi-Agent: 0.51    ★★★☆☆  Some cooperation ability
```

## Leaderboard Ranking Methodology

### Ranking Criteria

1. **Primary**: Agentick Score (overall performance)
2. **Tiebreaker 1**: Consistency (narrow CIs)
3. **Tiebreaker 2**: Worst capability score (breadth)

```python
from agentick.leaderboard.rankings import compute_leaderboard

results = {
    "PPO-CNN": compute_aggregate_score(ppo_task_scores),
    "A3C": compute_aggregate_score(a3c_task_scores),
    "DQN": compute_aggregate_score(dqn_task_scores),
}

leaderboard = compute_leaderboard(results)

for rank, (agent_name, score) in enumerate(leaderboard, 1):
    agg = results[agent_name]
    ci_width = agg.agentick_score_ci[1] - agg.agentick_score_ci[0]
    worst_cap = min(cap.mean_normalized_score for cap in agg.per_capability.values())

    print(f"{rank}. {agent_name:20s} {agg.agentick_score:.3f} ± {ci_width/2:.3f} "
          f"(worst cap: {worst_cap:.3f})")
```

Example output:
```
Agentick Leaderboard (Full Suite)

Rank  Agent Name           Score        95% CI       Worst Capability
----  -----------------   -------      ----------   -----------------
1.    Claude-Agent         0.687 ± 0.042 (0.645, 0.729) Adversarial (0.38)
2.    PPO-Transformer      0.654 ± 0.051 (0.603, 0.705) Meta (0.35)
3.    A3C                  0.631 ± 0.048 (0.583, 0.679) Adversarial (0.32)
4.    DQN                  0.598 ± 0.062 (0.536, 0.660) Reasoning (0.28)
5.    Random Agent         0.000 ± 0.001 (0.000, 0.001) All (0.00)
```

### Statistical Significance Testing

Pairwise comparisons using bootstrap hypothesis testing:

```python
from agentick.leaderboard.comparison import compare_agents

comparison = compare_agents(
    results_agent_a,
    results_agent_b,
    n_bootstrap=10000,
    alpha=0.05
)

print(f"Agent A: {comparison.score_a:.3f}")
print(f"Agent B: {comparison.score_b:.3f}")
print(f"Difference: {comparison.difference:.3f}")
print(f"95% CI of difference: {comparison.ci_difference}")
print(f"Significant? {comparison.is_significant}")
```

If CI of difference doesn't cross 0, agents are significantly different.

## Complete Scoring Example

```python
from agentick.leaderboard.scoring import compute_score_from_results

# Raw evaluation results
results = {
    "GoToGoal-v0": {
        "difficulty": "medium",
        "episode_returns": [0.95, 0.88, 0.92, ...],  # 50 episodes
        "success_flags": [True, True, True, ...],
    },
    "MazeNavigation-v0": {
        "difficulty": "medium",
        "episode_returns": [0.72, 0.68, 0.75, ...],
        "success_flags": [True, True, True, ...],
    },
    # ... 39 more tasks
}

# Load precomputed baselines
from agentick.leaderboard.baselines import load_baselines
baselines = load_baselines("leaderboard_data/baselines/agentick-full-v1_baselines.json")

# Compute full scoring
aggregate = compute_score_from_results(results, baselines)

# Report
print(f"Agentick Score: {aggregate.agentick_score:.3f}")
print(f"95% CI: {aggregate.agentick_score_ci}")
print(f"\nPer-Capability:")
for cap, score in aggregate.per_capability.items():
    print(f"  {cap}: {score.mean_normalized_score:.3f} ± "
          f"{(score.normalized_score_ci[1] - score.normalized_score_ci[0])/2:.3f}")
```

## Best Practices

**1. Compute confidence intervals always**
```python
# Never report point estimates without uncertainty
# BAD: "Score: 0.73"
# GOOD: "Score: 0.73 ± 0.04 (95% CI)"
```

**2. Use same baselines for all comparisons**
```python
# Baselines should be fixed for fair comparison
# Compute once, reuse for all agent evaluations
fixed_baselines = compute_baselines_for_suite(...)
```

**3. Report per-capability breakdown**
```python
# Don't just report aggregate score
# Show which capabilities agent excels/struggles in
for cap, score in per_capability.items():
    print(f"{cap}: {score.mean_normalized_score:.3f}")
```

**4. Document task distribution**
```python
# Note that some capabilities have more/fewer tasks
# This can affect overall score weighting
print(f"Navigation: {len(nav_tasks)} tasks")
print(f"Memory: {len(mem_tasks)} tasks")
print(f"Average: {sum(len(ts) for ts in task_groups) / len(task_groups):.1f} tasks/capability")
```

See [Rewards](rewards.md) for how task rewards relate to scores, and [World Model](worldmodel.md) for specialized evaluation.

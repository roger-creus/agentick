# Benchmark

Evaluation framework for running agents on predefined task suites and computing standardized metrics.

## Modules

### `suite.py` -- BenchmarkRunner, get_suite

- **`get_suite(name, difficulty, **kwargs)`** -- returns a list of environments from a predefined suite (e.g. `"full"`, `"quick"`, `"navigation"`, `"memory"`, `"reasoning"`) via `agentick.make_suite()`.
- **`BenchmarkRunner`** -- evaluates an agent callable across all tasks in a suite. Constructor takes `suite` name, `n_episodes`, and optional `seeds`. The `evaluate(agent_fn, difficulty)` method runs episodes and returns a dict mapping task names to `mean_return` and `success_rate`.

### `baselines.py` -- Baseline Agents

- **`RandomAgent`** -- uniform random action selection from valid actions. Seeded via `np.random.default_rng`.
- **`GreedyAgent`** -- heuristic that parses `state_dict` to locate the goal and moves toward it greedily (Manhattan priority: larger axis first).
- **`OracleAgent`** -- optimal solver (imported but defined elsewhere).

### `metrics.py` -- Metric Functions

All functions operate on lists/arrays of per-episode data:
- `success_rate(successes)` -- fraction of episodes with `success=True`
- `average_return(returns)` -- mean episode return
- `normalized_score(agent_return, random_baseline, optimal_return)` -- linear interpolation between random and optimal
- `sample_efficiency(returns, threshold)` -- episodes to reach a performance threshold
- `generalization_score(scores_by_difficulty)` -- mean across difficulty levels
- `capability_profile(results_by_capability)` -- per-capability mean scores (radar chart data)
- `agentick_score(normalized_scores)` -- single composite score across all tasks
- `confidence_interval(data, confidence, method)` -- bootstrap, normal, or t-based CIs (uses scipy.stats)

### `profiler.py` -- ProfilingResult, profile tasks

Measures `env.step()` throughput for any task/render mode. `ProfilingResult` dataclass stores `steps_per_sec`, `avg_step_time_us`, and percentile breakdowns (p50, p95, p99). Useful for identifying bottlenecks in training loops.

### `leaderboard.py` -- Leaderboard

Stores agent results in a dict, serializes to/from JSON, and prints a formatted comparison table with columns for agent name, mean return, and success rate.

## Quick Start

```python
from agentick.benchmark import BenchmarkRunner, RandomAgent, success_rate

runner = BenchmarkRunner(suite="quick", n_episodes=50)
agent = RandomAgent(seed=42)
results = runner.evaluate(agent, difficulty="easy")
```

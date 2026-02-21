# Analysis

Post-hoc statistical analysis tools for evaluating and comparing agent performance.

## Modules

### `statistics.py`
- **`StatisticalResult`** -- Generic container for test outputs (point estimates, p-values, CIs).
- **`bootstrap_ci()`** -- Bootstrap confidence interval for any statistic (default: mean). Configurable bootstrap count, CI level, and seed.
- **`welch_t_test()`**, **`mann_whitney_u()`** -- Parametric and non-parametric two-sample tests.
- **`permutation_test()`** -- Exact or approximate permutation test.
- **`cohens_d()`**, **`cliff_delta()`** -- Parametric and non-parametric effect size measures.
- **`holm_bonferroni()`**, **`benjamini_hochberg()`** -- Multiple comparison correction methods.
- **`iqr_outlier_detection()`** -- IQR-based outlier detection.

### `metrics.py`
- **`normalized_score()`** -- Maps raw returns to `[0, 1]` using `(return - random) / (optimal - random)` with bootstrap CI.
- **`agentick_score()`** -- Aggregates per-task normalized scores into a single benchmark score, optionally weighted.

### `comparisons.py`
- **`ComparisonResult`** -- Container with `n_tasks` and `winner` fields.
- **`compare_agents()`** -- Full pairwise comparison of two agents across shared tasks. Runs Welch t-test, Mann-Whitney U, and effect sizes per task, then applies Holm-Bonferroni correction across tasks.

### `learning_curves.py`
- **`compute_learning_curve()`** -- Smooths per-seed return curves with a uniform filter, computes cross-seed mean and bootstrap CI bands at each timestep. Returns `mean_curve`, `ci_lower`, `ci_upper`, and `raw_curves`.

## Usage

```python
from agentick.analysis.statistics import bootstrap_ci, welch_t_test
from agentick.analysis.comparisons import compare_agents
from agentick.analysis.metrics import normalized_score

# Bootstrap CI on returns
result = bootstrap_ci(agent_returns, ci=0.95)
print(f"{result.point_estimate:.3f} [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")

# Compare two agents
comparison = compare_agents(results_a, results_b, alpha=0.05)
print(comparison.winner)
```

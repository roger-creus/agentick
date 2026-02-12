"""Benchmark metrics with statistical rigor."""

from typing import Any

import numpy as np
from scipy import stats


def success_rate(successes):
    """Fraction of episodes with success=True."""
    return sum(successes) / len(successes) if successes else 0.0


def average_return(returns):
    """Mean episode return."""
    return np.mean(returns) if returns else 0.0


def normalized_score(agent_return, random_baseline, optimal_return):
    """Normalized score: (agent - random) / (optimal - random)."""
    if optimal_return == random_baseline:
        return 1.0 if agent_return >= optimal_return else 0.0
    return (agent_return - random_baseline) / (optimal_return - random_baseline)


def sample_efficiency(returns, threshold=0.8):
    """Episodes/steps to reach X% success."""
    for i, ret in enumerate(returns):
        if ret >= threshold:
            return i + 1
    return len(returns)


def generalization_score(scores_by_difficulty):
    """Performance across difficulty levels."""
    return np.mean(list(scores_by_difficulty.values()))


def capability_profile(results_by_capability):
    """Radar chart data across capability dimensions."""
    return {cap: np.mean(scores) for cap, scores in results_by_capability.items()}


def agentick_score(normalized_scores):
    """Weighted composite across all tasks."""
    return np.mean(normalized_scores)


def confidence_interval(
    data: list | np.ndarray,
    confidence: float = 0.95,
    method: str = "bootstrap",
) -> tuple[float, float, float]:
    """
    Compute confidence interval for data.

    Args:
        data: Array of measurements
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        method: "bootstrap", "normal", or "t"

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    data = np.asarray(data)

    if len(data) == 0:
        return 0.0, 0.0, 0.0

    mean = np.mean(data)

    if len(data) == 1:
        return mean, mean, mean

    if method == "bootstrap":
        lower, upper = bootstrap_ci(data, confidence=confidence)
    elif method == "normal":
        # Assumes normal distribution
        std_err = stats.sem(data)
        z_score = stats.norm.ppf((1 + confidence) / 2)
        margin = z_score * std_err
        lower, upper = mean - margin, mean + margin
    elif method == "t":
        # Student's t-distribution (better for small samples)
        std_err = stats.sem(data)
        df = len(data) - 1
        t_score = stats.t.ppf((1 + confidence) / 2, df)
        margin = t_score * std_err
        lower, upper = mean - margin, mean + margin
    else:
        raise ValueError(f"Unknown method: {method}")

    return mean, lower, upper


def bootstrap_ci(
    data: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    statistic: str = "mean",
) -> tuple[float, float]:
    """
    Bootstrap confidence interval.

    Args:
        data: Array of measurements
        confidence: Confidence level
        n_bootstrap: Number of bootstrap samples
        statistic: "mean", "median", or callable

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    data = np.asarray(data)

    if len(data) == 0:
        return 0.0, 0.0

    # Choose statistic function
    if statistic == "mean":
        stat_fn = np.mean
    elif statistic == "median":
        stat_fn = np.median
    elif callable(statistic):
        stat_fn = statistic
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    # Bootstrap sampling
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    bootstrap_stats = []

    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(stat_fn(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Compute percentiles
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return lower, upper


def statistical_significance_test(
    data1: np.ndarray,
    data2: np.ndarray,
    test: str = "t-test",
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """
    Test if two distributions are significantly different.

    Args:
        data1: First dataset
        data2: Second dataset
        test: "t-test", "mann-whitney", or "permutation"
        alternative: "two-sided", "less", or "greater"

    Returns:
        Dict with test_statistic, p_value, significant (at 0.05), and effect_size
    """
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    if len(data1) == 0 or len(data2) == 0:
        return {
            "test": test,
            "test_statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "effect_size": 0.0,
        }

    if test == "t-test":
        # Independent samples t-test
        stat, p_value = stats.ttest_ind(data1, data2, alternative=alternative)
    elif test == "mann-whitney":
        # Mann-Whitney U test (non-parametric)
        stat, p_value = stats.mannwhitneyu(data1, data2, alternative=alternative)
    elif test == "permutation":
        # Permutation test
        result = stats.permutation_test(
            (data1, data2),
            lambda x, y: np.mean(x) - np.mean(y),
            n_resamples=10000,
            alternative=alternative,
        )
        stat = result.statistic
        p_value = result.pvalue
    else:
        raise ValueError(f"Unknown test: {test}")

    # Cohen's d effect size
    pooled_std = np.sqrt((np.var(data1) + np.var(data2)) / 2)
    effect_size = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0.0

    return {
        "test": test,
        "test_statistic": float(stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "effect_size": float(effect_size),
    }


def aggregate_metrics_with_stats(results: list[float]) -> dict[str, Any]:
    """
    Compute aggregate metrics with statistical rigor.

    Args:
        results: List of measurement values

    Returns:
        Dict with mean, std, confidence intervals, and sample size
    """
    results = np.asarray(results)

    if len(results) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "n": 0,
            "ci_95": (0.0, 0.0),
            "ci_99": (0.0, 0.0),
        }

    mean, ci95_lower, ci95_upper = confidence_interval(results, confidence=0.95)
    _, ci99_lower, ci99_upper = confidence_interval(results, confidence=0.99)

    return {
        "mean": float(np.mean(results)),
        "std": float(np.std(results)),
        "median": float(np.median(results)),
        "min": float(np.min(results)),
        "max": float(np.max(results)),
        "n": int(len(results)),
        "ci_95": (float(ci95_lower), float(ci95_upper)),
        "ci_99": (float(ci99_lower), float(ci99_upper)),
    }

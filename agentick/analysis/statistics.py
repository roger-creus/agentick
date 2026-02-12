"""Core statistical functions for rigorous analysis."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import stats


class StatisticalResult:
    """Container for statistical test results with interpretation helpers."""

    def __init__(self, **kwargs: Any):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"


def bootstrap_ci(
    data: np.ndarray | list[float],
    statistic_fn: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    random_seed: int | None = None,
) -> StatisticalResult:
    """
    Compute bootstrap confidence interval for any statistic.

    Args:
        data: Input data
        statistic_fn: Function to compute statistic (default: mean)
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level (default: 0.95)
        random_seed: Random seed for reproducibility

    Returns:
        StatisticalResult with point_estimate, ci_lower, ci_upper, ci_level
    """
    data = np.asarray(data)
    rng = np.random.default_rng(random_seed)

    # Compute point estimate
    point_estimate = statistic_fn(data)

    # Bootstrap resampling
    bootstrap_estimates = np.empty(n_bootstrap)
    n = len(data)

    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_estimates[i] = statistic_fn(sample)

    # Compute percentile confidence interval
    alpha = 1 - ci
    ci_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))

    return StatisticalResult(
        point_estimate=float(point_estimate),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        ci_level=ci,
        n_bootstrap=n_bootstrap,
        bootstrap_estimates=bootstrap_estimates,
    )


def permutation_test(
    group_a: np.ndarray | list[float],
    group_b: np.ndarray | list[float],
    n_permutations: int = 10000,
    random_seed: int | None = None,
) -> StatisticalResult:
    """
    Non-parametric permutation test for difference in means.

    Args:
        group_a: First group
        group_b: Second group
        n_permutations: Number of permutations
        random_seed: Random seed

    Returns:
        StatisticalResult with p_value, observed_diff, null_distribution
    """
    group_a = np.asarray(group_a)
    group_b = np.asarray(group_b)
    rng = np.random.default_rng(random_seed)

    # Observed difference
    observed_diff = np.mean(group_a) - np.mean(group_b)

    # Combine groups
    combined = np.concatenate([group_a, group_b])
    n_a = len(group_a)

    # Permutation testing
    null_diffs = np.empty(n_permutations)

    for i in range(n_permutations):
        shuffled = rng.permutation(combined)
        perm_a = shuffled[:n_a]
        perm_b = shuffled[n_a:]
        null_diffs[i] = np.mean(perm_a) - np.mean(perm_b)

    # Two-tailed p-value
    p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))

    return StatisticalResult(
        p_value=float(p_value),
        observed_diff=float(observed_diff),
        null_distribution=null_diffs,
        n_permutations=n_permutations,
        significant=p_value < 0.05,
    )


def welch_t_test(
    group_a: np.ndarray | list[float], group_b: np.ndarray | list[float]
) -> StatisticalResult:
    """
    Welch's t-test for unequal variances.

    Args:
        group_a: First group
        group_b: Second group

    Returns:
        StatisticalResult with t_statistic, p_value, df
    """
    group_a = np.asarray(group_a)
    group_b = np.asarray(group_b)

    result = stats.ttest_ind(group_a, group_b, equal_var=False)

    return StatisticalResult(
        t_statistic=float(result.statistic),
        p_value=float(result.pvalue),
        df=float(result.df) if hasattr(result, "df") else None,
        significant=result.pvalue < 0.05,
    )


def mann_whitney_u(
    group_a: np.ndarray | list[float], group_b: np.ndarray | list[float]
) -> StatisticalResult:
    """
    Mann-Whitney U test (non-parametric alternative to t-test).

    Args:
        group_a: First group
        group_b: Second group

    Returns:
        StatisticalResult with u_statistic, p_value
    """
    group_a = np.asarray(group_a)
    group_b = np.asarray(group_b)

    result = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")

    return StatisticalResult(
        u_statistic=float(result.statistic),
        p_value=float(result.pvalue),
        significant=result.pvalue < 0.05,
    )


def cohens_d(
    group_a: np.ndarray | list[float], group_b: np.ndarray | list[float]
) -> StatisticalResult:
    """
    Cohen's d effect size.

    Args:
        group_a: First group
        group_b: Second group

    Returns:
        StatisticalResult with d, interpretation
    """
    group_a = np.asarray(group_a)
    group_b = np.asarray(group_b)

    mean_a = np.mean(group_a)
    mean_b = np.mean(group_b)
    std_a = np.std(group_a, ddof=1)
    std_b = np.std(group_b, ddof=1)
    n_a = len(group_a)
    n_b = len(group_b)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))

    d = (mean_a - mean_b) / pooled_std

    # Interpretation
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return StatisticalResult(
        d=float(d),
        abs_d=float(abs_d),
        interpretation=interpretation,
        pooled_std=float(pooled_std),
    )


def cliff_delta(
    group_a: np.ndarray | list[float], group_b: np.ndarray | list[float]
) -> StatisticalResult:
    """
    Cliff's delta (non-parametric effect size).

    Args:
        group_a: First group
        group_b: Second group

    Returns:
        StatisticalResult with delta, interpretation
    """
    group_a = np.asarray(group_a)
    group_b = np.asarray(group_b)

    n_a = len(group_a)
    n_b = len(group_b)

    # Count pairs where a > b, a < b
    greater = 0
    less = 0

    for a_val in group_a:
        for b_val in group_b:
            if a_val > b_val:
                greater += 1
            elif a_val < b_val:
                less += 1

    delta = (greater - less) / (n_a * n_b)

    # Interpretation
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        interpretation = "negligible"
    elif abs_delta < 0.33:
        interpretation = "small"
    elif abs_delta < 0.474:
        interpretation = "medium"
    else:
        interpretation = "large"

    return StatisticalResult(
        delta=float(delta),
        abs_delta=float(abs_delta),
        interpretation=interpretation,
    )


def holm_bonferroni(p_values: np.ndarray | list[float]) -> StatisticalResult:
    """
    Holm-Bonferroni multiple comparison correction.

    Args:
        p_values: List of p-values

    Returns:
        StatisticalResult with adjusted_p_values, significant
    """
    p_values = np.asarray(p_values)
    n = len(p_values)

    # Sort p-values
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # Holm-Bonferroni correction
    adjusted_p = np.zeros(n)
    for i in range(n):
        adjusted_p[sorted_indices[i]] = min(sorted_p[i] * (n - i), 1.0)

    # Enforce monotonicity
    for i in range(1, n):
        if adjusted_p[sorted_indices[i]] < adjusted_p[sorted_indices[i - 1]]:
            adjusted_p[sorted_indices[i]] = adjusted_p[sorted_indices[i - 1]]

    significant = adjusted_p < 0.05

    return StatisticalResult(
        original_p_values=p_values,
        adjusted_p_values=adjusted_p,
        significant=significant,
        n_tests=n,
    )


def benjamini_hochberg(p_values: np.ndarray | list[float], fdr: float = 0.05) -> StatisticalResult:
    """
    Benjamini-Hochberg FDR correction.

    Args:
        p_values: List of p-values
        fdr: False discovery rate threshold

    Returns:
        StatisticalResult with adjusted_p_values, significant
    """
    p_values = np.asarray(p_values)
    n = len(p_values)

    # Sort p-values
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # Benjamini-Hochberg correction
    adjusted_p = np.zeros(n)
    for i in range(n):
        adjusted_p[sorted_indices[i]] = min(sorted_p[i] * n / (i + 1), 1.0)

    # Enforce monotonicity (from right to left)
    for i in range(n - 2, -1, -1):
        if adjusted_p[sorted_indices[i]] > adjusted_p[sorted_indices[i + 1]]:
            adjusted_p[sorted_indices[i]] = adjusted_p[sorted_indices[i + 1]]

    significant = adjusted_p < fdr

    return StatisticalResult(
        original_p_values=p_values,
        adjusted_p_values=adjusted_p,
        significant=significant,
        n_tests=n,
        fdr=fdr,
    )


def iqr_outlier_detection(
    data: np.ndarray | list[float], iqr_multiplier: float = 1.5
) -> StatisticalResult:
    """
    Detect outliers using IQR method.

    Args:
        data: Input data
        iqr_multiplier: Multiplier for IQR (default: 1.5)

    Returns:
        StatisticalResult with outlier_mask, outlier_indices, bounds
    """
    data = np.asarray(data)

    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr

    outlier_mask = (data < lower_bound) | (data > upper_bound)
    outlier_indices = np.where(outlier_mask)[0]

    return StatisticalResult(
        outlier_mask=outlier_mask,
        outlier_indices=outlier_indices,
        n_outliers=int(np.sum(outlier_mask)),
        lower_bound=float(lower_bound),
        upper_bound=float(upper_bound),
        q1=float(q1),
        q3=float(q3),
        iqr=float(iqr),
    )

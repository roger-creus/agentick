"""Tests for statistical analysis functions."""

import numpy as np

from agentick.analysis.statistics import (
    benjamini_hochberg,
    bootstrap_ci,
    cliff_delta,
    cohens_d,
    holm_bonferroni,
    iqr_outlier_detection,
    mann_whitney_u,
    permutation_test,
    welch_t_test,
)


def test_bootstrap_ci():
    """Test bootstrap confidence interval."""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    result = bootstrap_ci(data, n_bootstrap=1000, ci=0.95)

    # CI should contain the mean
    mean = np.mean(data)
    assert result.ci_lower <= mean <= result.ci_upper

    # CI should be reasonable
    assert result.ci_lower >= np.min(data)
    assert result.ci_upper <= np.max(data)


def test_permutation_test():
    """Test permutation test."""
    rng = np.random.default_rng(42)

    # Same distribution - should not be significant
    group_a = rng.normal(0, 1, 50)
    group_b = rng.normal(0, 1, 50)

    result = permutation_test(group_a, group_b, n_permutations=1000)

    assert 0 <= result.p_value <= 1
    assert result.p_value > 0.05  # Should not be significant
    assert not result.significant

    # Different distributions - should be significant
    group_c = rng.normal(5, 1, 50)
    result2 = permutation_test(group_a, group_c, n_permutations=1000)

    assert result2.p_value < 0.05  # Should be significant
    assert result2.significant


def test_welch_t_test():
    """Test Welch's t-test."""
    # Same distribution (fixed seed for reproducibility)
    rng = np.random.default_rng(42)
    group_a = rng.normal(0, 1, 50)
    group_b = rng.normal(0, 1, 50)

    result = welch_t_test(group_a, group_b)

    assert isinstance(result.t_statistic, float)
    assert 0 <= result.p_value <= 1
    assert result.p_value > 0.05  # Should not be significant
    assert not result.significant

    # Different distributions
    group_c = rng.normal(5, 1, 50)
    result2 = welch_t_test(group_a, group_c)

    assert result2.p_value < 0.001  # Should be highly significant
    assert result2.significant


def test_mann_whitney_u():
    """Test Mann-Whitney U test."""
    group_a = np.array([1, 2, 3, 4, 5])
    group_b = np.array([6, 7, 8, 9, 10])

    result = mann_whitney_u(group_a, group_b)

    assert isinstance(result.u_statistic, float)
    assert 0 <= result.p_value <= 1
    assert result.p_value < 0.05  # Should be significant
    assert result.significant


def test_cohens_d():
    """Test Cohen's d effect size."""
    group_a = np.array([1, 2, 3, 4, 5])
    group_b = np.array([6, 7, 8, 9, 10])

    result = cohens_d(group_a, group_b)

    assert isinstance(result.d, float)
    # Large negative effect (group_a < group_b)
    assert result.d < -0.5
    assert result.interpretation == "large"


def test_cliff_delta():
    """Test Cliff's Delta effect size."""
    group_a = np.array([1, 2, 3, 4, 5])
    group_b = np.array([6, 7, 8, 9, 10])

    result = cliff_delta(group_a, group_b)

    assert isinstance(result.delta, float)
    assert -1 <= result.delta <= 1
    # Large negative effect (all of group_a < group_b)
    assert result.delta == -1
    assert result.interpretation == "large"


def test_holm_bonferroni():
    """Test Holm-Bonferroni correction."""
    p_values = [0.01, 0.03, 0.05, 0.10]

    result = holm_bonferroni(p_values)

    assert len(result.significant) == len(p_values)
    # First should be significant
    assert result.significant[0] == True  # noqa: E712
    # Last should not be significant
    assert result.significant[-1] == False  # noqa: E712


def test_benjamini_hochberg():
    """Test Benjamini-Hochberg FDR correction."""
    p_values = [0.01, 0.02, 0.03, 0.10]

    result = benjamini_hochberg(p_values, fdr=0.05)

    assert len(result.significant) == len(p_values)
    # Should have some significant results
    assert any(result.significant)


def test_iqr_outlier_detection():
    """Test IQR outlier detection."""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])

    result = iqr_outlier_detection(data, iqr_multiplier=1.5)

    assert len(result.outlier_indices) > 0
    # 100 should be detected as outlier (index 9)
    assert 9 in result.outlier_indices
    assert result.outlier_mask[-1]  # Last element is outlier


def test_bootstrap_ci_consistency():
    """Test bootstrap CI is consistent across runs with same seed."""
    data = np.array([1, 2, 3, 4, 5])

    ci1 = bootstrap_ci(data, n_bootstrap=1000, random_seed=42)
    ci2 = bootstrap_ci(data, n_bootstrap=1000, random_seed=42)

    assert ci1.point_estimate == ci2.point_estimate
    assert ci1.ci_lower == ci2.ci_lower
    assert ci1.ci_upper == ci2.ci_upper


def test_empty_input_handling():
    """Test handling of empty inputs."""
    # empty = np.array([])  # Reserved for future empty input testing

    # Empty data should raise or produce sensible results
    # For now just check it doesn't crash with valid inputs
    data = np.array([1, 2, 3])
    result = bootstrap_ci(data, n_bootstrap=100)
    assert result.point_estimate > 0

"""Learning curve analysis utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter1d

from agentick.analysis.statistics import bootstrap_ci


def compute_learning_curve(
    episode_returns: list[np.ndarray], window_size: int = 100, ci: float = 0.95
) -> dict[str, Any]:
    """
    Compute smoothed learning curve with CI bands.

    Args:
        episode_returns: List of return arrays (one per seed)
        window_size: Smoothing window size
        ci: Confidence level

    Returns:
        Dict with mean_curve, ci_lower, ci_upper, raw_curves
    """
    # Convert to numpy array
    curves = np.array(episode_returns)  # shape: (n_seeds, n_episodes)

    # Smooth each curve
    smoothed_curves = np.zeros_like(curves)
    for i in range(len(curves)):
        smoothed_curves[i] = uniform_filter1d(curves[i], size=window_size, mode="nearest")

    # Compute mean and CI at each time point
    n_episodes = curves.shape[1]
    mean_curve = np.mean(smoothed_curves, axis=0)
    ci_lower = np.zeros(n_episodes)
    ci_upper = np.zeros(n_episodes)

    for t in range(n_episodes):
        returns_at_t = smoothed_curves[:, t]
        result = bootstrap_ci(returns_at_t, ci=ci)
        ci_lower[t] = result.ci_lower
        ci_upper[t] = result.ci_upper

    return {
        "mean_curve": mean_curve,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "raw_curves": curves,
        "smoothed_curves": smoothed_curves,
        "window_size": window_size,
        "n_seeds": len(curves),
        "n_episodes": n_episodes,
    }


def estimate_convergence_point(
    curve: np.ndarray, threshold: float = 0.95, window: int = 100
) -> dict[str, Any]:
    """
    Estimate when agent reaches X% of final performance.

    Args:
        curve: Learning curve (returns over time)
        threshold: Fraction of final performance (e.g., 0.95)
        window: Window for computing final performance

    Returns:
        Dict with convergence_episode, convergence_value, final_performance
    """
    curve = np.asarray(curve)

    # Final performance: mean of last window
    final_performance = np.mean(curve[-window:])

    # Target value
    target = threshold * final_performance

    # Find first episode where curve crosses and stays above target
    above_target = curve >= target

    if not np.any(above_target):
        return {
            "convergence_episode": None,
            "convergence_value": None,
            "final_performance": float(final_performance),
            "threshold": threshold,
        }

    # Find first sustained crossing
    convergence_episode = None
    for i in range(len(curve) - window):
        if np.all(above_target[i : i + window]):
            convergence_episode = i
            break

    if convergence_episode is not None:
        convergence_value = curve[convergence_episode]
    else:
        convergence_value = None

    return {
        "convergence_episode": convergence_episode,
        "convergence_value": float(convergence_value) if convergence_value else None,
        "final_performance": float(final_performance),
        "threshold": threshold,
    }


def compute_sample_efficiency(curves_dict: dict[str, np.ndarray]) -> dict[str, Any]:
    """
    Compare sample efficiency across agents.

    Args:
        curves_dict: Agent name -> learning curve

    Returns:
        Dict with per-agent convergence points and ranking
    """
    agent_convergence = {}

    for agent_name, curve in curves_dict.items():
        conv = estimate_convergence_point(curve)
        agent_convergence[agent_name] = conv

    # Rank by convergence speed (lower episode = more efficient)
    ranked = sorted(
        [
            (name, data["convergence_episode"])
            for name, data in agent_convergence.items()
            if data["convergence_episode"] is not None
        ],
        key=lambda x: x[1],
    )

    return {
        "per_agent": agent_convergence,
        "ranked": [{"agent": name, "episodes": ep} for name, ep in ranked],
    }


def plateau_detection(
    curve: np.ndarray, window: int = 100, threshold: float = 0.01
) -> dict[str, Any]:
    """
    Detect when learning has plateaued.

    Args:
        curve: Learning curve
        window: Window for detecting plateau
        threshold: Maximum improvement rate to be considered plateau

    Returns:
        Dict with plateau_detected, plateau_start, plateau_value
    """
    curve = np.asarray(curve)

    if len(curve) < window:
        return {
            "plateau_detected": False,
            "plateau_start": None,
            "plateau_value": None,
        }

    # Compute rolling improvement rate
    plateau_start = None

    for i in range(len(curve) - window):
        window_start = curve[i]
        window_end = curve[i + window]

        improvement_rate = (window_end - window_start) / window_start if window_start != 0 else 0

        if abs(improvement_rate) < threshold:
            plateau_start = i
            break

    if plateau_start is not None:
        plateau_value = np.mean(curve[plateau_start:])
        plateau_detected = True
    else:
        plateau_value = None
        plateau_detected = False

    return {
        "plateau_detected": plateau_detected,
        "plateau_start": plateau_start,
        "plateau_value": float(plateau_value) if plateau_value else None,
        "threshold": threshold,
        "window": window,
    }


def compute_auc(curve: np.ndarray, max_steps: int | None = None) -> dict[str, Any]:
    """
    Compute area under learning curve (normalized).

    Args:
        curve: Learning curve
        max_steps: Maximum steps to consider (default: full curve)

    Returns:
        Dict with auc, normalized_auc
    """
    curve = np.asarray(curve)

    if max_steps is not None:
        curve = curve[:max_steps]

    # Compute AUC using trapezoidal rule
    auc = np.trapz(curve)

    # Normalize by optimal AUC (if always at max return)
    max_return = np.max(curve)
    optimal_auc = max_return * len(curve)

    normalized_auc = auc / optimal_auc if optimal_auc > 0 else 0.0

    return {
        "auc": float(auc),
        "normalized_auc": float(normalized_auc),
        "n_steps": len(curve),
    }

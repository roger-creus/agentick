"""Human baseline analysis.

Analyze human performance data to establish baselines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class HumanBaselineAnalyzer:
    """Analyze human performance data to compute baselines."""

    def __init__(self, data_dir: str | Path = "human_data"):
        """
        Initialize analyzer.

        Args:
            data_dir: Directory containing human data
        """
        self.data_dir = Path(data_dir)
        self.sessions = self._load_all_sessions()

    def _load_all_sessions(self) -> list[dict[str, Any]]:
        """Load all session data."""
        from agentick.human.recorder import load_session_data

        return load_session_data(self.data_dir)

    def compute_task_baseline(
        self,
        task_name: str,
        difficulty: str | None = None,
    ) -> dict[str, float]:
        """
        Compute baseline statistics for task.

        Args:
            task_name: Task name
            difficulty: Optional difficulty level

        Returns:
            Dict with baseline statistics
        """
        # Filter episodes for this task
        episodes = []
        for session in self.sessions:
            for episode in session["episodes"]:
                if episode["task_name"] == task_name:
                    if difficulty is None or episode.get("difficulty") == difficulty:
                        # Only count non-practice episodes
                        if not episode.get("practice", False):
                            episodes.append(episode)

        if not episodes:
            return {
                "n_episodes": 0,
                "n_participants": 0,
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "median_reward": 0.0,
                "mean_steps": 0.0,
                "success_rate": 0.0,
            }

        rewards = [ep.get("total_reward", 0.0) for ep in episodes]
        steps = [ep.get("step_count", 0) for ep in episodes]
        successes = sum(1 for ep in episodes if ep.get("success", False))

        # Count unique participants
        participants = set(ep.get("participant_id") for ep in episodes)

        return {
            "n_episodes": len(episodes),
            "n_participants": len(participants),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "median_reward": float(np.median(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "mean_steps": float(np.mean(steps)),
            "std_steps": float(np.std(steps)),
            "median_steps": float(np.median(steps)),
            "success_rate": float(successes / len(episodes)),
        }

    def compute_efficiency_vs_optimal(
        self,
        task_name: str,
        optimal_steps: int,
        difficulty: str | None = None,
    ) -> dict[str, float]:
        """
        Compute human efficiency relative to optimal solution.

        Args:
            task_name: Task name
            optimal_steps: Optimal solution length
            difficulty: Optional difficulty level

        Returns:
            Dict with efficiency metrics
        """
        episodes = []
        for session in self.sessions:
            for episode in session["episodes"]:
                if episode["task_name"] == task_name:
                    if difficulty is None or episode.get("difficulty") == difficulty:
                        if not episode.get("practice", False) and episode.get("success", False):
                            episodes.append(episode)

        if not episodes:
            return {"efficiency_ratio": 0.0, "n_successful": 0}

        steps = [ep.get("step_count", 0) for ep in episodes]
        efficiency_ratios = [optimal_steps / max(s, 1) for s in steps]

        return {
            "efficiency_ratio": float(np.mean(efficiency_ratios)),
            "std_efficiency": float(np.std(efficiency_ratios)),
            "median_efficiency": float(np.median(efficiency_ratios)),
            "n_successful": len(episodes),
            "mean_steps": float(np.mean(steps)),
            "optimal_steps": optimal_steps,
        }

    def analyze_learning_curves(
        self,
        task_name: str,
        difficulty: str | None = None,
    ) -> dict[str, Any]:
        """
        Analyze learning curves across attempts.

        Tests if humans improve with practice.

        Args:
            task_name: Task name
            difficulty: Optional difficulty level

        Returns:
            Dict with learning curve analysis
        """
        # Group episodes by participant
        participant_episodes: dict[str, list[dict]] = {}

        for session in self.sessions:
            for episode in session["episodes"]:
                if episode["task_name"] == task_name:
                    if difficulty is None or episode.get("difficulty") == difficulty:
                        pid = episode.get("participant_id")
                        if pid:
                            if pid not in participant_episodes:
                                participant_episodes[pid] = []
                            participant_episodes[pid].append(episode)

        # Sort each participant's episodes by timestamp
        for pid in participant_episodes:
            participant_episodes[pid].sort(key=lambda x: x.get("timestamp", ""))

        # Compute learning metrics
        improvements = []
        for pid, episodes in participant_episodes.items():
            if len(episodes) >= 2:
                first_reward = episodes[0].get("total_reward", 0.0)
                last_reward = episodes[-1].get("total_reward", 0.0)
                improvement = last_reward - first_reward
                improvements.append(improvement)

        if not improvements:
            return {
                "shows_learning": False,
                "mean_improvement": 0.0,
                "n_participants": 0,
            }

        return {
            "shows_learning": np.mean(improvements) > 0,
            "mean_improvement": float(np.mean(improvements)),
            "std_improvement": float(np.std(improvements)),
            "n_participants": len(improvements),
            "pct_improved": float(sum(1 for x in improvements if x > 0) / len(improvements)),
        }

    def analyze_strategy_diversity(
        self,
        task_name: str,
        difficulty: str | None = None,
    ) -> dict[str, Any]:
        """
        Analyze diversity of human strategies.

        Uses step count variance as proxy for strategy diversity.

        Args:
            task_name: Task name
            difficulty: Optional difficulty level

        Returns:
            Dict with diversity metrics
        """
        episodes = []
        for session in self.sessions:
            for episode in session["episodes"]:
                if episode["task_name"] == task_name:
                    if difficulty is None or episode.get("difficulty") == difficulty:
                        if episode.get("success", False):
                            episodes.append(episode)

        if len(episodes) < 2:
            return {"diversity_score": 0.0, "n_episodes": len(episodes)}

        steps = [ep.get("step_count", 0) for ep in episodes]
        durations = [ep.get("duration", 0.0) for ep in episodes]

        # Compute coefficient of variation as diversity measure
        cv_steps = np.std(steps) / np.mean(steps) if np.mean(steps) > 0 else 0.0
        cv_duration = np.std(durations) / np.mean(durations) if np.mean(durations) > 0 else 0.0

        return {
            "diversity_score": float((cv_steps + cv_duration) / 2),
            "step_variability": float(cv_steps),
            "duration_variability": float(cv_duration),
            "n_episodes": len(episodes),
        }

    def generate_full_report(self) -> dict[str, Any]:
        """
        Generate comprehensive report of all human baseline data.

        Returns:
            Dict with full analysis
        """
        # Get all unique tasks
        tasks = set()
        for session in self.sessions:
            for episode in session["episodes"]:
                task_name = episode.get("task_name")
                difficulty = episode.get("difficulty")
                if task_name:
                    tasks.add((task_name, difficulty))

        report = {
            "n_sessions": len(self.sessions),
            "n_participants": len(set(s.get("participant_id") for s in self.sessions)),
            "tasks": {},
        }

        for task_name, difficulty in tasks:
            key = f"{task_name}_{difficulty}" if difficulty else task_name

            report["tasks"][key] = {
                "baseline": self.compute_task_baseline(task_name, difficulty),
                "learning": self.analyze_learning_curves(task_name, difficulty),
                "diversity": self.analyze_strategy_diversity(task_name, difficulty),
            }

        return report


def estimate_human_baseline(
    task_name: str,
    optimal_reward: float,
    task_complexity: float = 0.5,
) -> dict[str, float]:
    """
    Estimate human baseline when no data is available.

    Uses task characteristics to estimate expected human performance.

    Args:
        task_name: Task name
        optimal_reward: Optimal/maximum reward
        task_complexity: Complexity estimate (0=trivial, 1=very hard)

    Returns:
        Dict with estimated baseline
    """
    # Heuristic: humans achieve 60-90% of optimal depending on complexity
    efficiency_factor = 0.9 - (0.3 * task_complexity)

    estimated_mean = optimal_reward * efficiency_factor
    estimated_std = optimal_reward * 0.1 * (1 + task_complexity)

    # Success rate decreases with complexity
    estimated_success_rate = max(0.3, 1.0 - (0.5 * task_complexity))

    return {
        "estimated": True,
        "mean_reward": estimated_mean,
        "std_reward": estimated_std,
        "success_rate": estimated_success_rate,
        "complexity": task_complexity,
        "note": "Estimated baseline - collect real human data for accurate baselines",
    }

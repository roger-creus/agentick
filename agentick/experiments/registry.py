"""Experiment registry for saving, loading, and comparing results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentick.experiments.runner import ExperimentResults


class ExperimentRegistry:
    """Registry for managing experiment results."""

    def __init__(self, base_dir: str | Path = "results"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def list_experiments(self, tag: str | None = None) -> list[dict[str, Any]]:
        """
        List all experiments, optionally filtered by tag.

        Args:
            tag: Optional tag to filter by

        Returns:
            List of experiment metadata dicts
        """
        experiments = []

        for exp_dir in self.base_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            metadata_path = exp_dir / "metadata.json"
            config_path = exp_dir / "config.yaml"

            if not metadata_path.exists() or not config_path.exists():
                continue

            with open(metadata_path) as f:
                metadata = json.load(f)

            # Load config to check tags
            from agentick.experiments.config import ExperimentConfig

            config = ExperimentConfig.from_yaml(config_path)

            # Filter by tag if specified
            if tag is not None:
                if tag not in config.tags:
                    continue

            experiments.append(
                {
                    "name": config.name,
                    "path": str(exp_dir),
                    "timestamp": metadata.get("timestamp"),
                    "tags": config.tags,
                    "agent_type": config.agent.type,
                }
            )

        # Sort by timestamp (most recent first)
        experiments.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        return experiments

    def load_experiment(self, name: str) -> ExperimentResults:
        """
        Load experiment by name.

        Args:
            name: Experiment name or path

        Returns:
            ExperimentResults
        """
        # Try as direct path first
        path = Path(name)
        if path.exists() and path.is_dir():
            return ExperimentResults.load(path)

        # Try finding in base_dir
        for exp_dir in self.base_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            config_path = exp_dir / "config.yaml"
            if not config_path.exists():
                continue

            from agentick.experiments.config import ExperimentConfig

            config = ExperimentConfig.from_yaml(config_path)

            if config.name == name:
                return ExperimentResults.load(exp_dir)

        raise ValueError(f"Experiment not found: {name}")

    def load_latest(self, name: str) -> ExperimentResults:
        """
        Load most recent run of experiment by name.

        Args:
            name: Experiment name

        Returns:
            ExperimentResults of most recent run
        """
        matching_runs = []

        for exp_dir in self.base_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            config_path = exp_dir / "config.yaml"
            if not config_path.exists():
                continue

            from agentick.experiments.config import ExperimentConfig

            config = ExperimentConfig.from_yaml(config_path)

            if config.name == name:
                metadata_path = exp_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    matching_runs.append((metadata.get("timestamp", ""), exp_dir))

        if not matching_runs:
            raise ValueError(f"No runs found for experiment: {name}")

        # Sort by timestamp and get most recent
        matching_runs.sort(reverse=True)
        latest_dir = matching_runs[0][1]

        return ExperimentResults.load(latest_dir)

    def compare_experiments(
        self, names: list[str], output_dir: str | Path | None = None
    ) -> dict[str, Any]:
        """
        Compare multiple experiments.

        Args:
            names: List of experiment names or paths
            output_dir: Optional directory to save comparison

        Returns:
            Comparison results dict
        """
        results = [self.load_experiment(name) for name in names]

        comparison = {
            "experiments": [
                {
                    "name": r.config.name,
                    "agent_type": r.config.agent.type,
                    "timestamp": r.metadata.get("timestamp"),
                }
                for r in results
            ],
            "summary": {},
            "per_task": {},
        }

        # Compare summary metrics
        for metric in ["mean_return", "success_rate", "mean_length"]:
            comparison["summary"][metric] = {r.config.name: r.summary.get(metric) for r in results}

        # Compare per-task metrics
        all_tasks = set()
        for r in results:
            all_tasks.update(r.per_task_results.keys())

        for task in sorted(all_tasks):
            comparison["per_task"][task] = {}

            for metric in ["mean_return", "success_rate"]:
                task_comparison = {}

                for r in results:
                    if task in r.per_task_results:
                        task_results = r.per_task_results[task]
                        agg_metrics = task_results.get("aggregate_metrics", {})
                        task_comparison[r.config.name] = agg_metrics.get(metric)

                comparison["per_task"][task][metric] = task_comparison

        # Save comparison if output_dir specified
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            comparison_path = output_dir / "comparison.json"
            with open(comparison_path, "w") as f:
                json.dump(comparison, f, indent=2)

            print(f"Comparison saved to: {comparison_path}")

        return comparison

    def query(self, **filters: Any) -> list[ExperimentResults]:
        """
        Query experiments by filters.

        Args:
            **filters: Filters to apply (e.g., agent_type="ppo", tag="navigation")

        Returns:
            List of matching ExperimentResults
        """
        results = []

        for exp_dir in self.base_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            config_path = exp_dir / "config.yaml"
            if not config_path.exists():
                continue

            from agentick.experiments.config import ExperimentConfig

            config = ExperimentConfig.from_yaml(config_path)

            # Check filters
            match = True

            if "agent_type" in filters:
                if config.agent.type != filters["agent_type"]:
                    match = False

            if "tag" in filters:
                if filters["tag"] not in config.tags:
                    match = False

            if "name" in filters:
                if config.name != filters["name"]:
                    match = False

            if match:
                results.append(ExperimentResults.load(exp_dir))

        return results


# Global registry instance
_default_registry = None


def get_registry(base_dir: str | Path = "results") -> ExperimentRegistry:
    """Get or create default registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ExperimentRegistry(base_dir)
    return _default_registry

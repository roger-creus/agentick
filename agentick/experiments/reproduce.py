"""Reproducibility utilities for experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from agentick.experiments.config import ExperimentConfig
from agentick.experiments.runner import ExperimentResults, run_experiment


def reproduce(results_dir: str | Path) -> ExperimentResults:
    """
    Reproduce an experiment from saved results.

    Args:
        results_dir: Directory containing saved results

    Returns:
        New ExperimentResults from re-run
    """
    results_dir = Path(results_dir)

    # Load original config
    config_path = results_dir / "config.yaml"
    if not config_path.exists():
        raise ValueError(f"Config not found in {results_dir}")

    config = ExperimentConfig.from_yaml(config_path)

    print(f"Reproducing experiment: {config.name}")
    print(f"Original results: {results_dir}")

    # Run experiment with same config
    new_results = run_experiment(config)

    print(f"New results: {new_results.output_dir}")

    return new_results


def verify(
    original_dir: str | Path,
    reproduced_dir: str | Path,
    rtol: float = 1e-2,
    atol: float = 1e-3,
) -> dict[str, Any]:
    """
    Verify that reproduced results match original within tolerance.

    Args:
        original_dir: Original results directory
        reproduced_dir: Reproduced results directory
        rtol: Relative tolerance for numpy.allclose
        atol: Absolute tolerance for numpy.allclose

    Returns:
        Verification report dict
    """
    original = ExperimentResults.load(original_dir)
    reproduced = ExperimentResults.load(reproduced_dir)

    report = {
        "original_dir": str(original_dir),
        "reproduced_dir": str(reproduced_dir),
        "config_match": True,
        "summary_match": True,
        "per_task_match": True,
        "differences": [],
    }

    # Check config match
    if original.config.name != reproduced.config.name:
        report["config_match"] = False
        report["differences"].append(
            f"Config name mismatch: {original.config.name} vs {reproduced.config.name}"
        )

    # Check summary metrics
    for metric in ["mean_return", "success_rate", "mean_length"]:
        if metric in original.summary and metric in reproduced.summary:
            orig_val = original.summary[metric]
            repro_val = reproduced.summary[metric]

            if not np.allclose(orig_val, repro_val, rtol=rtol, atol=atol):
                report["summary_match"] = False
                report["differences"].append(
                    f"Summary {metric}: {orig_val:.4f} vs {repro_val:.4f} "
                    f"(diff: {abs(orig_val - repro_val):.4f})"
                )

    # Check per-task metrics
    all_tasks = set(original.per_task_results.keys()) | set(reproduced.per_task_results.keys())

    for task in all_tasks:
        if task not in original.per_task_results:
            report["per_task_match"] = False
            report["differences"].append(f"Task {task} missing in original")
            continue

        if task not in reproduced.per_task_results:
            report["per_task_match"] = False
            report["differences"].append(f"Task {task} missing in reproduced")
            continue

        orig_task = original.per_task_results[task]
        repro_task = reproduced.per_task_results[task]

        orig_metrics = orig_task.get("aggregate_metrics", {})
        repro_metrics = repro_task.get("aggregate_metrics", {})

        for metric in ["mean_return", "success_rate"]:
            if metric in orig_metrics and metric in repro_metrics:
                orig_val = orig_metrics[metric]
                repro_val = repro_metrics[metric]

                if not np.allclose(orig_val, repro_val, rtol=rtol, atol=atol):
                    report["per_task_match"] = False
                    report["differences"].append(
                        f"Task {task} {metric}: {orig_val:.4f} vs {repro_val:.4f} "
                        f"(diff: {abs(orig_val - repro_val):.4f})"
                    )

    # Overall pass/fail
    report["passed"] = (
        report["config_match"] and report["summary_match"] and report["per_task_match"]
    )

    return report


def diff(
    results_dir_a: str | Path,
    results_dir_b: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Compare two experiment runs and show what changed.

    Args:
        results_dir_a: First results directory
        results_dir_b: Second results directory
        output_path: Optional path to save diff report

    Returns:
        Diff report dict
    """
    results_a = ExperimentResults.load(results_dir_a)
    results_b = ExperimentResults.load(results_dir_b)

    diff_report = {
        "experiment_a": {
            "name": results_a.config.name,
            "path": str(results_dir_a),
            "timestamp": results_a.metadata.get("timestamp"),
        },
        "experiment_b": {
            "name": results_b.config.name,
            "path": str(results_dir_b),
            "timestamp": results_b.metadata.get("timestamp"),
        },
        "config_diff": _diff_configs(results_a.config, results_b.config),
        "summary_diff": _diff_dicts(results_a.summary, results_b.summary),
        "per_task_diff": {},
    }

    # Diff per-task results
    all_tasks = set(results_a.per_task_results.keys()) | set(results_b.per_task_results.keys())

    for task in sorted(all_tasks):
        task_a = results_a.per_task_results.get(task, {})
        task_b = results_b.per_task_results.get(task, {})

        metrics_a = task_a.get("aggregate_metrics", {})
        metrics_b = task_b.get("aggregate_metrics", {})

        task_diff = _diff_dicts(metrics_a, metrics_b)

        if task_diff:
            diff_report["per_task_diff"][task] = task_diff

    # Save if output path specified
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(diff_report, f, indent=2)

        print(f"Diff report saved to: {output_path}")

    return diff_report


def _diff_configs(config_a: ExperimentConfig, config_b: ExperimentConfig) -> dict[str, Any]:
    """Compare two configs."""
    dict_a = config_a.model_dump()
    dict_b = config_b.model_dump()

    return _diff_dicts(dict_a, dict_b)


def _diff_dicts(dict_a: dict[str, Any], dict_b: dict[str, Any]) -> dict[str, Any]:
    """Compare two dicts and return differences."""
    diff = {}

    all_keys = set(dict_a.keys()) | set(dict_b.keys())

    for key in all_keys:
        val_a = dict_a.get(key)
        val_b = dict_b.get(key)

        if val_a != val_b:
            diff[key] = {
                "a": val_a,
                "b": val_b,
                "change": _compute_change(val_a, val_b),
            }

    return diff


def _compute_change(val_a: Any, val_b: Any) -> Any:
    """Compute change between two values."""
    if val_a is None or val_b is None:
        return None

    if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
        diff_val = val_b - val_a
        if val_a != 0:
            pct = (diff_val / val_a) * 100
            return {"absolute": diff_val, "percent": pct}
        else:
            return {"absolute": diff_val, "percent": None}

    return None


# Aliases for backward compatibility
reproduce_experiment = reproduce
diff_experiments = diff


class ReproductionComparison:
    """Results of comparing two reproductions."""

    def __init__(self, results_a: ExperimentResults, results_b: ExperimentResults):
        self.results_a = results_a
        self.results_b = results_b
        self._compute_differences()

    def _compute_differences(self):
        """Compute differences between results."""
        self.max_diff = 0.0
        self.differing_tasks = []

        # Compare summary metrics
        for metric in ["mean_return", "success_rate"]:
            if metric in self.results_a.summary and metric in self.results_b.summary:
                diff = abs(self.results_a.summary[metric] - self.results_b.summary[metric])
                self.max_diff = max(self.max_diff, diff)

        # Compare per-task
        for task in self.results_a.per_task_results.keys():
            if task in self.results_b.per_task_results:
                metrics_a = self.results_a.per_task_results[task].get("aggregate_metrics", {})
                metrics_b = self.results_b.per_task_results[task].get("aggregate_metrics", {})

                for metric in ["mean_return", "success_rate"]:
                    if metric in metrics_a and metric in metrics_b:
                        diff = abs(metrics_a[metric] - metrics_b[metric])
                        if diff > 1e-10:
                            self.differing_tasks.append(task)
                            self.max_diff = max(self.max_diff, diff)
                            break

    def is_identical(self, tolerance: float = 1e-10) -> bool:
        """Check if results are identical within tolerance."""
        return self.max_diff < tolerance


def compare_reproductions(
    results_a: ExperimentResults | str | Path,
    results_b: ExperimentResults | str | Path,
) -> ReproductionComparison:
    """
    Compare two reproduction runs.

    Args:
        results_a: First results (or path to load)
        results_b: Second results (or path to load)

    Returns:
        Comparison object
    """
    if isinstance(results_a, (str, Path)):
        results_a = ExperimentResults.load(results_a)
    if isinstance(results_b, (str, Path)):
        results_b = ExperimentResults.load(results_b)

    return ReproductionComparison(results_a, results_b)

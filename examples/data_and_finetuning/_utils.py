"""Shared helpers for data collection and fine-tuning example scripts.

Provides reusable argparse argument groups, multi-task oracle data collection,
eval config generation, and eval execution via ExperimentRunner.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

ALL_DIFFICULTIES = ["easy", "medium", "hard", "expert"]


def add_task_args(parser: argparse.ArgumentParser) -> None:
    """Add --tasks and --difficulties arguments."""
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Task names (default: all oracle tasks)",
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        default=ALL_DIFFICULTIES,
        choices=ALL_DIFFICULTIES,
        help="Difficulty levels (default: all four)",
    )


def add_collection_args(parser: argparse.ArgumentParser) -> None:
    """Add data collection arguments."""
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Episodes per task/difficulty (default: 10)",
    )
    parser.add_argument(
        "--render-mode",
        default="language",
        choices=["language", "ascii", "language_structured", "rgb_array"],
        help="Observation render mode (default: language)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for collected data",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Starting seed offset (default: 0)",
    )


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    """Add evaluation arguments."""
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False,
        help="Run evaluation after training",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        default=False,
        help="Skip evaluation (default)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Episodes per task/difficulty during eval (default: 5)",
    )
    parser.add_argument(
        "--eval-seeds",
        type=int,
        default=3,
        help="Number of eval seeds (default: 3)",
    )
    parser.add_argument(
        "--harness",
        default="markovian_zero_shot",
        help="Harness preset for eval (default: markovian_zero_shot)",
    )


def add_hub_args(parser: argparse.ArgumentParser) -> None:
    """Add HuggingFace Hub arguments."""
    parser.add_argument(
        "--push-to-hub",
        default=None,
        metavar="REPO_ID",
        help="Push to HuggingFace Hub (e.g. user/my-model)",
    )


def resolve_tasks(tasks: list[str] | None) -> list[str]:
    """Resolve task list, defaulting to all oracle tasks."""
    if tasks is not None:
        return tasks
    from agentick.oracles import list_oracles

    return list_oracles()


def collect_multi_task_data(
    tasks: list[str],
    difficulties: list[str],
    n_episodes: int = 10,
    render_mode: str = "language",
    output_dir: str | Path = "trajectories/oracle",
    export_format: str = "conversation",
    seed_offset: int = 0,
    push_to_hub: str | None = None,
) -> Path:
    """Collect oracle trajectories for all task x difficulty combos.

    Uses DataCollector + get_oracle() for each combination, exports to
    HuggingFace format, and optionally merges into a combined dataset.

    Args:
        tasks: Task names to collect from.
        difficulties: Difficulty levels.
        n_episodes: Episodes per task/difficulty.
        render_mode: Observation render mode.
        output_dir: Base output directory.
        export_format: HuggingFace export format.
        seed_offset: Starting seed offset.
        push_to_hub: Optional HuggingFace repo ID to push combined dataset.

    Returns:
        Path to the combined HuggingFace dataset directory.
    """
    import agentick
    from agentick.data import DataCollector
    from agentick.oracles import get_oracle

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_hf_paths: list[Path] = []
    total_episodes = 0

    for task_name in tasks:
        for difficulty in difficulties:
            print(f"\nCollecting {task_name} @ {difficulty}...")
            try:
                env = agentick.make(task_name, difficulty=difficulty, render_mode=render_mode)
            except Exception as e:
                print(f"  Skipping {task_name} @ {difficulty}: {e}")
                continue

            try:
                oracle = get_oracle(task_name, env)
            except Exception as e:
                print(f"  No oracle for {task_name}: {e}")
                env.close()
                continue

            modalities = [render_mode] if render_mode != "rgb_array" else ["rgb_array"]
            collector = DataCollector(env, oracle, record_modalities=modalities)
            seeds = range(seed_offset, seed_offset + n_episodes)
            dataset = collector.collect(num_episodes=n_episodes, seeds=seeds, show_progress=False)

            # Print summary
            successes = sum(1 for t in dataset.trajectories if t.success)
            print(
                f"  {len(dataset.trajectories)} episodes, "
                f"{successes}/{len(dataset.trajectories)} success"
            )

            # Export per-task/difficulty
            safe_name = task_name.replace("-", "_").lower()
            hf_subdir = output_dir / f"hf_{safe_name}_{difficulty}"
            dataset.export_to_huggingface(hf_subdir, format=export_format)
            all_hf_paths.append(hf_subdir)
            total_episodes += len(dataset.trajectories)

            env.close()

    if not all_hf_paths:
        raise RuntimeError("No data collected. Check task names and difficulties.")

    # Merge all sub-datasets into one combined dataset
    combined_path = output_dir / "hf_combined"
    try:
        from datasets import Dataset, concatenate_datasets

        sub_datasets = []
        for p in all_hf_paths:
            sub_datasets.append(Dataset.load_from_disk(str(p)))

        combined = concatenate_datasets(sub_datasets)
        combined.save_to_disk(str(combined_path))
        print(f"\nCombined dataset: {len(combined)} examples -> {combined_path}")
    except ImportError:
        print("\nWARNING: 'datasets' not installed, skipping merge.")
        print("Individual datasets saved to sub-directories.")
        combined_path = all_hf_paths[0]

    # Optionally push to Hub
    if push_to_hub:
        try:
            from datasets import Dataset

            ds = Dataset.load_from_disk(str(combined_path))
            ds.push_to_hub(push_to_hub)
            print(f"Pushed dataset to https://huggingface.co/datasets/{push_to_hub}")
        except Exception as e:
            print(f"WARNING: Failed to push to Hub: {e}")

    print(f"\nTotal: {total_episodes} episodes across {len(all_hf_paths)} task/difficulty combos")
    return combined_path


def build_eval_config(
    name: str,
    model_path: str,
    tasks: list[str],
    difficulties: list[str],
    harness: str = "markovian_zero_shot",
    render_mode: str = "language",
    n_episodes: int = 5,
    n_seeds: int = 3,
    output_dir: str = "results",
) -> dict[str, Any]:
    """Build an ExperimentConfig dict matching the full_benchmark YAML format.

    Args:
        name: Experiment name.
        model_path: Path or HF repo ID for the trained model.
        tasks: Task names to evaluate on.
        difficulties: Difficulty levels.
        harness: Harness preset name.
        render_mode: Observation render mode.
        n_episodes: Episodes per task/difficulty.
        n_seeds: Number of random seeds.
        output_dir: Results output directory.

    Returns:
        Config dict suitable for ExperimentConfig.from_dict().
    """
    return {
        "name": name,
        "description": f"Eval of {model_path}",
        "agent": {
            "type": "llm",
            "hyperparameters": {
                "model_path": model_path,
                "harness": harness,
                "observation_modes": [render_mode],
            },
        },
        "tasks": tasks,
        "difficulties": difficulties,
        "n_episodes": n_episodes,
        "n_seeds": n_seeds,
        "render_modes": [render_mode],
        "output_dir": output_dir,
        "metrics": ["mean_return", "success_rate", "mean_length"],
    }


def save_eval_config(config_dict: dict[str, Any], path: str | Path) -> Path:
    """Save an eval config dict as YAML.

    Args:
        config_dict: Config dict from build_eval_config().
        path: Output YAML path.

    Returns:
        Path to the saved YAML file.
    """
    from agentick.experiments.config import ExperimentConfig

    path = Path(path)
    config = ExperimentConfig.from_dict(config_dict)
    config.to_yaml(path)
    print(f"Eval config saved to {path}")
    print(f"  Re-run with: python -m agentick.experiments.run --config {path}")
    return path


def run_eval(config_dict: dict[str, Any]) -> Any:
    """Run evaluation using ExperimentRunner.

    Args:
        config_dict: Config dict from build_eval_config().

    Returns:
        ExperimentResults from the run.
    """
    from agentick.experiments.config import ExperimentConfig
    from agentick.experiments.runner import ExperimentRunner

    config = ExperimentConfig.from_dict(config_dict)
    runner = ExperimentRunner(config)
    results = runner.run()
    return results

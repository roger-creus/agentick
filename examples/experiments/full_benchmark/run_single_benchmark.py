"""Run a single benchmark experiment from config.

This script loads a YAML configuration file and runs a benchmark experiment,
saving results, videos, and interaction traces.

Usage:
    uv run python examples/experiments/full_benchmark/run_single_benchmark.py configs/random_agent.yaml
    uv run python examples/experiments/full_benchmark/run_single_benchmark.py configs/greedy_baseline.yaml

For LLM agents (requires API keys):
    uv run python examples/experiments/full_benchmark/run_single_benchmark.py configs/openai_text.yaml
    uv run python examples/experiments/full_benchmark/run_single_benchmark.py configs/anthropic_text.yaml
"""

import argparse
import sys
from pathlib import Path

from agentick.leaderboard.experiment import ExperimentRunner


def main():
    """Run a single benchmark experiment."""
    parser = argparse.ArgumentParser(
        description="Run a single benchmark experiment from config"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to experiment config YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: results/<experiment_name>)",
    )
    args = parser.parse_args()

    # Check config exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        print("\nAvailable configs in full_benchmark/configs/:")
        configs_dir = Path("examples/experiments/full_benchmark/configs")
        if configs_dir.exists():
            for cfg in sorted(configs_dir.glob("*.yaml")):
                print(f"  - {cfg.name}")
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Use experiment name from config
        import yaml
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        exp_name = config_data.get("name", "experiment")
        output_dir = f"results/full_benchmark/{exp_name}"

    print("=" * 80)
    print("FULL BENCHMARK - SINGLE EXPERIMENT")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print()

    # Create runner
    runner = ExperimentRunner(
        config_path=str(config_path),
        output_dir=output_dir,
    )

    # Run experiment
    print("Starting experiment...")
    print()
    results = runner.run()

    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Total episodes: {len(results)}")
    print(f"Results saved to: {output_dir}/")
    print()

    # Compute statistics
    if results:
        avg_reward = sum(r.get("total_reward", 0) for r in results) / len(results)
        avg_steps = sum(r.get("steps", 0) for r in results) / len(results)
        success_rate = sum(1 for r in results if r.get("success", False)) / len(results)

        print(f"Average reward: {avg_reward:.2f}")
        print(f"Average steps: {avg_steps:.1f}")
        print(f"Success rate: {success_rate:.1%}")
        print()

        # Per-task breakdown
        print("Per-task results:")
        task_results = {}
        for r in results:
            task = r.get("task", "unknown")
            if task not in task_results:
                task_results[task] = []
            task_results[task].append(r)

        for task, task_res in sorted(task_results.items()):
            avg_rew = sum(r.get("total_reward", 0) for r in task_res) / len(task_res)
            success = sum(1 for r in task_res if r.get("success", False)) / len(task_res)
            print(f"  {task}: reward={avg_rew:.2f}, success={success:.1%}")

    print()
    print("💡 Next steps:")
    print("  - View videos: ls -lh " + output_dir + "/videos/")
    print("  - View traces: ls -lh " + output_dir + "/traces/")
    print("  - Generate plots: uv run python examples/experiments/full_benchmark/plot_all_results.py")


if __name__ == "__main__":
    main()

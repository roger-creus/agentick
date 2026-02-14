"""
Build and run a custom experiment programmatically.

This example demonstrates:
- Creating experiment configs in code
- Running custom experiments
- Customizing evaluation parameters

Requirements:
    uv sync --extra all

Usage:
    uv run python examples/experiments/run_custom.py
"""

from agentick.leaderboard.experiment import ExperimentConfig, ExperimentRunner


def main():
    """Run custom experiment."""
    print("Running Custom Experiment")
    print("=" * 80)

    # Build config programmatically
    config = ExperimentConfig(
        name="custom_navigation_test",
        description="Custom test on navigation tasks",
        tasks=[
            "GoToGoal-v0",
            "MazeNavigation-v0",
            "KeyDoorPuzzle-v0",
        ],
        agent_type="random",
        num_seeds=3,
        episodes_per_seed=1,
        render_mode="ascii",
        timeout=100,
    )

    print(f"\nExperiment: {config.name}")
    print(f"Tasks: {', '.join(config.tasks)}")
    print(f"Agent: {config.agent_type}")
    print(f"Seeds: {config.num_seeds}")
    print(f"Episodes per seed: {config.episodes_per_seed}")
    print()

    # Save config to file (optional)
    config_path = "results/custom_experiment_config.yaml"
    from pathlib import Path

    import yaml

    Path("results").mkdir(exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(vars(config), f)

    print(f"Config saved to: {config_path}")
    print()

    # Create runner
    runner = ExperimentRunner(
        config_path=config_path,
        output_dir="results/custom_experiment",
    )

    # Run experiment
    print("Running experiment...")
    results = runner.run()

    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    if results:
        # Group by task
        task_results = {}
        for result in results:
            task = result.get('task', 'unknown')
            if task not in task_results:
                task_results[task] = []
            task_results[task].append(result)

        print("\nResults by task:")
        print("-" * 80)

        for task, task_res in task_results.items():
            avg_reward = sum(r.get('total_reward', 0) for r in task_res) / len(task_res)
            success_rate = sum(1 for r in task_res if r.get('success', False)) / len(task_res)

            print(f"{task:30} Reward: {avg_reward:6.2f}  Success: {success_rate:5.1%}  Episodes: {len(task_res)}")

        print()
        print(f"Total episodes: {len(results)}")
        print("Results saved to: results/custom_experiment")

    print("\n💡 Next steps:")
    print("  - Modify this script to test different tasks/agents")
    print("  - Use examples/experiments/generate_plots.py to visualize")
    print("  - Compare with other experiments using compare_experiments.py")


if __name__ == "__main__":
    main()

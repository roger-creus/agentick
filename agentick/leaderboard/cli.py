"""CLI commands for leaderboard evaluation."""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

import agentick
from agentick.leaderboard.suites import list_suites
from agentick.tasks.registry import list_tasks

# Load environment variables from .env file
load_dotenv()


def cmd_evaluate(args):
    """Run evaluation using the experiment runner."""
    from agentick.experiments import ExperimentRunner
    from agentick.experiments.config import load_config

    config = load_config(args.config)
    runner = ExperimentRunner(config)
    runner.run()

    print("\n✓ Evaluation complete!")
    return 0


def cmd_list_tasks(args):
    """List all registered tasks."""
    capability = getattr(args, "capability", None)
    difficulty = getattr(args, "difficulty", None)

    tasks = list_tasks(capability=capability, difficulty=difficulty)

    # Build filter description
    filters = []
    if capability:
        filters.append(f"capability={capability}")
    if difficulty:
        filters.append(f"difficulty={difficulty}")

    filter_desc = f" (filtered by {', '.join(filters)})" if filters else ""

    print(f"Registered Tasks ({len(tasks)} total{filter_desc}):")
    for task in tasks:
        print(f"  - {task}")
    return 0


def cmd_list_suites(args):
    """List all benchmark suites."""
    suites = list_suites()
    print(f"Available Benchmark Suites ({len(suites)} total):")
    for suite in suites:
        print(f"  - {suite}")
    return 0


def cmd_info(args):
    """Show detailed information about a task."""
    from agentick.tasks.registry import get_task_class

    try:
        task_class = get_task_class(args.task_name)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1

    # Display task information
    print(f"Task: {task_class.name}")
    print(f"{'=' * 80}")

    # Description
    if hasattr(task_class, "description"):
        print(f"\nDescription:\n  {task_class.description}")

    # Capabilities
    if hasattr(task_class, "capability_tags"):
        print(f"\nCapabilities: {', '.join(task_class.capability_tags)}")

    # Difficulties
    if hasattr(task_class, "difficulty_configs"):
        difficulties = ", ".join(task_class.difficulty_configs.keys())
        print(f"Difficulty Levels: {difficulties}")

    # Create a test instance to show spaces
    try:
        import agentick

        test_env = agentick.make(task_class.name, difficulty="easy")

        print(f"\nObservation Space: {test_env.observation_space}")
        print(f"Action Space: {test_env.action_space}")

        # Action meanings if available
        if hasattr(test_env, "get_action_meanings"):
            meanings = test_env.get_action_meanings()
            print("\nActions:")
            for i, meaning in enumerate(meanings):
                print(f"  {i}: {meaning}")

        test_env.close()
    except Exception as e:
        print(f"\nCould not create test environment: {e}")

    # Baselines if available
    if hasattr(task_class, "baselines"):
        print("\nBaseline Scores:")
        for baseline_name, score in task_class.baselines.items():
            print(f"  {baseline_name}: {score:.2f}")

    print(f"\n{'=' * 80}")
    print("To run this task:")
    print("  import agentick")
    print(f"  env = agentick.make('{task_class.name}', difficulty='medium')")

    return 0


def cmd_experiment_run(args):
    """Run an experiment from config file."""
    from agentick.experiments import ExperimentRunner
    from agentick.experiments.config import load_config

    config = load_config(args.config)
    runner = ExperimentRunner(config)

    runner.run()

    print(f"\n✓ Experiment complete! Results saved to: {config.output_dir}")
    return 0


def cmd_submit_init(args):
    """Initialize a new submission file."""
    from pathlib import Path

    import yaml

    template = {
        "agent_name": "my-agent-v1",
        "author": "Your Name or Organization",
        "description": "A brief description of your agent and how it works",
        "url": "https://github.com/yourusername/yourrepo",
        "tags": ["llm", "text", "zero-shot"],
        "license": "proprietary",
        "open_weights": False,
        "agent_type": "api",
        "observation_mode": "ascii",
        "config": {
            "provider": "openai",
            "model": "gpt-4o",
            "api_key_env": "OPENAI_API_KEY",
        },
        "suites": ["agentick-full-v2"],
        "hardware": "API (OpenAI)",
        "estimated_cost": "$10-50",
        "training_data": "None (zero-shot)",
        "training_compute": "N/A",
    }

    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        print(f"❌ File exists: {output_path}")
        print("Use --force to overwrite")
        return 1

    with open(output_path, "w") as f:
        yaml.dump(template, f, default_flow_style=False)

    print(f"✓ Created submission template: {output_path}")
    print("\nEdit the file and then validate with:")
    print(f"  uv run agentick submit validate {output_path}")

    return 0


def cmd_submit_validate(args):
    """Validate a submission results directory."""
    print("Use 'python scripts/validate_submission.py <results_dir>' to validate submissions.")
    return 0


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Agentick - Universal benchmark for AI agents",
        prog="agentick",
    )
    parser.add_argument("--version", action="version", version=f"agentick {agentick.__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List tasks
    list_tasks_parser = subparsers.add_parser("list-tasks", help="List all registered tasks")
    list_tasks_parser.add_argument("--capability", help="Filter by capability tag")
    list_tasks_parser.add_argument("--difficulty", help="Filter by difficulty level")
    list_tasks_parser.set_defaults(func=cmd_list_tasks)

    # List suites
    list_suites_parser = subparsers.add_parser("list-suites", help="List all benchmark suites")
    list_suites_parser.set_defaults(func=cmd_list_suites)

    # Info command
    info_parser = subparsers.add_parser("info", help="Show detailed information about a task")
    info_parser.add_argument("task_name", help="Name of the task (e.g., GoToGoal-v0)")
    info_parser.set_defaults(func=cmd_info)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation from config")
    eval_parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    eval_parser.set_defaults(func=cmd_evaluate)

    # Experiment command group
    experiment_parser = subparsers.add_parser("experiment", help="Experiment commands")
    experiment_subparsers = experiment_parser.add_subparsers(dest="experiment_command")

    # Experiment run
    exp_run_parser = experiment_subparsers.add_parser("run", help="Run an experiment")
    exp_run_parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    exp_run_parser.add_argument("--output", default="results", help="Output directory")
    exp_run_parser.set_defaults(func=cmd_experiment_run)

    # Submit command group
    submit_parser = subparsers.add_parser("submit", help="Submission commands")
    submit_subparsers = submit_parser.add_subparsers(dest="submit_command")

    # Submit init
    submit_init_parser = submit_subparsers.add_parser("init", help="Create submission template")
    submit_init_parser.add_argument("--output", default="submission.yaml", help="Output file path")
    submit_init_parser.add_argument("--force", action="store_true", help="Overwrite existing file")
    submit_init_parser.set_defaults(func=cmd_submit_init)

    # Submit validate
    submit_validate_parser = submit_subparsers.add_parser(
        "validate", help="Validate submission file"
    )
    submit_validate_parser.add_argument("submission", help="Path to submission YAML")
    submit_validate_parser.set_defaults(func=cmd_submit_validate)

    # Parse and run
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    exit(main())

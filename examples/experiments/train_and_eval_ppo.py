"""Train PPO agent from pixels and evaluate on benchmark tasks.

Trains a separate PPO (CnnPolicy) for each (task, difficulty) pair using
Atari-style preprocessing (84x84 grayscale, frame-stacked). Evaluates
periodically and records videos of the final policy.

Requirements:
    uv sync --extra rl

Usage:
    # Full benchmark from config:
    uv run python examples/experiments/train_and_eval_ppo.py \
        --config examples/experiments/configs/ppo_pixels_dense.yaml

    # Quick single-task test:
    uv run python examples/experiments/train_and_eval_ppo.py \
        --config examples/experiments/configs/ppo_pixels_dense.yaml \
        --tasks GoToGoal-v0 --difficulties easy --timesteps 10000

    # Resume a crashed run:
    uv run python examples/experiments/train_and_eval_ppo.py \
        --config examples/experiments/configs/ppo_pixels_dense.yaml \
        --resume results/ppo_benchmarks/ppo-pixels-dense-300k_20260219_120000
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PPO from pixels on Agentick benchmark tasks",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML experiment config",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Override tasks (e.g. GoToGoal-v0 MazeNavigation-v0)",
    )
    parser.add_argument(
        "--difficulties",
        type=str,
        nargs="+",
        default=None,
        help="Override difficulties (e.g. easy medium)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total_timesteps per task",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to previous run directory to resume from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cpu, cuda, auto)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (all results go here)",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default=None,
        help="Override render mode (e.g. rgb_array)",
    )
    args = parser.parse_args()

    # Check dependencies
    try:
        import stable_baselines3  # noqa: F401
    except ImportError:
        print("stable-baselines3 is required for PPO training.")
        print("Install with: uv sync --extra rl")
        sys.exit(1)

    from agentick.experiments.config import load_config
    from agentick.experiments.training_runner import TrainingBenchmarkRunner

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.tasks:
        config.tasks = args.tasks
    if args.difficulties:
        config.difficulties = args.difficulties
    if args.timesteps and config.training:
        config.training.total_timesteps = args.timesteps
    if args.device and config.training:
        config.training.device = args.device
    if args.render_mode:
        config.render_modes = [args.render_mode]

    # Run
    runner = TrainingBenchmarkRunner(config)
    runner.run(resume_from=args.resume, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
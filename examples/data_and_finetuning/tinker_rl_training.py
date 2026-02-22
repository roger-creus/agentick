"""Tinker RL training pipeline.

Trains an LLM with PPO/REINFORCE on live agentick environment interactions using
Tinker's remote LoRA training infrastructure. Supports multi-task training by
looping over task x difficulty combos (TinkerRLTrainer takes a single task_id).

Evaluates using the same inline NonMarkovianReasoner harness as tinker_sft_training.py.

Requires: pip install tinker; export TINKER_API_KEY="..."

Usage:
    # Quick single-task
    uv run python examples/data_and_finetuning/tinker_rl_training.py \
        --tasks GoToGoal-v0 --difficulties easy --num-episodes 10

    # Multi-task RL (loops over each task/difficulty)
    uv run python examples/data_and_finetuning/tinker_rl_training.py \
        --tasks GoToGoal-v0 MazeNavigation-v0 --eval

    # With warm-start from SFT
    uv run python examples/data_and_finetuning/tinker_rl_training.py \
        --tasks GoToGoal-v0 --difficulties easy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _utils import (
    add_eval_args,
    add_task_args,
    resolve_tasks,
)


def main():
    parser = argparse.ArgumentParser(
        description="Tinker RL pipeline: PPO/REINFORCE on live env interactions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Task args
    add_task_args(parser)

    # Tinker-specific args
    tinker_group = parser.add_argument_group("tinker")
    tinker_group.add_argument(
        "--model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model for Tinker"
    )
    tinker_group.add_argument("--rank", type=int, default=32, help="LoRA rank")
    tinker_group.add_argument(
        "--loss-fn", default="ppo", choices=["ppo", "importance_sampling"], help="RL loss function"
    )
    tinker_group.add_argument(
        "--num-episodes", type=int, default=50, help="Training episodes per task/difficulty"
    )
    tinker_group.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    tinker_group.add_argument("--render-mode", default="language", help="Observation render mode")
    tinker_group.add_argument("--output-dir", default="models/tinker_rl", help="Output directory")

    # Eval args
    add_eval_args(parser)

    args = parser.parse_args()

    # Check Tinker
    try:
        from agentick.training.tinker.rl import TinkerRLTrainer
    except ImportError:
        print("ERROR: Tinker is not installed.")
        print("Install with: pip install tinker")
        print("Then set: export TINKER_API_KEY='your-api-key'")
        return

    tasks = resolve_tasks(args.tasks)

    print("Tinker RL Training Pipeline")
    print("=" * 80)
    print(f"Tasks: {len(tasks)}")
    print(f"Difficulties: {args.difficulties}")
    print(f"Model: {args.model}, rank={args.rank}, loss_fn={args.loss_fn}")
    print(f"Episodes per task/difficulty: {args.num_episodes}")
    print("=" * 80)

    # Train on each task x difficulty combo
    all_metrics: dict[str, dict[str, dict]] = {}
    last_trainer = None

    for task_name in tasks:
        all_metrics[task_name] = {}
        for difficulty in args.difficulties:
            print(f"\n{'=' * 60}")
            print(f"Training: {task_name} @ {difficulty}")
            print(f"{'=' * 60}")

            task_output = Path(args.output_dir) / task_name / difficulty

            try:
                trainer = TinkerRLTrainer(
                    base_model=args.model,
                    task_id=task_name,
                    difficulty=difficulty,
                    rank=args.rank,
                    loss_fn=args.loss_fn,
                    output_dir=str(task_output),
                    render_mode=args.render_mode,
                )

                metrics = trainer.train(
                    num_episodes=args.num_episodes,
                    learning_rate=args.learning_rate,
                )

                final_rewards = metrics["episode_rewards"][-10:]
                final_successes = metrics["episode_successes"][-10:]
                print("\n  Final 10 episodes:")
                print(f"    Avg reward: {np.mean(final_rewards):.3f}")
                print(f"    Success rate: {np.mean(final_successes):.0%}")

                all_metrics[task_name][difficulty] = {
                    "final_reward": float(np.mean(final_rewards)),
                    "final_success": float(np.mean(final_successes)),
                }
                last_trainer = trainer

            except Exception as e:
                print(f"  ERROR: {e}")
                all_metrics[task_name][difficulty] = {"error": str(e)}

    # Evaluate with NonMarkovianReasoner harness
    if args.eval and not args.no_eval and last_trainer is not None:
        print("\n" + "=" * 80)
        print("Evaluating RL-trained agent with NonMarkovianReasoner harness...")
        print("=" * 80)

        # Import the inline harness from tinker_sft_training
        from tinker_sft_training import TinkerNonMarkovianReasoner, evaluate_tinker_agent

        tinker_agent = last_trainer.as_agent()
        reasoner = TinkerNonMarkovianReasoner(
            sampling_client=tinker_agent.sampling_client,
            tokenizer=tinker_agent.tokenizer,
        )

        eval_results = evaluate_tinker_agent(
            agent=reasoner,
            tasks=tasks,
            difficulties=args.difficulties,
            n_episodes=args.eval_episodes,
            render_mode=args.render_mode,
        )

        print("\nEval Summary:")
        for task_name, diff_results in eval_results.items():
            for diff, sr in diff_results.items():
                print(f"  {task_name} @ {diff}: {sr:.0%}")

    # Print training summary
    print("\n" + "=" * 80)
    print("TINKER RL PIPELINE COMPLETE!")
    print("=" * 80)
    print("\nTraining Summary:")
    for task_name, diffs in all_metrics.items():
        for diff, m in diffs.items():
            if "error" in m:
                print(f"  {task_name} @ {diff}: ERROR - {m['error']}")
            else:
                print(
                    f"  {task_name} @ {diff}: "
                    f"reward={m['final_reward']:.3f} success={m['final_success']:.0%}"
                )

    print("\nTip: For better results, warm-start from SFT:")
    print("  1. Run tinker_sft_training.py first")
    print("  2. Then run RL training on the SFT checkpoint")


if __name__ == "__main__":
    main()

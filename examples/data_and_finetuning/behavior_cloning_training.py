"""Behavior Cloning training pipeline.

Trains a CNN (Nature CNN architecture) from pixel observations to actions using
multi-task oracle demonstrations. Pure PyTorch, no HuggingFace dependencies.

Requires: torch, Pillow

Usage:
    # Quick single-task
    uv run python examples/data_and_finetuning/behavior_cloning_training.py \
        --tasks GoToGoal-v0 --difficulties easy --n-episodes 5

    # Full multi-task training
    uv run python examples/data_and_finetuning/behavior_cloning_training.py

    # Train + evaluate
    uv run python examples/data_and_finetuning/behavior_cloning_training.py --eval
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _utils import (
    add_task_args,
    resolve_tasks,
)


def main():
    parser = argparse.ArgumentParser(
        description="Behavior cloning pipeline: collect pixel data -> train CNN -> eval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Task args
    add_task_args(parser)

    # Data collection args
    data_group = parser.add_argument_group("data")
    data_group.add_argument(
        "--n-episodes", type=int, default=20, help="Episodes per task/difficulty"
    )
    data_group.add_argument("--seed-offset", type=int, default=0, help="Starting seed offset")

    # Training args
    train_group = parser.add_argument_group("training")
    train_group.add_argument("--output-dir", default="models/bc", help="Output directory")
    train_group.add_argument("--num-epochs", type=int, default=20, help="Training epochs")
    train_group.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_group.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")

    # Eval args (self-contained, no ExperimentRunner)
    eval_group = parser.add_argument_group("evaluation")
    eval_group.add_argument("--eval", action="store_true", default=False, help="Run evaluation")
    eval_group.add_argument("--no-eval", action="store_true", default=False, help="Skip evaluation")
    eval_group.add_argument(
        "--eval-episodes", type=int, default=5, help="Eval episodes per task/difficulty"
    )

    args = parser.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        print("ERROR: PyTorch is required. Install with: pip install torch")
        return

    tasks = resolve_tasks(args.tasks)

    print("Behavior Cloning Training Pipeline")
    print("=" * 80)
    print(f"Tasks: {len(tasks)} ({', '.join(tasks[:5])}{'...' if len(tasks) > 5 else ''})")
    print(f"Difficulties: {args.difficulties}")
    print(f"Episodes per combo: {args.n_episodes}")
    print("=" * 80)

    # Step 1: Collect multi-task pixel demonstrations
    print("\nStep 1: Collecting oracle pixel demonstrations...")
    import agentick
    from agentick.data import DataCollector
    from agentick.oracles import get_oracle
    from agentick.training.behavior_cloning import BehaviorCloningTrainer

    all_datasets = []
    for task_name in tasks:
        for difficulty in args.difficulties:
            print(f"\n  Collecting {task_name} @ {difficulty}...")
            try:
                env = agentick.make(task_name, difficulty=difficulty, render_mode="rgb_array")
            except Exception as e:
                print(f"    Skipping: {e}")
                continue

            try:
                oracle = get_oracle(task_name, env)
            except Exception as e:
                print(f"    No oracle: {e}")
                env.close()
                continue

            collector = DataCollector(env, oracle, record_modalities=["rgb_array"])
            seeds = range(args.seed_offset, args.seed_offset + args.n_episodes)
            dataset = collector.collect(
                num_episodes=args.n_episodes, seeds=seeds, show_progress=False
            )

            successes = sum(1 for t in dataset.trajectories if t.success)
            print(
                f"    {len(dataset.trajectories)} episodes, "
                f"{successes}/{len(dataset.trajectories)} success"
            )

            all_datasets.append(dataset)
            env.close()

    if not all_datasets:
        print("\nERROR: No data collected.")
        return

    # Save first dataset and merge the rest into it for BC training
    data_path = Path(args.output_dir) / "data"
    first = all_datasets[0]

    # Combine all trajectories into a single dataset
    for ds in all_datasets[1:]:
        first.trajectories.extend(ds.trajectories)

    first.save(data_path, save_pixels=True)
    total_steps = sum(t.length for t in first.trajectories)
    print(f"\nTotal: {len(first.trajectories)} episodes, {total_steps} steps -> {data_path}")

    # Step 2: Train BC model
    print("\nStep 2: Training BC model...")
    trainer = BehaviorCloningTrainer(
        collected_dataset=first,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    metrics = trainer.train()
    print(f"\nTraining complete. Final loss: {metrics.get('final_loss', 'N/A')}")

    # Step 3: Self-contained evaluation
    if args.eval and not args.no_eval:
        print("\nStep 3: Evaluating BC agent...")
        eval_results = trainer.evaluate(
            tasks=tasks,
            num_episodes=args.eval_episodes,
            difficulty=args.difficulties[0],  # BC evaluates on one difficulty at a time
        )

        print("\nEval Results:")
        for task_name, sr in eval_results.items():
            print(f"  {task_name}: {sr:.0%}")

    print("\n" + "=" * 80)
    print("BC PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\nModel saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

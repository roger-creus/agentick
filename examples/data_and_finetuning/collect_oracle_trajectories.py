"""Collect oracle agent demonstrations for supervised fine-tuning.

Comprehensive multi-task oracle data collector using DataCollector + get_oracle().
Collects trajectories across all tasks and difficulties by default, exports to
HuggingFace format, and optionally pushes to Hub.

Usage:
    # All tasks, all difficulties (default)
    uv run python examples/data_and_finetuning/collect_oracle_trajectories.py

    # Specific tasks and difficulties
    uv run python examples/data_and_finetuning/collect_oracle_trajectories.py \
        --tasks GoToGoal-v0 KeyDoorPuzzle-v0 --difficulties easy medium

    # Push to HuggingFace Hub
    uv run python examples/data_and_finetuning/collect_oracle_trajectories.py \
        --push-to-hub user/agentick-oracle-data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script from the examples directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _utils import (
    add_collection_args,
    add_hub_args,
    add_task_args,
    collect_multi_task_data,
    resolve_tasks,
)


def main():
    parser = argparse.ArgumentParser(
        description="Collect oracle demonstrations for fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_task_args(parser)
    add_collection_args(parser)
    add_hub_args(parser)
    parser.add_argument(
        "--export-format",
        default="conversation",
        choices=["conversation", "decision", "trajectory"],
        help="HuggingFace export format",
    )

    args = parser.parse_args()

    tasks = resolve_tasks(args.tasks)
    output_dir = args.output_dir or "trajectories/oracle"

    print("Oracle Trajectory Collection")
    print("=" * 80)
    print(f"Tasks: {len(tasks)} ({', '.join(tasks[:5])}{'...' if len(tasks) > 5 else ''})")
    print(f"Difficulties: {args.difficulties}")
    print(f"Episodes per combo: {args.n_episodes}")
    print(f"Render mode: {args.render_mode}")
    print(f"Export format: {args.export_format}")
    print(f"Output: {output_dir}")
    if args.push_to_hub:
        print(f"Push to Hub: {args.push_to_hub}")
    print("=" * 80)

    combined_path = collect_multi_task_data(
        tasks=tasks,
        difficulties=args.difficulties,
        n_episodes=args.n_episodes,
        render_mode=args.render_mode,
        output_dir=output_dir,
        export_format=args.export_format,
        seed_offset=args.seed_offset,
        push_to_hub=args.push_to_hub,
    )

    print("\n" + "=" * 80)
    print("COLLECTION COMPLETE")
    print("=" * 80)
    print(f"Combined dataset: {combined_path}")
    print("\nNext steps:")
    print("  1. Train with SFT:    python sft_with_trl.py")
    print("  2. Train with Tinker: python tinker_sft_training.py")
    print("  3. Train with BC:     python behavior_cloning_training.py")


if __name__ == "__main__":
    main()

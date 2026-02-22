"""Export trajectories to HuggingFace Datasets format.

Loads previously collected trajectory data and converts to HuggingFace-compatible
JSONL conversation format for supervised fine-tuning with LLMs.

Usage:
    # Default (reads from data/oracle_trajectories)
    uv run python examples/data_and_finetuning/export_to_huggingface.py

    # Custom paths and format
    uv run python examples/data_and_finetuning/export_to_huggingface.py \
        --input-dir trajectories/oracle --output-dir data/hf_export \
        --format conversation

    # Push to Hub
    uv run python examples/data_and_finetuning/export_to_huggingface.py \
        --input-dir trajectories/oracle --push-to-hub user/agentick-data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Export trajectories to HuggingFace format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        default="data/oracle_trajectories",
        help="Directory containing trajectory JSON files",
    )
    parser.add_argument(
        "--output-dir",
        default="data/hf_export",
        help="Output directory for HuggingFace dataset",
    )
    parser.add_argument(
        "--format",
        default="conversation",
        choices=["conversation", "decision", "trajectory"],
        help="Export format",
    )
    parser.add_argument(
        "--push-to-hub",
        default=None,
        metavar="REPO_ID",
        help="Push dataset to HuggingFace Hub",
    )

    args = parser.parse_args()

    print("Export Trajectories to HuggingFace Format")
    print("=" * 80)

    input_dir = Path(args.input_dir)

    # Try loading as CollectedDataset first (has meta.json)
    meta_file = input_dir / "meta.json"
    if meta_file.exists():
        from agentick.data.collector import CollectedDataset

        dataset = CollectedDataset.load(input_dir)
        print(f"Loaded CollectedDataset: {len(dataset)} trajectories")

        output_path = Path(args.output_dir)
        dataset.export_to_huggingface(output_path, format=args.format)
        print(f"Exported to {output_path}")

        if args.push_to_hub:
            url = dataset.push_to_hub(args.push_to_hub, format=args.format)
            print(f"Pushed to {url}")

        print("\n" + "=" * 80)
        print("EXPORT COMPLETE")
        print("=" * 80)
        return

    # Fall back to loading raw trajectory JSON files
    if not input_dir.exists():
        print(f"ERROR: {input_dir} not found.")
        print("Run collect_oracle_trajectories.py first.")
        return

    traj_files = sorted(input_dir.glob("trajectory_*.json"))
    if not traj_files:
        traj_files = sorted(input_dir.glob("*.json"))

    print(f"Found {len(traj_files)} trajectory files")

    trajectories = []
    for traj_file in traj_files:
        with open(traj_file) as f:
            traj_data = json.load(f)
        trajectories.append(traj_data)
        if len(trajectories) <= 5:
            length = traj_data.get("length", len(traj_data.get("steps", [])))
            reward = traj_data.get("total_reward", 0)
            print(f"  Loaded {traj_file.name}: {length} steps, reward={reward:.2f}")

    if len(trajectories) > 5:
        print(f"  ... and {len(trajectories) - 5} more")

    # Export to conversation format
    print(f"\nExporting to {args.format} format...")
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / "conversations.jsonl"

    conversations = []
    for i, traj in enumerate(trajectories):
        conversation: dict = {"id": f"trajectory_{i:04d}", "messages": []}

        conversation["messages"].append(
            {
                "role": "system",
                "content": (
                    "You are an expert agent. "
                    "Given the observation, respond with the best action number."
                ),
            }
        )

        # Handle both step-based and flat formats
        steps = traj.get("steps", [])
        if not steps and "observations" in traj:
            for j, (obs, action) in enumerate(zip(traj["observations"], traj["actions"])):
                conversation["messages"].append(
                    {
                        "role": "user",
                        "content": f"Step {j}: Observation: {obs}",
                    }
                )
                conversation["messages"].append(
                    {
                        "role": "assistant",
                        "content": f"Action: {action}",
                    }
                )
        else:
            for step in steps:
                obs = step.get("observation", step.get("observations", ""))
                if isinstance(obs, dict):
                    obs = obs.get("language", obs.get("ascii", str(obs)))
                conversation["messages"].append(
                    {
                        "role": "user",
                        "content": str(obs),
                    }
                )
                conversation["messages"].append(
                    {
                        "role": "assistant",
                        "content": f"Action: {step.get('action', 0)}",
                    }
                )

        conversations.append(conversation)

    with open(output_file, "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")

    print(f"Exported {len(conversations)} conversations to {output_file}")

    # Push to Hub
    if args.push_to_hub:
        try:
            from datasets import Dataset

            ds = Dataset.from_json(str(output_file))
            ds.push_to_hub(args.push_to_hub)
            print(f"Pushed to https://huggingface.co/datasets/{args.push_to_hub}")
        except Exception as e:
            print(f"WARNING: Failed to push to Hub: {e}")

    print("\n" + "=" * 80)
    print("EXPORT COMPLETE")
    print("=" * 80)
    print(f"Dataset: {output_file}")
    print(f"Total conversations: {len(conversations)}")
    print("\nNext steps:")
    print("  1. Load: datasets.load_dataset('json', data_files='...')")
    print("  2. Train: python sft_with_trl.py")


if __name__ == "__main__":
    main()
